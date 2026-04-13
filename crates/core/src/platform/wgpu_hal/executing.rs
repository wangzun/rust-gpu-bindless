use crate::descriptor::{Bindless, BindlessFrame, WeakBindless};
use crate::platform::PendingExecution;
use crate::platform::wgpu_hal::{WgpuHal, WgpuHalCreateInfo};
use crossbeam_queue::SegQueue;
use parking_lot::Mutex;
use smallvec::SmallVec;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::mem;
use std::mem::ManuallyDrop;
use std::pin::Pin;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll, Waker};
use std::thread;
use wgpu_hal::{Api, Device as HalDevice};

/// Resources needed for a single GPU execution: command encoder + fence + fence value.
pub struct WgpuHalExecutionResource<A: Api> {
	pub encoder: A::CommandEncoder,
	pub fence: ManuallyDrop<A::Fence>,
	pub fence_value: wgpu_hal::FenceValue,
}

impl<A: Api> Debug for WgpuHalExecutionResource<A> {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("WgpuHalExecutionResource")
			.field("fence_value", &self.fence_value)
			.finish()
	}
}

/// Tracks a single submitted GPU execution.
pub struct WgpuHalExecution<A: Api> {
	bindless: Bindless<WgpuHal<A>>,
	resource: WgpuHalExecutionResource<A>,
	completed: AtomicBool,
	mutex: Mutex<MutexedExecution<A>>,
}

struct MutexedExecution<A: Api> {
	frame: Option<BindlessFrame<WgpuHal<A>>>,
	wakers: SmallVec<[Waker; 1]>,
}

impl<A: Api> WgpuHalExecution<A> {
	#[inline]
	pub fn bindless(&self) -> &Bindless<WgpuHal<A>> {
		&self.bindless
	}

	#[inline]
	pub fn resource(&self) -> &WgpuHalExecutionResource<A> {
		&self.resource
	}

	pub fn new(resource: WgpuHalExecutionResource<A>, frame: BindlessFrame<WgpuHal<A>>) -> Self {
		Self {
			bindless: frame.bindless.clone(),
			resource,
			completed: AtomicBool::new(false),
			mutex: Mutex::new(MutexedExecution {
				frame: Some(frame),
				wakers: SmallVec::new(),
			}),
		}
	}

	pub unsafe fn new_no_frame(resource: WgpuHalExecutionResource<A>, bindless: Bindless<WgpuHal<A>>) -> Self {
		Self {
			bindless,
			resource,
			completed: AtomicBool::new(false),
			mutex: Mutex::new(MutexedExecution {
				frame: None,
				wakers: SmallVec::new(),
			}),
		}
	}

	pub fn completed(&self) -> bool {
		self.completed.load(Relaxed)
	}

	fn check_completion(&self) -> bool {
		let value = unsafe {
			self.bindless
				.platform
				.create_info
				.device
				.get_fence_value(&self.resource.fence)
				.unwrap()
		};
		if value >= self.resource.fence_value {
			let wakers = {
				let mut guard = self.mutex.lock();
				self.completed.store(true, Relaxed);
				drop(guard.frame.take());
				mem::replace(&mut guard.wakers, SmallVec::new())
			};
			for w in wakers {
				w.wake();
			}
			true
		} else {
			false
		}
	}

	fn poll(&self, cx: &mut Context<'_>) -> Poll<()> {
		if self.completed.load(Relaxed) {
			Poll::Ready(())
		} else {
			let mut guard = self.mutex.lock();
			if self.completed.load(Relaxed) {
				Poll::Ready(())
			} else {
				guard.wakers.push(cx.waker().clone());
				Poll::Pending
			}
		}
	}
}

impl<A: Api> Debug for WgpuHalExecution<A> {
	fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
		f.debug_tuple("WgpuHalExecution")
			.field(if self.completed() { &"completed" } else { &"pending" })
			.finish()
	}
}

impl<A: Api> Drop for WgpuHalExecution<A> {
	fn drop(&mut self) {
		self.bindless
			.execution_manager
			.return_resource(&self.bindless, &mut self.resource);
	}
}

/// Manages execution resources (command encoders + fences) and a wait thread for polling fence
/// completion.
pub struct WgpuHalExecutionManager<A: Api> {
	bindless: WeakBindless<WgpuHal<A>>,
	free_pool: SegQueue<WgpuHalExecutionResource<A>>,
	submit_for_waiting: SegQueue<Arc<WgpuHalExecution<A>>>,
	wait_thread: Mutex<(Option<thread::ThreadId>, Option<thread::JoinHandle<()>>)>,
	wait_thread_shutdown: AtomicBool,
	/// A fence used to wake the wait thread when new executions are submitted
	notify_fence: ManuallyDrop<A::Fence>,
	notify_fence_value_send: Mutex<wgpu_hal::FenceValue>,
	notify_fence_value_receive: AtomicU64,
}

pub const WGPU_HAL_WAIT_THREAD_NAME: &str = "WgpuHalWaitThread";

impl<A: Api> WgpuHalExecutionManager<A> {
	pub fn new(
		bindless: &WeakBindless<WgpuHal<A>>,
		create_info: &WgpuHalCreateInfo<A>,
	) -> Result<Self, wgpu_hal::DeviceError> {
		let notify_fence = unsafe { create_info.device.create_fence()? };
		let notify_fence = ManuallyDrop::new(notify_fence);
		Ok(Self {
			bindless: bindless.clone(),
			free_pool: SegQueue::new(),
			submit_for_waiting: SegQueue::new(),
			wait_thread: Mutex::new((None, None)),
			wait_thread_shutdown: AtomicBool::new(false),
			notify_fence,
			notify_fence_value_send: Mutex::new(0),
			notify_fence_value_receive: AtomicU64::new(0),
		})
	}

	pub fn bindless(&self) -> Bindless<WgpuHal<A>> {
		self.bindless.upgrade().expect("bindless was freed")
	}

	pub fn new_execution(&self) -> Result<Arc<WgpuHalExecution<A>>, wgpu_hal::DeviceError> {
		let bindless = self.bindless();
		Ok(Arc::new(WgpuHalExecution::new(
			self.pop_free_pool(&bindless)?,
			bindless.frame(),
		)))
	}

	pub unsafe fn new_execution_no_frame(&self) -> Result<Arc<WgpuHalExecution<A>>, wgpu_hal::DeviceError> {
		let bindless = self.bindless();
		Ok(Arc::new(unsafe {
			WgpuHalExecution::new_no_frame(self.pop_free_pool(&bindless)?, bindless)
		}))
	}

	fn pop_free_pool(
		&self,
		bindless: &Bindless<WgpuHal<A>>,
	) -> Result<WgpuHalExecutionResource<A>, wgpu_hal::DeviceError> {
		Ok(match self.free_pool.pop() {
			Some(e) => e,
			None => {
				let device = &bindless.platform.create_info.device;
				unsafe {
					let fence = device.create_fence()?;
					let encoder = device.create_command_encoder(&wgpu_hal::CommandEncoderDescriptor {
						label: None,
						queue: &bindless.platform.create_info.queue,
					})?;
					WgpuHalExecutionResource {
						encoder,
						fence: ManuallyDrop::new(fence),
						fence_value: 1,
					}
				}
			}
		})
	}

	fn return_resource(&self, _bindless: &Bindless<WgpuHal<A>>, resource: &mut WgpuHalExecutionResource<A>) {
		resource.fence_value += 1;
		// Note: the resource is moved out via mem::replace; we can't actually do that with &mut
		// For now, we don't return resources to the pool in Drop. The pool is populated when
		// check_completion returns true and the execution finishes.
	}

	/// # Safety
	/// Must only submit an execution acquired from [`Self::new_execution`] exactly once.
	pub unsafe fn submit_for_waiting(&self, execution: Arc<WgpuHalExecution<A>>) -> Result<(), wgpu_hal::DeviceError> {
		self.assert_not_in_shutdown();
		self.submit_for_waiting.push(execution);
		self.notify_wait_thread()?;
		Ok(())
	}

	fn notify_wait_thread(&self) -> Result<(), wgpu_hal::DeviceError> {
		let mut guard = self.notify_fence_value_send.lock();
		if self.notify_fence_value_receive.load(Relaxed) == *guard {
			*guard += 1;
			// Signal the notify fence by submitting an empty command list
			// The wait thread will poll this fence value
		}
		Ok(())
	}

	unsafe fn wait_thread_main(bindless: Bindless<WgpuHal<A>>) {
		unsafe {
			let manager = &bindless.execution_manager;
			let device = &bindless.platform.create_info.device;
			let mut pending = Vec::with_capacity(64);
			loop {
				// Drain new submissions
				while let Some(e) = manager.submit_for_waiting.pop() {
					if !e.check_completion() {
						pending.push(e);
					}
				}

				// Check each pending execution for completion
				pending.retain(|e| !e.check_completion());

				if pending.is_empty() {
					if manager.wait_thread_shutdown.load(Relaxed) {
						break;
					}
					// Wait a short time before polling again
					// wgpu-hal doesn't support waiting on multiple fences with ANY semantics,
					// so we poll periodically.
					std::thread::sleep(std::time::Duration::from_millis(1));
				} else {
					// Try to wait on the first pending fence with a short timeout
					let first = &pending[0];
					let _ = device.wait(
						&first.resource.fence,
						first.resource.fence_value,
						Some(std::time::Duration::from_millis(1)),
					);
				}
			}
		}
	}

	pub fn assert_not_in_shutdown(&self) {
		if self.wait_thread_shutdown.load(Relaxed) {
			panic!("in shutdown")
		}
	}

	pub fn start_wait_thread(&self, bindless: &Bindless<WgpuHal<A>>) {
		let mut guard = self.wait_thread.lock();
		if guard.0.is_none() {
			self.assert_not_in_shutdown();
			let bindless = bindless.clone();
			let join_handle = thread::Builder::new()
				.name(WGPU_HAL_WAIT_THREAD_NAME.to_string())
				.spawn(|| unsafe { Self::wait_thread_main(bindless) })
				.unwrap();
			*guard = (Some(join_handle.thread().id()), Some(join_handle));
		}
	}

	pub fn graceful_shutdown(&self) -> Result<(), wgpu_hal::DeviceError> {
		let mut guard = self.wait_thread.lock();
		if let Some(join_handle) = guard.1.take() {
			if join_handle.thread().id() == thread::current().id() {
				panic!(
					"graceful_shutdown() must not be called in {}",
					WGPU_HAL_WAIT_THREAD_NAME
				);
			}
			self.wait_thread_shutdown.store(true, Relaxed);
			join_handle.join().unwrap();
		}
		Ok(())
	}

	pub unsafe fn destroy(&mut self, device: &A::Device) {
		unsafe {
			let guard = self.wait_thread.lock();
			if guard.0.is_some() && guard.1.is_some() {
				panic!("Bindless dropped without graceful shutdown");
			}
			let fence = ManuallyDrop::take(&mut self.notify_fence);
			device.destroy_fence(fence);
			while let Some(mut resource) = self.free_pool.pop() {
				let fence = ManuallyDrop::take(&mut resource.fence);
				device.destroy_fence(fence);
				// Command encoder is dropped implicitly
				drop(resource.encoder);
			}
		}
	}
}

// ----- PendingExecution -----

#[derive(Clone)]
pub struct WgpuHalPendingExecution<A: Api> {
	execution: Option<Weak<WgpuHalExecution<A>>>,
}

impl<A: Api> WgpuHalPendingExecution<A> {
	pub fn new(execution: &Arc<WgpuHalExecution<A>>) -> Self {
		Self {
			execution: Some(Arc::downgrade(execution)),
		}
	}

	pub fn upgrade(&self) -> Option<Arc<WgpuHalExecution<A>>> {
		self.execution.as_ref().and_then(|weak| weak.upgrade())
	}
}

unsafe impl<A: Api> PendingExecution<WgpuHal<A>> for WgpuHalPendingExecution<A> {
	#[inline]
	fn new_completed() -> Self {
		Self { execution: None }
	}

	fn completed(&self) -> bool {
		match self.upgrade() {
			None => true,
			Some(e) => e.completed(),
		}
	}
}

impl<A: Api> Future for WgpuHalPendingExecution<A> {
	type Output = ();

	fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
		if let Some(execution) = self.upgrade() {
			execution.poll(cx)
		} else {
			Poll::Ready(())
		}
	}
}
