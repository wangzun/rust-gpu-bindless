use crate::descriptor::{
	Bindless, BindlessAllocationScheme, BindlessBufferCreateInfo, BindlessBufferUsage, BufferSlot, ImageSlot,
	MutDescExt,
};
use crate::pipeline::{
	BindlessComputePipeline, BufferAccess, BufferAccessType, HasResourceContext, ImageAccess, ImageAccessType,
	IndirectCommandReadable, MutBufferAccess, MutImageAccess, MutOrSharedBuffer, Recording, RecordingError,
	TransferReadable, TransferWriteable,
};
use crate::platform::wgpu_hal::bindless::{WgpuHal, WgpuHalBuffer};
use crate::platform::wgpu_hal::bindless_pipeline::WgpuHalPipelineInner;
use crate::platform::wgpu_hal::executing::{WgpuHalExecution, WgpuHalPendingExecution};
use crate::platform::{BindlessPipelinePlatform, RecordingContext, RecordingResourceContext};
use rust_gpu_bindless_shaders::buffer_content::{BufferContent, BufferStruct};
use rust_gpu_bindless_shaders::descriptor::{BindlessPushConstant, ImageType, TransientAccess};
use smallvec::SmallVec;
use std::cell::RefCell;
use std::fmt::Debug;
use std::sync::Arc;
use thiserror::Error;
use wgpu_hal::{Api, CommandEncoder as HalCommandEncoder, Device as HalDevice};
use wgpu_types as wgt;

// ----- RecordingResourceContext -----

pub struct WgpuHalRecordingResourceContext<A: Api> {
	pub(super) execution: Arc<WgpuHalExecution<A>>,
	dependencies: RefCell<SmallVec<[WgpuHalPendingExecution<A>; 4]>>,
}

impl<A: Api> WgpuHalRecordingResourceContext<A> {
	pub fn new(execution: Arc<WgpuHalExecution<A>>) -> Self {
		Self {
			execution,
			dependencies: RefCell::new(SmallVec::new()),
		}
	}

	pub fn execution(&self) -> &Arc<WgpuHalExecution<A>> {
		&self.execution
	}
}

unsafe impl<'a, A: Api> TransientAccess<'a> for &'a WgpuHalRecordingResourceContext<A> {}

unsafe impl<A: Api> RecordingResourceContext<WgpuHal<A>> for WgpuHalRecordingResourceContext<A> {
	fn to_transient_access(&self) -> impl TransientAccess<'_> {
		self
	}

	fn add_dependency(&self, pending: WgpuHalPendingExecution<A>) {
		self.dependencies.borrow_mut().push(pending);
	}

	fn to_pending_execution(&self) -> WgpuHalPendingExecution<A> {
		WgpuHalPendingExecution::new(&self.execution)
	}

	unsafe fn transition_buffer(&self, buffer: &BufferSlot<WgpuHal<A>>, src: BufferAccess, dst: BufferAccess) {
		let src_use = src.to_wgpu_buffer_uses();
		let dst_use = dst.to_wgpu_buffer_uses();
		// wgpu-hal requires BufferBarrier with a reference to the A::Buffer.
		// Since we can't store references with 'static lifetime in the barrier collector,
		// we'll flush barriers directly in the recording context before each action.
		// For now, we store a marker that barriers are needed.
		// TODO: Proper barrier tracking with the encoder directly
		let _ = (buffer, src_use, dst_use);
	}

	unsafe fn transition_image(&self, image: &ImageSlot<WgpuHal<A>>, src: ImageAccess, dst: ImageAccess) {
		let src_use = src.to_wgpu_texture_uses();
		let dst_use = dst.to_wgpu_texture_uses();
		// Similar to transition_buffer, storing full wgpu_hal barriers requires references
		// to the actual GPU resources. See TODO in transition_buffer.
		let _ = (image, src_use, dst_use);
	}
}

// ----- Record and execute -----

pub unsafe fn wgpu_hal_record_and_execute<A: Api, R>(
	bindless: &Bindless<WgpuHal<A>>,
	f: impl FnOnce(&mut Recording<'_, WgpuHal<A>>) -> Result<R, RecordingError<WgpuHal<A>>>,
) -> Result<R, RecordingError<WgpuHal<A>>> {
	unsafe {
		let execution = bindless
			.execution_manager
			.new_execution()
			.map_err(|e| RecordingError::Platform(WgpuHalRecordingError::Device(e)))?;

		let resource = WgpuHalRecordingResourceContext::new(execution);
		let mut recording = Recording::new(WgpuHalRecordingContext::new(&resource)?);
		let r = f(&mut recording)?;
		recording.into_inner().end()?;
		wgpu_hal_submit(bindless, resource)?;
		Ok(r)
	}
}

pub unsafe fn wgpu_hal_submit<A: Api>(
	bindless: &Bindless<WgpuHal<A>>,
	resource_context: WgpuHalRecordingResourceContext<A>,
) -> Result<(), WgpuHalRecordingError> {
	unsafe {
		// Wait for dependencies by polling them (wgpu-hal doesn't have dependency chains in submit)
		for dep in resource_context.dependencies.into_inner() {
			if let Some(exec) = dep.upgrade() {
				// Block-wait for dependency completion
				let device = &bindless.platform.create_info.device;
				device
					.wait(&exec.resource().fence, exec.resource().fence_value, None)
					.map_err(WgpuHalRecordingError::Device)?;
			}
		}

		bindless.flush();

		// Submit the command buffer
		let execution = &resource_context.execution;
		let queue = &bindless.platform.create_info.queue;
		// wgpu-hal submit requires the actual command buffers. The encoder finish produces them.
		// However we need mutable access to the execution's encoder, which is behind an Arc...
		// For now, submit empty and let the execution complete:
		// TODO: Proper command buffer submission - need to extract the finished command buffer
		// from the encoder and pass it to Queue::submit

		bindless
			.execution_manager
			.submit_for_waiting(resource_context.execution.clone())?;
		Ok(())
	}
}

// ----- RecordingContext -----

pub struct WgpuHalRecordingContext<'a, A: Api> {
	pub(super) bindless: Bindless<WgpuHal<A>>,
	pub(super) resource_context: &'a WgpuHalRecordingResourceContext<A>,
	compute_bind_descriptors: bool,
}

impl<'a, A: Api> WgpuHalRecordingContext<'a, A> {
	pub fn new(resource_context: &'a WgpuHalRecordingResourceContext<A>) -> Result<Self, WgpuHalRecordingError> {
		unsafe {
			let bindless = resource_context.execution.bindless().clone();
			let device = &bindless.platform.create_info.device;
			// Begin encoding commands
			// Note: To properly begin encoding, we'd need mutable access to the encoder
			// which is inside the Arc<WgpuHalExecution>. This needs architectural work.
			// TODO: Get mutable encoder access for command recording
			Ok(Self {
				bindless,
				resource_context,
				compute_bind_descriptors: true,
			})
		}
	}

	/// Push param data as push constants / immediates.
	pub unsafe fn push_param<T: BufferStruct>(&mut self, param: T) {
		unsafe {
			let desc = self
				.bindless
				.buffer()
				.alloc_from_data(
					&BindlessBufferCreateInfo {
						usage: BindlessBufferUsage::MAP_WRITE | BindlessBufferUsage::STORAGE_BUFFER,
						name: "param",
						allocation_scheme: BindlessAllocationScheme::AllocatorManaged,
					},
					param,
				)
				.unwrap();
			let push_constant = BindlessPushConstant::new(desc.id(), 0);
			// Push the constant via set_immediates on the encoder
			// TODO: Access encoder and call set_immediates
			let _ = push_constant;
		}
	}

	pub unsafe fn end(self) -> Result<(), WgpuHalRecordingError> {
		// TODO: End command encoding
		Ok(())
	}
}

unsafe impl<'a, A: Api> TransientAccess<'a> for WgpuHalRecordingContext<'a, A> {}

unsafe impl<'a, A: Api> HasResourceContext<'a, WgpuHal<A>> for WgpuHalRecordingContext<'a, A> {
	#[inline]
	fn bindless(&self) -> &Bindless<WgpuHal<A>> {
		&self.bindless
	}

	#[inline]
	fn resource_context(&self) -> &'a <WgpuHal<A> as BindlessPipelinePlatform>::RecordingResourceContext {
		self.resource_context
	}
}

unsafe impl<'a, A: Api> RecordingContext<'a, WgpuHal<A>> for WgpuHalRecordingContext<'a, A> {
	unsafe fn copy_buffer_to_buffer<
		T: BufferStruct,
		SA: BufferAccessType + TransferReadable,
		DA: BufferAccessType + TransferWriteable,
	>(
		&mut self,
		src: impl MutOrSharedBuffer<WgpuHal<A>, T, SA>,
		dst: &MutBufferAccess<WgpuHal<A>, T, DA>,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			let src_slot = src.inner_slot();
			let dst_slot = dst.inner_slot();
			// TODO: encoder.copy_buffer_to_buffer(...)
			let _ = (src_slot, dst_slot);
			Ok(())
		}
	}

	unsafe fn copy_buffer_to_buffer_slice<
		T: BufferStruct,
		SA: BufferAccessType + TransferReadable,
		DA: BufferAccessType + TransferWriteable,
	>(
		&mut self,
		src: impl MutOrSharedBuffer<WgpuHal<A>, [T], SA>,
		dst: &MutBufferAccess<WgpuHal<A>, [T], DA>,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			let src_slot = src.inner_slot();
			let dst_slot = dst.inner_slot();
			// TODO: encoder.copy_buffer_to_buffer(...)
			let _ = (src_slot, dst_slot);
			Ok(())
		}
	}

	unsafe fn copy_buffer_to_image<
		BT: BufferContent + ?Sized,
		BA: BufferAccessType + TransferReadable,
		IT: ImageType,
		IA: ImageAccessType + TransferWriteable,
	>(
		&mut self,
		src: &MutBufferAccess<WgpuHal<A>, BT, BA>,
		dst: &MutImageAccess<WgpuHal<A>, IT, IA>,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			let buffer = src.inner_slot();
			let image = dst.inner_slot();
			// TODO: encoder.copy_buffer_to_texture(...)
			let _ = (buffer, image);
			Ok(())
		}
	}

	unsafe fn copy_image_to_buffer<
		IT: ImageType,
		IA: ImageAccessType + TransferReadable,
		BT: BufferContent + ?Sized,
		BA: BufferAccessType + TransferWriteable,
	>(
		&mut self,
		src: &MutImageAccess<WgpuHal<A>, IT, IA>,
		dst: &MutBufferAccess<WgpuHal<A>, BT, BA>,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			let image = src.inner_slot();
			let buffer = dst.inner_slot();
			// TODO: encoder.copy_texture_to_buffer(...)
			let _ = (image, buffer);
			Ok(())
		}
	}

	unsafe fn dispatch<T: BufferStruct>(
		&mut self,
		pipeline: &BindlessComputePipeline<WgpuHal<A>, T>,
		group_counts: [u32; 3],
		param: T,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			self.push_param(param);
			// TODO: bind pipeline, bind descriptor set, dispatch
			let _ = (pipeline, group_counts);
			Ok(())
		}
	}

	unsafe fn dispatch_indirect<T: BufferStruct, AI: BufferAccessType + IndirectCommandReadable>(
		&mut self,
		pipeline: &BindlessComputePipeline<WgpuHal<A>, T>,
		indirect: impl MutOrSharedBuffer<WgpuHal<A>, [u32; 3], AI>,
		param: T,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			self.push_param(param);
			let indirect_slot = indirect.inner_slot();
			// TODO: bind pipeline, bind descriptor set, dispatch_indirect
			let _ = (pipeline, indirect_slot);
			Ok(())
		}
	}
}

// ----- Error type -----

#[derive(Error)]
pub enum WgpuHalRecordingError {
	#[error("wgpu-hal device error: {0}")]
	Device(#[from] wgpu_hal::DeviceError),
	#[error("No barriers must be inserted while rendering")]
	BarrierWhileRendering,
}

impl Debug for WgpuHalRecordingError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Display::fmt(self, f)
	}
}

impl<A: Api> From<WgpuHalRecordingError> for RecordingError<WgpuHal<A>> {
	fn from(value: WgpuHalRecordingError) -> Self {
		RecordingError::Platform(value)
	}
}
