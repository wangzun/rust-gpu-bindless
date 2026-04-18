#![recursion_limit = "256"]

use anyhow::{Context, Result};
use ash::vk::{
	Fence, FenceCreateInfo, PipelineStageFlags, PresentInfoKHR, SemaphoreCreateInfo, SharingMode, SubmitInfo,
	SwapchainCreateInfoKHR, TimelineSemaphoreSubmitInfo,
};
use bevy::ecs::system::NonSendMut;
use bevy::prelude::*;
use bevy::window::{PrimaryWindow, RawHandleWrapper, WindowResized};
use bevy::winit::WINIT_WINDOWS;
use rust_gpu_bindless_core::backing::table::RcTableSlot;
use rust_gpu_bindless_core::descriptor::{
	Bindless, BindlessImageCreateInfo, BindlessImageUsage, BindlessInstance, DescriptorCounts, Extent, Image2d,
	ImageSlot, MutDesc, MutDescExt, MutImage, SampleCount, SwapchainImageId,
};
use rust_gpu_bindless_core::pipeline::{AccessLock, ImageAccess};
use rust_gpu_bindless_core::platform::ash::Debuggers;
use rust_gpu_bindless_core::platform::ash::{
	Ash, AshAllocationError, AshImage, AshMemoryAllocation, AshPendingExecution, AshSingleGraphicsQueueCreateInfo,
	ash_init_single_graphics_queue,
};
use rust_gpu_bindless_winit::ash::{
	AshSwapchainParams, PresentError, SwapchainImageFormatPreference, ash_enumerate_required_extensions,
};
use std::ops::Deref;
use std::time::Duration;

pub struct BindlessPlugin;

impl Plugin for BindlessPlugin {
	fn build(&self, app: &mut App) {
		app.insert_non_send_resource(BindlessRenderManagerState::default());
		app.add_systems(Update, (init_bindless_manager, mark_swapchain_for_recreate));
	}
}

#[derive(Default)]
pub struct BindlessRenderManagerState {
	manager: Option<BindlessRenderManager>,
}

impl BindlessRenderManagerState {
	pub fn get(&self) -> Option<&BindlessRenderManager> {
		self.manager.as_ref()
	}

	pub fn get_mut(&mut self) -> Option<&mut BindlessRenderManager> {
		self.manager.as_mut()
	}
}

pub struct BindlessRenderManager {
	_instance: BindlessInstance<Ash>,
	bindless: Bindless<Ash>,
	window_entity: Entity,
	surface: ash::vk::SurfaceKHR,
	params: AshSwapchainParams,
	swapchain: ash::vk::SwapchainKHR,
	images: Vec<Option<RcTableSlot>>,
	image_semaphores: Vec<Option<SwapchainSync>>,
	sync_pool: Vec<SwapchainSync>,
	should_recreate: bool,
}

struct SwapchainSync {
	acquire: ash::vk::Semaphore,
	present: ash::vk::Semaphore,
	acquire_fence: ash::vk::Fence,
}

impl SwapchainSync {
	unsafe fn new(bindless: &Bindless<Ash>) -> Result<Self> {
		Ok(Self {
			acquire: unsafe {
				bindless
					.device
					.create_semaphore(&SemaphoreCreateInfo::default(), None)?
			},
			present: unsafe {
				bindless
					.device
					.create_semaphore(&SemaphoreCreateInfo::default(), None)?
			},
			acquire_fence: unsafe { bindless.device.create_fence(&FenceCreateInfo::default(), None)? },
		})
	}

	unsafe fn destroy(&mut self, bindless: &Bindless<Ash>) {
		unsafe {
			bindless.device.destroy_semaphore(self.acquire, None);
			bindless.device.destroy_semaphore(self.present, None);
			bindless.device.destroy_fence(self.acquire_fence, None);
		}
	}
}

impl BindlessRenderManager {
	pub fn bindless(&self) -> &Bindless<Ash> {
		&self.bindless
	}

	pub fn window_entity(&self) -> Entity {
		self.window_entity
	}

	pub fn params(&self) -> &AshSwapchainParams {
		&self.params
	}

	pub fn force_recreate(&mut self) {
		self.should_recreate = true;
	}

	pub fn acquire_image(&mut self, timeout: Option<Duration>) -> Result<MutDesc<Ash, MutImage<Image2d>>> {
		unsafe {
			const RECREATE_ATTEMPTS: u32 = 10;
			let swapchain_ext = self.bindless.extensions.swapchain();

			for _ in 0..RECREATE_ATTEMPTS {
				if self.should_recreate {
					self.should_recreate = false;
					self.bindless.device.device_wait_idle()?;
					let (swapchain, images) = Self::create_swapchain(
						&self.bindless,
						self.window_entity,
						self.surface,
						&self.params,
						self.swapchain,
					)?;
					swapchain_ext.destroy_swapchain(self.swapchain, None);
					self.swapchain = swapchain;

					assert_eq!(self.images.len(), images.len());
					for (index, image) in images.into_iter().enumerate() {
						drop(self.images[index].replace(image));
					}
				}

				let sync = self
					.sync_pool
					.pop()
					.map_or_else(|| SwapchainSync::new(&self.bindless), Ok)?;

				match swapchain_ext.acquire_next_image(
					self.swapchain,
					timeout.map(|value| value.as_nanos() as u64).unwrap_or(!0),
					sync.acquire,
					Fence::null(),
				) {
					Ok((id, suboptimal)) => {
						if suboptimal {
							self.should_recreate = true;
						}

						let image = self.images[id as usize]
							.take()
							.with_context(|| format!("Swapchain image {id} was already checked out"))?;

						let device = &self.bindless.device;
						let execution = self.bindless.execution_manager.new_execution_no_frame()?;
						{
							let queue = self.bindless.queue.lock();
							device.queue_submit(
								*queue,
								&[SubmitInfo::default()
									.wait_semaphores(&[sync.acquire])
									.wait_dst_stage_mask(&[PipelineStageFlags::ALL_COMMANDS])
									.signal_semaphores(&[execution.resource().semaphore])
									.push_next(
										&mut TimelineSemaphoreSubmitInfo::default()
											.wait_semaphore_values(&[0])
											.signal_semaphore_values(&[execution.resource().timeline_value]),
									)],
								sync.acquire_fence,
							)?;
						}

						let pending = AshPendingExecution::new(&execution);
						self.bindless.execution_manager.submit_for_waiting(execution)?;
						if let Some(previous) = self.image_semaphores[id as usize].replace(sync) {
							device.wait_for_fences(&[previous.acquire_fence], true, !0)?;
							device.reset_fences(&[previous.acquire_fence])?;
							self.sync_pool.push(previous);
						}

						let desc = MutDesc::<Ash, MutImage<Image2d>>::new(image, pending);
						desc.inner_slot().access_lock.unlock(ImageAccess::Present);
						return Ok(desc);
					}
					Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => {
						self.sync_pool.push(sync);
						self.should_recreate = true;
					}
					Err(err) => {
						self.sync_pool.push(sync);
						return Err(err.into());
					}
				}
			}

			panic!(
				"looped {RECREATE_ATTEMPTS} times trying to acquire a swapchain image and failed repeatedly"
			);
		}
	}

	pub fn present_image(&mut self, image: MutDesc<Ash, MutImage<Image2d>>) -> Result<(), PresentError> {
		unsafe {
			let id = {
				let slot = image.inner_slot();
				if !slot.usage.contains(BindlessImageUsage::SWAPCHAIN) {
					return Err(PresentError::NotASwapchainImage(slot.debug_name.clone()));
				}

				let id = slot
					.swapchain_image_id
					.get()
					.expect("swapchain usage without a swapchain image id");
				if self.images[id as usize].is_some() {
					return Err(PresentError::SwapchainIdOccupied(id));
				}

				let access = slot.access_lock.try_lock()?;
				if !matches!(access, ImageAccess::Present | ImageAccess::General) {
					slot.access_lock.unlock(access);
					return Err(PresentError::IncorrectLayout(slot.debug_name.clone(), access));
				}

				id
			};

			let device = &self.bindless.device;
			let swapchain_ext = self.bindless.extensions.swapchain();
			let (rc_slot, last) = image.into_inner();
			let semaphore = self.image_semaphores[id as usize]
				.as_ref()
				.expect("missing swapchain semaphore for acquired image")
				.present;
			let dependency: Option<_> = last.upgrade_ash_resource();

			let suboptimal = {
				let queue = self.bindless.queue.lock();
				device.queue_submit(
					*queue,
					&[SubmitInfo::default()
						.wait_semaphores(dependency.as_ref().map(|entry| entry.resource().semaphore).as_slice())
						.wait_dst_stage_mask(
							dependency
								.as_ref()
								.map(|_| PipelineStageFlags::ALL_COMMANDS)
								.as_slice(),
						)
						.signal_semaphores(&[semaphore])
						.push_next(
							&mut TimelineSemaphoreSubmitInfo::default()
								.wait_semaphore_values(
									dependency
										.as_ref()
										.map(|entry| entry.resource().timeline_value)
										.as_slice(),
								)
								.signal_semaphore_values(&[0]),
						)],
					Fence::null(),
				)?;

				match swapchain_ext.queue_present(
					*queue,
					&PresentInfoKHR::default()
						.wait_semaphores(&[semaphore])
						.swapchains(&[self.swapchain])
						.image_indices(&[id]),
				) {
					Ok(value) => Ok(value),
					Err(ash::vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
					Err(err) => Err(err),
				}?
			};

			if suboptimal {
				self.should_recreate = true;
			}

			self.images[id as usize].replace(rc_slot);
			Ok(())
		}
	}

	unsafe fn create_swapchain(
		bindless: &Bindless<Ash>,
		window_entity: Entity,
		surface: ash::vk::SurfaceKHR,
		params: &AshSwapchainParams,
		old_swapchain: ash::vk::SwapchainKHR,
	) -> Result<(ash::vk::SwapchainKHR, Vec<RcTableSlot>)> {
		let extent = current_window_extent(bindless, window_entity, surface)?;
		let swapchain_ext = bindless.extensions.swapchain();
		let swapchain = unsafe {
			swapchain_ext.create_swapchain(
				&swapchain_create_info(params, surface, extent).old_swapchain(old_swapchain),
				None,
			)
		}
			.map_err(AshAllocationError::from)?;
		let images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }
			.map_err(AshAllocationError::from)?
			.into_iter()
			.enumerate()
			.map(|(id, image)| unsafe {
				Self::create_swapchain_image(bindless, params, extent, id as u32, image)
			})
			.collect::<Result<Vec<_>, _>>()?;
		Ok((swapchain, images))
	}

	unsafe fn create_swapchain_image(
		bindless: &Bindless<Ash>,
		params: &AshSwapchainParams,
		extent: Extent,
		id: u32,
		image: ash::vk::Image,
	) -> Result<RcTableSlot, rust_gpu_bindless_core::descriptor::ImageAllocationError<Ash>> {
		let debug_name = format!("Swapchain Image {id}");
		unsafe { bindless.set_debug_object_name(image, &debug_name) }
			.map_err(AshAllocationError::from)?;
		let image_view = unsafe {
			bindless.create_image_view(
				image,
				&BindlessImageCreateInfo::<Image2d> {
					format: params.format,
					extent,
					mip_levels: 1,
					array_layers: 1,
					samples: SampleCount::default(),
					usage: params.image_usage,
					allocation_scheme: Default::default(),
					name: &debug_name,
					_phantom: Default::default(),
				},
			)?
		};
		let image = unsafe { bindless.image().alloc_slot::<Image2d>(ImageSlot {
			platform: AshImage {
				image,
				image_view,
				allocation: AshMemoryAllocation::none(),
			},
			usage: params.image_usage,
			format: params.format,
			extent,
			mip_levels: 1,
			array_layers: 1,
			access_lock: AccessLock::new_locked(),
			debug_name,
			swapchain_image_id: SwapchainImageId::new(id),
		})? };
		Ok(image.into_inner().0)
	}
}

impl Drop for BindlessRenderManager {
	fn drop(&mut self) {
		unsafe {
			self.bindless.device.device_wait_idle().unwrap();
			for sync in self.sync_pool.iter_mut() {
				sync.destroy(&self.bindless);
			}
			for sync in self.image_semaphores.iter_mut().flatten() {
				sync.destroy(&self.bindless);
			}
			self.bindless
				.extensions
				.swapchain()
				.destroy_swapchain(self.swapchain, None);
			self.bindless.extensions.surface().destroy_surface(self.surface, None);
		}
	}
}

fn init_bindless_manager(
	mut state: NonSendMut<BindlessRenderManagerState>,
	windows: Query<(Entity, &RawHandleWrapper), With<PrimaryWindow>>,
) {
	if state.manager.is_some() {
		return;
	}

	let Ok((window_entity, handle)) = windows.single() else {
		return;
	};

	let extensions = ash_enumerate_required_extensions(handle.get_display_handle())
		.expect("failed to enumerate required Vulkan instance extensions for the Bevy window");
	let instance = unsafe {
		BindlessInstance::<Ash>::new(
			ash_init_single_graphics_queue(AshSingleGraphicsQueueCreateInfo {
				instance_extensions: extensions,
				extensions: &[ash::khr::swapchain::NAME],
				debug: debugger(),
				..AshSingleGraphicsQueueCreateInfo::default()
			})
			.expect("failed to initialize the ash graphics device"),
			DescriptorCounts::REASONABLE_DEFAULTS,
		)
	};
	let bindless = instance.deref().clone();

	let surface = unsafe {
		ash_window::create_surface(
			&bindless.entry,
			&bindless.instance,
			handle.get_display_handle(),
			handle.get_window_handle(),
			None,
		)
		.expect("failed to create a Vulkan surface for the Bevy primary window")
	};

	let params = unsafe {
		AshSwapchainParams::automatic_best(
			&bindless,
			&surface,
			BindlessImageUsage::COLOR_ATTACHMENT,
			SwapchainImageFormatPreference::SRGB,
		)
		.expect("failed to select swapchain parameters for the Bevy primary window")
	};
	let (swapchain, images) = unsafe {
		BindlessRenderManager::create_swapchain(&bindless, window_entity, surface, &params, ash::vk::SwapchainKHR::null())
			.expect("failed to create the initial Bevy swapchain")
	};
	let images = images.into_iter().map(Some).collect::<Vec<_>>();
	let image_semaphores = (0..images.len()).map(|_| None).collect();

	state.manager = Some(BindlessRenderManager {
		_instance: instance,
		bindless,
		window_entity,
		surface,
		params,
		swapchain,
		images,
		image_semaphores,
		sync_pool: Vec::new(),
		should_recreate: false,
	});
}

fn mark_swapchain_for_recreate(
	mut state: NonSendMut<BindlessRenderManagerState>,
	mut resized: MessageReader<WindowResized>,
) {
	let Some(manager) = state.get_mut() else {
		return;
	};

	for event in resized.read() {
		if event.window == manager.window_entity {
			manager.force_recreate();
		}
	}
}

fn current_window_extent(bindless: &Bindless<Ash>, window_entity: Entity, surface: ash::vk::SurfaceKHR) -> Result<Extent> {
	WINIT_WINDOWS.with_borrow(|winit_windows| {
		let window = winit_windows
			.get_window(window_entity)
			.context("missing primary Bevy winit window for bindless swapchain recreation")?;
		let window_size = window.inner_size();
		let surface_ext = bindless.extensions.surface();
		let capabilities = unsafe {
			surface_ext.get_physical_device_surface_capabilities(bindless.physical_device, surface)?
		};
		let min = capabilities.min_image_extent;
		let max = capabilities.max_image_extent;
		Ok(Extent::from([
			u32::clamp(window_size.width, min.width, max.width),
			u32::clamp(window_size.height, min.height, max.height),
		]))
	})
}

fn swapchain_create_info(
	params: &AshSwapchainParams,
	surface: ash::vk::SurfaceKHR,
	extent: Extent,
) -> SwapchainCreateInfoKHR<'_> {
	SwapchainCreateInfoKHR::default()
		.surface(surface)
		.min_image_count(params.image_count)
		.image_format(params.format)
		.image_color_space(params.colorspace)
		.image_extent(extent.into())
		.image_array_layers(1)
		.image_usage(params.image_usage.to_ash_image_usage_flags())
		.image_sharing_mode(SharingMode::EXCLUSIVE)
		.pre_transform(params.pre_transform)
		.composite_alpha(params.composite_alpha)
		.present_mode(params.present_mode)
		.clipped(true)
}

/// The global debugger configuration used by the Bevy integration.
pub fn debugger() -> Debuggers {
	// Validation layer does not yet support timeline semaphores properly, leading to many false positives.
	// On Linux RADV, gpu assisted validation may still crash during graphics pipeline creation.
	Debuggers::Validation
}
