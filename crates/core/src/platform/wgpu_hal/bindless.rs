use crate::backing::range_set::DescriptorIndexIterator;
use crate::backing::table::DrainFlushQueue;
use crate::descriptor::{
	Bindless, BindlessBufferCreateInfo, BindlessBufferUsage, BindlessImageCreateInfo, BindlessImageUsage,
	BindlessSamplerCreateInfo, BufferAllocationError, BufferInterface, BufferSlot, DescriptorCounts,
	ImageAllocationError, ImageInterface, SamplerAllocationError, SamplerInterface, WeakBindless,
};
use crate::platform::BindlessPlatform;
use crate::platform::wgpu_hal::convert::{
	bindless_image_type_to_wgpu_dimension, bindless_image_type_to_wgpu_view_dimension,
};
use crate::platform::wgpu_hal::executing::{WgpuHalExecutionManager, WgpuHalPendingExecution};
use parking_lot::Mutex;
use presser::Slab;
use rust_gpu_bindless_shaders::descriptor::{
	BINDING_BUFFER, BINDING_SAMPLED_IMAGE, BINDING_SAMPLER, BINDING_STORAGE_IMAGE, BindlessPushConstant, ImageType,
};
use static_assertions::assert_impl_all;
use std::mem::size_of;
use std::num::NonZeroU32;
use std::ops::Deref;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, Ordering};
use thiserror::Error;
use wgpu_hal::{Api, Device as HalDevice};
use wgpu_types as wgt;

/// The wgpu-hal platform, generic over the wgpu-hal `Api` (e.g., `wgpu_hal::vulkan::Api`).
pub struct WgpuHal<A: Api> {
	pub create_info: WgpuHalCreateInfo<A>,
	pub execution_manager: WgpuHalExecutionManager<A>,
}

// assert_impl_all!(Bindless<WgpuHal<wgpu_hal::vulkan::Api>>: Send, Sync);

impl<A: Api> WgpuHal<A> {
	pub fn new(
		create_info: WgpuHalCreateInfo<A>,
		bindless: &WeakBindless<Self>,
	) -> Result<Self, wgpu_hal::DeviceError> {
		Ok(WgpuHal {
			execution_manager: WgpuHalExecutionManager::new(bindless, &create_info)?,
			create_info,
		})
	}
}

/// Initialization info for the wgpu-hal platform. The user is expected to create the `Instance`,
/// `Adapter`, `Device`, and `Queue` externally and pass them in.
pub struct WgpuHalCreateInfo<A: Api> {
	pub device: A::Device,
	pub queue: A::Queue,
	/// Shader stages to use in pipeline layout / descriptor set layout. Defaults to VERTEX | FRAGMENT | COMPUTE.
	pub shader_stages: wgt::ShaderStages,
}

impl<A: Api> Deref for WgpuHal<A> {
	type Target = WgpuHalCreateInfo<A>;

	fn deref(&self) -> &Self::Target {
		&self.create_info
	}
}

impl<A: Api> Drop for WgpuHal<A> {
	fn drop(&mut self) {
		unsafe {
			self.execution_manager.destroy(&self.create_info.device);
		}
	}
}

// ----- Platform resource types -----

pub struct WgpuHalBuffer<A: Api> {
	pub buffer: A::Buffer,
	/// Raw pointer to mapped memory. Null if the buffer is not mapped.
	pub mapped_ptr: std::sync::atomic::AtomicPtr<u8>,
	pub size: u64,
}

pub struct WgpuHalImage<A: Api> {
	pub texture: A::Texture,
	pub texture_view: Option<A::TextureView>,
}

// ----- Descriptor set (bind group layout + bind group) -----

pub struct WgpuHalBindlessDescriptorSet<A: Api> {
	pub bind_group_layout: A::BindGroupLayout,
	pub pipeline_layout: A::PipelineLayout,
	/// The current bind group. Wrapped in a Mutex because `update_descriptor_set` takes `&self`,
	/// and we need to rebuild the bind group on each update.
	pub bind_group: Mutex<Option<A::BindGroup>>,
}

// ----- Error types -----

#[derive(Error)]
pub enum WgpuHalAllocationError {
	#[error("wgpu-hal device error: {0}")]
	Device(#[from] wgpu_hal::DeviceError),
	#[error("wgpu-hal shader error")]
	Shader,
}

impl core::fmt::Debug for WgpuHalAllocationError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Display::fmt(self, f)
	}
}

impl<A: Api> From<WgpuHalAllocationError> for BufferAllocationError<WgpuHal<A>> {
	fn from(value: WgpuHalAllocationError) -> Self {
		BufferAllocationError::Platform(value)
	}
}

impl<A: Api> From<WgpuHalAllocationError> for ImageAllocationError<WgpuHal<A>> {
	fn from(value: WgpuHalAllocationError) -> Self {
		ImageAllocationError::Platform(value)
	}
}

impl<A: Api> From<WgpuHalAllocationError> for SamplerAllocationError<WgpuHal<A>> {
	fn from(value: WgpuHalAllocationError) -> Self {
		SamplerAllocationError::Platform(value)
	}
}

// ----- Mapped buffer slab wrapper -----

/// A presser::Slab implementation wrapping a raw mapped pointer.
pub struct MappedBufferSlab {
	ptr: NonNull<u8>,
	size: usize,
}

unsafe impl Slab for MappedBufferSlab {
	fn base_ptr(&self) -> *const u8 {
		self.ptr.as_ptr()
	}

	fn base_ptr_mut(&mut self) -> *mut u8 {
		self.ptr.as_ptr()
	}

	fn size(&self) -> usize {
		self.size
	}
}

// ----- BindlessPlatform trait implementation -----

unsafe impl<A: Api> BindlessPlatform for WgpuHal<A> {
	type PlatformCreateInfo = WgpuHalCreateInfo<A>;
	type PlatformCreateError = wgpu_hal::DeviceError;
	type Buffer = WgpuHalBuffer<A>;
	type Image = WgpuHalImage<A>;
	type Sampler = A::Sampler;
	type AllocationError = WgpuHalAllocationError;
	type BindlessDescriptorSet = WgpuHalBindlessDescriptorSet<A>;
	type PendingExecution = WgpuHalPendingExecution<A>;

	unsafe fn create_platform(
		create_info: Self::PlatformCreateInfo,
		bindless_cyclic: &WeakBindless<Self>,
	) -> Result<Self, Self::PlatformCreateError> {
		WgpuHal::new(create_info, bindless_cyclic)
	}

	unsafe fn update_after_bind_descriptor_limits(&self) -> DescriptorCounts {
		// wgpu-hal doesn't have a direct query for update-after-bind limits.
		// Return reasonable defaults that should work on most Vulkan implementations.
		DescriptorCounts::REASONABLE_DEFAULTS
	}

	unsafe fn create_descriptor_set(&self, counts: DescriptorCounts) -> Self::BindlessDescriptorSet {
		unsafe {
			let entries = [
				wgt::BindGroupLayoutEntry {
					binding: BINDING_BUFFER,
					visibility: self.shader_stages,
					ty: wgt::BindingType::Buffer {
						ty: wgt::BufferBindingType::Storage { read_only: false },
						has_dynamic_offset: false,
						min_binding_size: None,
					},
					count: NonZeroU32::new(counts.buffers),
				},
				wgt::BindGroupLayoutEntry {
					binding: BINDING_STORAGE_IMAGE,
					visibility: self.shader_stages,
					ty: wgt::BindingType::StorageTexture {
						access: wgt::StorageTextureAccess::ReadWrite,
						format: wgt::TextureFormat::Rgba8Unorm,
						view_dimension: wgt::TextureViewDimension::D2,
					},
					count: NonZeroU32::new(counts.image),
				},
				wgt::BindGroupLayoutEntry {
					binding: BINDING_SAMPLED_IMAGE,
					visibility: self.shader_stages,
					ty: wgt::BindingType::Texture {
						sample_type: wgt::TextureSampleType::Float { filterable: true },
						view_dimension: wgt::TextureViewDimension::D2,
						multisampled: false,
					},
					count: NonZeroU32::new(counts.image),
				},
				wgt::BindGroupLayoutEntry {
					binding: BINDING_SAMPLER,
					visibility: self.shader_stages,
					ty: wgt::BindingType::Sampler(wgt::SamplerBindingType::Filtering),
					count: NonZeroU32::new(counts.samplers),
				},
			];

			let bind_group_layout = self
				.device
				.create_bind_group_layout(&wgpu_hal::BindGroupLayoutDescriptor {
					label: Some("bindless_descriptor_set_layout"),
					flags: wgpu_hal::BindGroupLayoutFlags::PARTIALLY_BOUND,
					entries: &entries,
				})
				.unwrap();

			let pipeline_layout = self
				.device
				.create_pipeline_layout(&wgpu_hal::PipelineLayoutDescriptor {
					label: Some("bindless_pipeline_layout"),
					flags: wgpu_hal::PipelineLayoutFlags::empty(),
					bind_group_layouts: &[Some(&bind_group_layout)],
					immediate_size: size_of::<BindlessPushConstant>() as u32,
				})
				.unwrap();

			WgpuHalBindlessDescriptorSet {
				bind_group_layout,
				pipeline_layout,
				bind_group: Mutex::new(None),
			}
		}
	}

	unsafe fn bindless_initialized(&self, bindless: &Bindless<Self>) {
		self.execution_manager.start_wait_thread(bindless);
	}

	unsafe fn bindless_shutdown(&self, _bindless: &Bindless<Self>) {
		self.execution_manager.graceful_shutdown().unwrap();
	}

	unsafe fn update_descriptor_set(
		&self,
		_set: &Self::BindlessDescriptorSet,
		_buffers: DrainFlushQueue<BufferInterface<Self>>,
		_images: DrainFlushQueue<ImageInterface<Self>>,
		_samplers: DrainFlushQueue<SamplerInterface<Self>>,
	) {
		// In wgpu-hal, bind groups are immutable after creation. Rather than updating in-place,
		// we need to rebuild the entire bind group before the next submission.
		// The actual bind group creation happens at submit time in the recording module.
		//
		// TODO: Track dirty resources and rebuild bind group at submit time.
		// For now, this is a no-op. The bind group must be rebuilt before each command submission.
	}

	unsafe fn destroy_descriptor_set(&self, set: Self::BindlessDescriptorSet) {
		unsafe {
			if let Some(bg) = set.bind_group.into_inner() {
				self.device.destroy_bind_group(bg);
			}
			self.device.destroy_pipeline_layout(set.pipeline_layout);
			self.device.destroy_bind_group_layout(set.bind_group_layout);
		}
	}

	unsafe fn alloc_buffer(
		&self,
		create_info: &BindlessBufferCreateInfo,
		size: u64,
	) -> Result<Self::Buffer, Self::AllocationError> {
		unsafe {
			let usage = create_info.usage.to_wgpu_buffer_uses();
			let buffer = self.device.create_buffer(&wgpu_hal::BufferDescriptor {
				label: Some(create_info.name),
				size,
				usage,
				memory_flags: if create_info.usage.contains(BindlessBufferUsage::MAP_WRITE) {
					wgpu_hal::MemoryFlags::PREFER_COHERENT
				} else if create_info.usage.contains(BindlessBufferUsage::MAP_READ) {
					wgpu_hal::MemoryFlags::PREFER_COHERENT
				} else {
					wgpu_hal::MemoryFlags::empty()
				},
			})?;

			let mapped_ptr = if create_info.usage.is_mappable() {
				let mapping = self.device.map_buffer(&buffer, 0..size)?;
				AtomicPtr::new(mapping.ptr.as_ptr())
			} else {
				AtomicPtr::new(ptr::null_mut())
			};

			Ok(WgpuHalBuffer {
				buffer,
				mapped_ptr,
				size,
			})
		}
	}

	unsafe fn alloc_image<T: ImageType>(
		&self,
		create_info: &BindlessImageCreateInfo<T>,
	) -> Result<Self::Image, Self::AllocationError> {
		unsafe {
			let dimension = bindless_image_type_to_wgpu_dimension::<T>().expect("Unsupported ImageType");
			let usage = create_info.usage.to_wgpu_texture_uses();
			let format = create_info.format;
			let wgpu_format = ash_format_to_wgpu(format);

			let texture = self.device.create_texture(&wgpu_hal::TextureDescriptor {
				label: Some(create_info.name),
				size: create_info.extent.to_wgpu(),
				mip_level_count: create_info.mip_levels,
				sample_count: create_info.samples.to_wgpu(),
				dimension,
				format: wgpu_format,
				usage,
				memory_flags: wgpu_hal::MemoryFlags::empty(),
				view_formats: vec![],
			})?;

			let texture_view = if create_info.usage.has_texture_view() {
				let view_dimension =
					bindless_image_type_to_wgpu_view_dimension::<T>().expect("Unsupported ImageType view dimension");
				let view = self.device.create_texture_view(
					&texture,
					&wgpu_hal::TextureViewDescriptor {
						label: Some(create_info.name),
						format: wgpu_format,
						dimension: view_dimension,
						usage,
						range: wgt::ImageSubresourceRange {
							aspect: wgt::TextureAspect::All,
							base_mip_level: 0,
							mip_level_count: Some(create_info.mip_levels),
							base_array_layer: 0,
							array_layer_count: Some(create_info.array_layers),
						},
					},
				)?;
				Some(view)
			} else {
				None
			};

			Ok(WgpuHalImage { texture, texture_view })
		}
	}

	unsafe fn alloc_sampler(
		&self,
		create_info: &BindlessSamplerCreateInfo,
	) -> Result<Self::Sampler, Self::AllocationError> {
		unsafe {
			let sampler = self.device.create_sampler(&wgpu_hal::SamplerDescriptor {
				label: Some("bindless_sampler"),
				address_modes: [
					create_info.address_mode_u.to_wgpu(),
					create_info.address_mode_v.to_wgpu(),
					create_info.address_mode_w.to_wgpu(),
				],
				mag_filter: create_info.mag_filter.to_wgpu(),
				min_filter: create_info.min_filter.to_wgpu(),
				mipmap_filter: create_info.mipmap_mode.to_wgpu_mipmap(),
				lod_clamp: create_info.min_lod..create_info.max_lod.unwrap_or(f32::MAX),
				compare: None,
				anisotropy_clamp: create_info
					.max_anisotropy
					.map(|a| a.min(16.0).max(1.0) as u16)
					.unwrap_or(1),
				border_color: Some(create_info.border_color.to_wgpu()),
			})?;
			Ok(sampler)
		}
	}

	#[allow(clippy::mut_from_ref)]
	unsafe fn mapped_buffer_to_slab(buffer: &BufferSlot<Self>) -> &mut (impl Slab + '_) {
		let raw = buffer.platform.mapped_ptr.load(Ordering::Relaxed);
		let ptr = NonNull::new(raw).expect("Buffer is not mapped");
		// Safety: the buffer is mapped and we have exclusive access as per trait contract
		let slab = Box::new(MappedBufferSlab {
			ptr,
			size: buffer.size as usize,
		});
		Box::leak(slab)
	}

	unsafe fn destroy_buffers<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		buffers: impl DescriptorIndexIterator<'a, BufferInterface<Self>>,
	) {
		unsafe {
			for (_, buffer) in buffers.into_iter() {
				if !buffer.platform.mapped_ptr.load(Ordering::Relaxed).is_null() {
					self.device.unmap_buffer(&buffer.platform.buffer);
				}
				// Safety: this is a destroy method, the slot will not be accessed again
				let buf = std::ptr::read(&buffer.platform.buffer);
				self.device.destroy_buffer(buf);
			}
		}
	}

	unsafe fn destroy_images<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		images: impl DescriptorIndexIterator<'a, ImageInterface<Self>>,
	) {
		unsafe {
			for (_, image) in images.into_iter() {
				// Safety: this is a destroy method, the slot will not be accessed again
				if let Some(view) = std::ptr::read(&image.platform.texture_view) {
					self.device.destroy_texture_view(view);
				}
				let tex = std::ptr::read(&image.platform.texture);
				self.device.destroy_texture(tex);
			}
		}
	}

	unsafe fn destroy_samplers<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		samplers: impl DescriptorIndexIterator<'a, SamplerInterface<Self>>,
	) {
		unsafe {
			for (_, sampler) in samplers.into_iter() {
				// Safety: this is a destroy method, the slot will not be accessed again
				let s = std::ptr::read(sampler);
				self.device.destroy_sampler(s);
			}
		}
	}
}

/// Convert an ash::vk::Format to a wgpu TextureFormat.
///
/// This is a best-effort mapping for common formats. Extend as needed.
pub fn ash_format_to_wgpu(format: ash::vk::Format) -> wgt::TextureFormat {
	use ash::vk::Format;
	match format {
		Format::R8_UNORM => wgt::TextureFormat::R8Unorm,
		Format::R8_SNORM => wgt::TextureFormat::R8Snorm,
		Format::R8_UINT => wgt::TextureFormat::R8Uint,
		Format::R8_SINT => wgt::TextureFormat::R8Sint,
		Format::R16_UINT => wgt::TextureFormat::R16Uint,
		Format::R16_SINT => wgt::TextureFormat::R16Sint,
		Format::R16_SFLOAT => wgt::TextureFormat::R16Float,
		Format::R8G8_UNORM => wgt::TextureFormat::Rg8Unorm,
		Format::R8G8_SNORM => wgt::TextureFormat::Rg8Snorm,
		Format::R8G8_UINT => wgt::TextureFormat::Rg8Uint,
		Format::R8G8_SINT => wgt::TextureFormat::Rg8Sint,
		Format::R32_UINT => wgt::TextureFormat::R32Uint,
		Format::R32_SINT => wgt::TextureFormat::R32Sint,
		Format::R32_SFLOAT => wgt::TextureFormat::R32Float,
		Format::R16G16_UINT => wgt::TextureFormat::Rg16Uint,
		Format::R16G16_SINT => wgt::TextureFormat::Rg16Sint,
		Format::R16G16_SFLOAT => wgt::TextureFormat::Rg16Float,
		Format::R8G8B8A8_UNORM => wgt::TextureFormat::Rgba8Unorm,
		Format::R8G8B8A8_SRGB => wgt::TextureFormat::Rgba8UnormSrgb,
		Format::R8G8B8A8_SNORM => wgt::TextureFormat::Rgba8Snorm,
		Format::R8G8B8A8_UINT => wgt::TextureFormat::Rgba8Uint,
		Format::R8G8B8A8_SINT => wgt::TextureFormat::Rgba8Sint,
		Format::B8G8R8A8_UNORM => wgt::TextureFormat::Bgra8Unorm,
		Format::B8G8R8A8_SRGB => wgt::TextureFormat::Bgra8UnormSrgb,
		Format::A2B10G10R10_UNORM_PACK32 => wgt::TextureFormat::Rgb10a2Unorm,
		Format::R16G16B16A16_UINT => wgt::TextureFormat::Rgba16Uint,
		Format::R16G16B16A16_SINT => wgt::TextureFormat::Rgba16Sint,
		Format::R16G16B16A16_SFLOAT => wgt::TextureFormat::Rgba16Float,
		Format::R32G32_UINT => wgt::TextureFormat::Rg32Uint,
		Format::R32G32_SINT => wgt::TextureFormat::Rg32Sint,
		Format::R32G32_SFLOAT => wgt::TextureFormat::Rg32Float,
		Format::R32G32B32A32_UINT => wgt::TextureFormat::Rgba32Uint,
		Format::R32G32B32A32_SINT => wgt::TextureFormat::Rgba32Sint,
		Format::R32G32B32A32_SFLOAT => wgt::TextureFormat::Rgba32Float,
		Format::D16_UNORM => wgt::TextureFormat::Depth16Unorm,
		Format::D32_SFLOAT => wgt::TextureFormat::Depth32Float,
		Format::D24_UNORM_S8_UINT => wgt::TextureFormat::Depth24PlusStencil8,
		Format::D32_SFLOAT_S8_UINT => wgt::TextureFormat::Depth32FloatStencil8,
		_ => panic!("Unsupported format for wgpu-hal conversion: {:?}", format),
	}
}
