use crate::backing::range_set::{DescriptorIndexIterator, DescriptorIndexRangeSet};
use crate::backing::table::DrainFlushQueue;
use crate::descriptor::{
	Bindless, BindlessBufferCreateInfo, BindlessBufferUsage, BindlessImageCreateInfo, BindlessImageUsage,
	BindlessSamplerCreateInfo, BufferAllocationError, BufferInterface, BufferSlot, DescriptorCounts,
	ImageAllocationError, ImageInterface, SamplerAllocationError, SamplerInterface, WeakBindless,
};
use crate::platform::BindlessPlatform;
use crate::platform::ash::image_format::FormatExt;
use crate::platform::ash::{
	AshExecutionManager, AshPendingExecution, bindless_image_type_to_vk_image_type,
	bindless_image_type_to_vk_image_view_type,
};
use ash::ext::{debug_utils, mesh_shader};
use ash::khr::{surface, swapchain};
use ash::prelude::VkResult;
use ash::vk::{
	ComponentMapping, DebugUtilsObjectNameInfoEXT, DescriptorBindingFlags, DescriptorBufferInfo, DescriptorImageInfo,
	DescriptorPool, DescriptorPoolCreateFlags, DescriptorPoolCreateInfo, DescriptorPoolSize, DescriptorSet,
	DescriptorSetAllocateInfo, DescriptorSetLayout, DescriptorSetLayoutBindingFlagsCreateInfo,
	DescriptorSetLayoutCreateFlags, DescriptorSetLayoutCreateInfo, DescriptorType, Handle, ImageLayout,
	ImageSubresourceRange, ImageTiling, ImageViewCreateInfo, LOD_CLAMP_NONE, PhysicalDeviceProperties2,
	PhysicalDeviceVulkan12Properties, PipelineCache, PipelineLayout, PipelineLayoutCreateInfo, PushConstantRange,
	SamplerCreateInfo, ShaderStageFlags, SharingMode, WriteDescriptorSet,
};
use gpu_allocator::AllocationError;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, Allocator};
use parking_lot::lock_api::MutexGuard;
use parking_lot::{Mutex, RawMutex};
use presser::Slab;
use rangemap::RangeSet;
use rust_gpu_bindless_shaders::descriptor::{
	BINDING_BUFFER, BINDING_SAMPLED_IMAGE, BINDING_SAMPLER, BINDING_STORAGE_IMAGE, BindlessPushConstant, ImageType,
};
use static_assertions::assert_impl_all;
use std::cell::UnsafeCell;
use std::ffi::CString;
use std::mem::size_of;
use std::ops::Deref;
use thiserror::Error;

pub struct Ash {
	pub create_info: AshCreateInfo,
	pub execution_manager: AshExecutionManager,
}
assert_impl_all!(Bindless<Ash>: Send, Sync);

impl Ash {
	pub fn new(create_info: AshCreateInfo, bindless: &WeakBindless<Self>) -> VkResult<Self> {
		Ok(Ash {
			execution_manager: AshExecutionManager::new(bindless, &create_info)?,
			create_info,
		})
	}

	pub unsafe fn set_debug_object_name(&self, handle: impl Handle, name: &str) -> VkResult<()> {
		unsafe {
			if let Some(debug_marker) = self.extensions.debug_utils.as_ref() {
				debug_marker.set_debug_utils_object_name(
					&DebugUtilsObjectNameInfoEXT::default()
						.object_handle(handle)
						.object_name(&CString::new(name).unwrap()),
				)?;
			}
			Ok(())
		}
	}

	pub unsafe fn create_image_view<T: ImageType>(
		&self,
		image: ash::vk::Image,
		create_info: &BindlessImageCreateInfo<T>,
	) -> Result<Option<ash::vk::ImageView>, <Ash as BindlessPlatform>::AllocationError> {
		unsafe {
			let image_view_type = bindless_image_type_to_vk_image_view_type::<T>().expect("Unsupported ImageType");
			Ok(if create_info.usage.has_image_view() {
				let image_view = self.device.create_image_view(
					&ImageViewCreateInfo::default()
						.image(image)
						.view_type(image_view_type)
						.format(create_info.format)
						.components(ComponentMapping::default()) // identity
						.subresource_range(ImageSubresourceRange {
							aspect_mask: create_info.format.aspect(),
							base_mip_level: 0,
							level_count: create_info.mip_levels,
							base_array_layer: 0,
							layer_count: create_info.array_layers,
						}),
					None,
				)?;
				self.set_debug_object_name(image_view, create_info.name)?;
				Some(image_view)
			} else {
				None
			})
		}
	}

	pub unsafe fn create_ray_tracing_descriptor_set(&self, counts: DescriptorCounts) -> AshBindlessRayDescriptorSet {
		unsafe {
			let bindings = [
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_BUFFER)
					.descriptor_type(DescriptorType::STORAGE_BUFFER)
					.descriptor_count(counts.buffers)
					.stage_flags(self.shader_stages),
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_STORAGE_IMAGE)
					.descriptor_type(DescriptorType::STORAGE_IMAGE)
					.descriptor_count(counts.image)
					.stage_flags(self.shader_stages),
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_SAMPLED_IMAGE)
					.descriptor_type(DescriptorType::SAMPLED_IMAGE)
					.descriptor_count(counts.image)
					.stage_flags(self.shader_stages),
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_SAMPLER)
					.descriptor_type(DescriptorType::SAMPLER)
					.descriptor_count(counts.samplers)
					.stage_flags(self.shader_stages),
			];
			let binding_flags = [DescriptorBindingFlags::UPDATE_AFTER_BIND
				| DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
				| DescriptorBindingFlags::PARTIALLY_BOUND; 4];
			assert_eq!(bindings.len(), binding_flags.len());

			// ray query pipeline
			let ray_query_bindings = [ash::vk::DescriptorSetLayoutBinding::default()
				.binding(0)
				.descriptor_type(DescriptorType::ACCELERATION_STRUCTURE_KHR)
				.descriptor_count(counts.buffers)
				.stage_flags(self.shader_stages)];
			let ray_query_binding_flags = [DescriptorBindingFlags::UPDATE_AFTER_BIND
				| DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
				| DescriptorBindingFlags::PARTIALLY_BOUND; 1];
			assert_eq!(ray_query_bindings.len(), ray_query_binding_flags.len());

			let set_layout = self
				.device
				.create_descriptor_set_layout(
					&DescriptorSetLayoutCreateInfo::default()
						.flags(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
						.bindings(&bindings)
						.push_next(
							&mut DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags),
						),
					None,
				)
				.unwrap();

			// bindings + ray query bindings
			let pool_sizes = bindings
				.iter()
				.chain(ray_query_bindings.iter())
				.map(|b| {
					DescriptorPoolSize::default()
						.ty(b.descriptor_type)
						.descriptor_count(b.descriptor_count)
				})
				.collect::<Vec<_>>();
			let pool = self
				.device
				.create_descriptor_pool(
					&DescriptorPoolCreateInfo::default()
						.flags(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
						.pool_sizes(&pool_sizes)
						.max_sets(1),
					None,
				)
				.unwrap();

			let set = self
				.device
				.allocate_descriptor_sets(
					&DescriptorSetAllocateInfo::default()
						.descriptor_pool(pool)
						.set_layouts(&[set_layout]),
				)
				.unwrap()
				.into_iter()
				.next()
				.unwrap();

			let ray_query_descriptor_set_layout = self
				.device
				.create_descriptor_set_layout(
					&DescriptorSetLayoutCreateInfo::default()
						.flags(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
						.bindings(&ray_query_bindings)
						.push_next(
							&mut DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags),
						),
					None,
				)
				.unwrap();

			let pipeline_layout = self
				.device
				.create_pipeline_layout(
					&PipelineLayoutCreateInfo::default()
						.set_layouts(&[set_layout, ray_query_descriptor_set_layout])
						.push_constant_ranges(&[PushConstantRange {
							offset: 0,
							size: size_of::<BindlessPushConstant>() as u32,
							stage_flags: self.shader_stages,
						}]),
					None,
				)
				.unwrap();

			let ray_query_descriptor_set = self
				.device
				.allocate_descriptor_sets(
					&DescriptorSetAllocateInfo::default()
						.descriptor_pool(pool)
						.set_layouts(&[ray_query_descriptor_set_layout]),
				)
				.unwrap()
				.into_iter()
				.next()
				.unwrap();

			AshBindlessRayDescriptorSet {
				pipeline_layout,
				pool,
				set_layout,
				set,
				ray_query_descriptor_set_layout,
				ray_query_descriptor_set,
			}
		}
	}

	pub unsafe fn build_bottom_level_acceleration_structure(
		&self,
		vertex_buffer: ash::vk::Buffer,
		vertex_count: u32,
		vertex_stride: u64,
		index_buffer: ash::vk::Buffer,
		index_count: u32,
	) -> Result<Option<ash::vk::DeviceAddress>, <Ash as BindlessPlatform>::AllocationError> {
		let ray_device = self.extensions.ray_device.as_ref().expect("Ray tracing not supported");

		// build acceleration structure geometry
		let geometry = ash::vk::AccelerationStructureGeometryKHR::default()
			.geometry_type(ash::vk::GeometryTypeKHR::TRIANGLES)
			.geometry(ash::vk::AccelerationStructureGeometryDataKHR {
				triangles: ash::vk::AccelerationStructureGeometryTrianglesDataKHR::default()
					.vertex_format(ash::vk::Format::R32G32B32_SFLOAT)
					.vertex_data(ash::vk::DeviceOrHostAddressConstKHR {
						device_address: unsafe {
							self.device.get_buffer_device_address(
								&ash::vk::BufferDeviceAddressInfo::default().buffer(vertex_buffer),
							)
						},
					})
					.vertex_stride(vertex_stride)
					.max_vertex(vertex_count)
					.index_type(ash::vk::IndexType::UINT32)
					.index_data(ash::vk::DeviceOrHostAddressConstKHR {
						device_address: unsafe {
							self.device.get_buffer_device_address(
								&ash::vk::BufferDeviceAddressInfo::default().buffer(index_buffer),
							)
						},
					}),
			});

		let primitive_count = index_count / 3;
		let build_range_info = ash::vk::AccelerationStructureBuildRangeInfoKHR::default()
			.first_vertex(0)
			.primitive_count(primitive_count)
			.primitive_offset(0)
			.transform_offset(0);

		let geometries = [geometry];
		let mut build_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
			.flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
			.geometries(&geometries)
			.mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
			.ty(ash::vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL);

		let mut size_info = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
		unsafe {
			ray_device.get_acceleration_structure_build_sizes(
				ash::vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&build_info,
				&[primitive_count],
				&mut size_info,
			);
		}

		// Create acceleration structure buffer and allocate memory
		let bottom_as_buffer = unsafe {
			self.device
				.create_buffer(
					&ash::vk::BufferCreateInfo::default()
						.size(size_info.acceleration_structure_size)
						.usage(
							ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
								| ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
						)
						.sharing_mode(SharingMode::EXCLUSIVE),
					None,
				)
				.unwrap()
		};
		let bottom_as_buffer_req = unsafe { self.device.get_buffer_memory_requirements(bottom_as_buffer) };
		let bottom_as_buffer_alloc = self
			.memory_allocator()
			.allocate(&AllocationCreateDesc {
				requirements: bottom_as_buffer_req,
				name: "bottom_level_as_buffer",
				location: gpu_allocator::MemoryLocation::GpuOnly,
				allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(bottom_as_buffer),
				linear: true,
			})
			.unwrap();
		unsafe {
			self.device
				.bind_buffer_memory(
					bottom_as_buffer,
					bottom_as_buffer_alloc.memory(),
					bottom_as_buffer_alloc.offset(),
				)
				.unwrap();
		}

		// Create acceleration structure
		let as_create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
			.ty(build_info.ty)
			.size(size_info.acceleration_structure_size)
			.buffer(bottom_as_buffer)
			.offset(0);
		let bottom_as = unsafe { ray_device.create_acceleration_structure(&as_create_info, None) }.unwrap();
		build_info.dst_acceleration_structure = bottom_as;

		// Create scratch buffer and allocate memory
		let scratch_buffer = unsafe {
			self.device
				.create_buffer(
					&ash::vk::BufferCreateInfo::default()
						.size(size_info.build_scratch_size)
						.usage(
							ash::vk::BufferUsageFlags::STORAGE_BUFFER
								| ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
						)
						.sharing_mode(SharingMode::EXCLUSIVE),
					None,
				)
				.unwrap()
		};
		let scratch_buffer_req = unsafe { self.device.get_buffer_memory_requirements(scratch_buffer) };
		let scratch_alloc = self
			.memory_allocator()
			.allocate(&AllocationCreateDesc {
				requirements: scratch_buffer_req,
				name: "as_scratch_buffer",
				location: gpu_allocator::MemoryLocation::GpuOnly,
				allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(scratch_buffer),
				linear: true,
			})
			.unwrap();
		unsafe {
			self.device
				.bind_buffer_memory(scratch_buffer, scratch_alloc.memory(), scratch_alloc.offset())
				.unwrap();
		}

		build_info.scratch_data = ash::vk::DeviceOrHostAddressKHR {
			device_address: unsafe {
				self.device
					.get_buffer_device_address(&ash::vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer))
			},
		};

		// Create command pool and record build commands
		let command_pool = unsafe {
			self.device
				.create_command_pool(
					&ash::vk::CommandPoolCreateInfo::default()
						.flags(ash::vk::CommandPoolCreateFlags::TRANSIENT)
						.queue_family_index(self.queue_family_index),
					None,
				)
				.unwrap()
		};

		let build_command_buffer = unsafe {
			self.device
				.allocate_command_buffers(
					&ash::vk::CommandBufferAllocateInfo::default()
						.command_buffer_count(1)
						.command_pool(command_pool)
						.level(ash::vk::CommandBufferLevel::PRIMARY),
				)
				.unwrap()[0]
		};

		unsafe {
			self.device
				.begin_command_buffer(
					build_command_buffer,
					&ash::vk::CommandBufferBeginInfo::default()
						.flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
				)
				.unwrap();

			ray_device.cmd_build_acceleration_structures(build_command_buffer, &[build_info], &[&[build_range_info]]);
			self.device.end_command_buffer(build_command_buffer).unwrap();

			let queue = self.queue.lock();
			self.device
				.queue_submit(
					*queue,
					&[ash::vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
					ash::vk::Fence::null(),
				)
				.expect("queue submit failed.");
			self.device.queue_wait_idle(*queue).unwrap();
			drop(queue);

			// Cleanup temporary resources
			self.device.free_command_buffers(command_pool, &[build_command_buffer]);
			self.device.destroy_command_pool(command_pool, None);
			self.device.destroy_buffer(scratch_buffer, None);
			self.memory_allocator().free(scratch_alloc).unwrap();
		}

		// Return acceleration structure device address
		let as_addr_info =
			ash::vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(bottom_as);
		unsafe {
			Ok(Some(
				ray_device.get_acceleration_structure_device_address(&as_addr_info),
			))
		}
	}

	pub unsafe fn build_top_level_acceleration_structure(
		&self,
		instance_addresses: &[ash::vk::DeviceAddress],
		transforms: &[glam::Mat4],
		instance_count: u32,
	) -> Result<Option<ash::vk::AccelerationStructureKHR>, <Ash as BindlessPlatform>::AllocationError> {
		let ray_device = self.extensions.ray_device.as_ref().expect("Ray tracing not supported");

		if instance_count == 0 {
			return Ok(None);
		}

		assert_eq!(instance_addresses.len(), instance_count as usize);
		assert_eq!(transforms.len(), instance_count as usize);

		// Build instance data (row-major 3x4 transform matrix)
		let instances = instance_addresses
			.iter()
			.zip(transforms.iter())
			.map(
				|(&instance_addr, transform)| ash::vk::AccelerationStructureInstanceKHR {
					transform: ash::vk::TransformMatrixKHR {
						matrix: [
							transform.x_axis.x,
							transform.y_axis.x,
							transform.z_axis.x,
							transform.w_axis.x,
							transform.x_axis.y,
							transform.y_axis.y,
							transform.z_axis.y,
							transform.w_axis.y,
							transform.x_axis.z,
							transform.y_axis.z,
							transform.z_axis.z,
							transform.w_axis.z,
						],
					},
					instance_custom_index_and_mask: ash::vk::Packed24_8::new(0, 0xff),
					instance_shader_binding_table_record_offset_and_flags: ash::vk::Packed24_8::new(
						0,
						ash::vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE.as_raw() as u8,
					),
					acceleration_structure_reference: ash::vk::AccelerationStructureReferenceKHR {
						device_handle: instance_addr,
					},
				},
			)
			.collect::<Vec<_>>();

		// Create instance buffer (CPU-to-GPU so we can map and upload)
		let instance_buffer_size =
			(std::mem::size_of::<ash::vk::AccelerationStructureInstanceKHR>() * instances.len()) as u64;
		let instance_buffer = unsafe {
			self.device.create_buffer(
				&ash::vk::BufferCreateInfo::default()
					.size(instance_buffer_size)
					.usage(
						ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR
							| ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
					)
					.sharing_mode(SharingMode::EXCLUSIVE),
				None,
			)?
		};
		let instance_buffer_req = unsafe { self.device.get_buffer_memory_requirements(instance_buffer) };
		let instance_buffer_alloc = self.memory_allocator().allocate(&AllocationCreateDesc {
			requirements: instance_buffer_req,
			name: "top_level_as_instance_buffer",
			location: gpu_allocator::MemoryLocation::CpuToGpu,
			allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(instance_buffer),
			linear: true,
		})?;
		unsafe {
			self.device.bind_buffer_memory(
				instance_buffer,
				instance_buffer_alloc.memory(),
				instance_buffer_alloc.offset(),
			)?;
		}

		// Upload instance data via mapped pointer from gpu-allocator
		unsafe {
			let mapped = instance_buffer_alloc
				.mapped_ptr()
				.expect("Instance buffer allocation is not host-visible")
				.as_ptr() as *mut ash::vk::AccelerationStructureInstanceKHR;
			mapped.copy_from_nonoverlapping(instances.as_ptr(), instances.len());
		}

		let instance_buffer_address = unsafe {
			self.device
				.get_buffer_device_address(&ash::vk::BufferDeviceAddressInfo::default().buffer(instance_buffer))
		};

		// Build geometry for instances
		let geometry = ash::vk::AccelerationStructureGeometryKHR::default()
			.geometry_type(ash::vk::GeometryTypeKHR::INSTANCES)
			.geometry(ash::vk::AccelerationStructureGeometryDataKHR {
				instances: ash::vk::AccelerationStructureGeometryInstancesDataKHR::default()
					.array_of_pointers(false)
					.data(ash::vk::DeviceOrHostAddressConstKHR {
						device_address: instance_buffer_address,
					}),
			});

		let build_range_info = ash::vk::AccelerationStructureBuildRangeInfoKHR::default()
			.first_vertex(0)
			.primitive_count(instance_count)
			.primitive_offset(0)
			.transform_offset(0);

		let geometries = [geometry];
		let mut build_info = ash::vk::AccelerationStructureBuildGeometryInfoKHR::default()
			.flags(ash::vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE)
			.geometries(&geometries)
			.mode(ash::vk::BuildAccelerationStructureModeKHR::BUILD)
			.ty(ash::vk::AccelerationStructureTypeKHR::TOP_LEVEL);

		// Query required sizes
		let mut size_info = ash::vk::AccelerationStructureBuildSizesInfoKHR::default();
		unsafe {
			ray_device.get_acceleration_structure_build_sizes(
				ash::vk::AccelerationStructureBuildTypeKHR::DEVICE,
				&build_info,
				&[instance_count],
				&mut size_info,
			);
		}

		// Create TLAS buffer and allocate memory
		let top_as_buffer = unsafe {
			self.device.create_buffer(
				&ash::vk::BufferCreateInfo::default()
					.size(size_info.acceleration_structure_size)
					.usage(
						ash::vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
							| ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
					)
					.sharing_mode(SharingMode::EXCLUSIVE),
				None,
			)?
		};
		let top_as_buffer_req = unsafe { self.device.get_buffer_memory_requirements(top_as_buffer) };
		let top_as_buffer_alloc = self.memory_allocator().allocate(&AllocationCreateDesc {
			requirements: top_as_buffer_req,
			name: "top_level_as_buffer",
			location: gpu_allocator::MemoryLocation::GpuOnly,
			allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(top_as_buffer),
			linear: true,
		})?;
		unsafe {
			self.device.bind_buffer_memory(
				top_as_buffer,
				top_as_buffer_alloc.memory(),
				top_as_buffer_alloc.offset(),
			)?;
		}

		// Create the top-level acceleration structure
		let as_create_info = ash::vk::AccelerationStructureCreateInfoKHR::default()
			.ty(build_info.ty)
			.size(size_info.acceleration_structure_size)
			.buffer(top_as_buffer)
			.offset(0);
		let top_as = unsafe { ray_device.create_acceleration_structure(&as_create_info, None)? };
		build_info.dst_acceleration_structure = top_as;

		// Create scratch buffer and allocate memory
		let scratch_buffer = unsafe {
			self.device.create_buffer(
				&ash::vk::BufferCreateInfo::default()
					.size(size_info.build_scratch_size)
					.usage(ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
					.sharing_mode(SharingMode::EXCLUSIVE),
				None,
			)?
		};
		let scratch_buffer_req = unsafe { self.device.get_buffer_memory_requirements(scratch_buffer) };
		let scratch_alloc = self.memory_allocator().allocate(&AllocationCreateDesc {
			requirements: scratch_buffer_req,
			name: "tlas_scratch_buffer",
			location: gpu_allocator::MemoryLocation::GpuOnly,
			allocation_scheme: gpu_allocator::vulkan::AllocationScheme::DedicatedBuffer(scratch_buffer),
			linear: true,
		})?;
		unsafe {
			self.device
				.bind_buffer_memory(scratch_buffer, scratch_alloc.memory(), scratch_alloc.offset())?;
		}

		build_info.scratch_data = ash::vk::DeviceOrHostAddressKHR {
			device_address: unsafe {
				self.device
					.get_buffer_device_address(&ash::vk::BufferDeviceAddressInfo::default().buffer(scratch_buffer))
			},
		};

		// Create command pool and record build commands
		let command_pool = unsafe {
			self.device.create_command_pool(
				&ash::vk::CommandPoolCreateInfo::default()
					.flags(ash::vk::CommandPoolCreateFlags::TRANSIENT)
					.queue_family_index(self.queue_family_index),
				None,
			)?
		};

		let build_command_buffer = unsafe {
			self.device.allocate_command_buffers(
				&ash::vk::CommandBufferAllocateInfo::default()
					.command_buffer_count(1)
					.command_pool(command_pool)
					.level(ash::vk::CommandBufferLevel::PRIMARY),
			)?[0]
		};

		unsafe {
			self.device.begin_command_buffer(
				build_command_buffer,
				&ash::vk::CommandBufferBeginInfo::default().flags(ash::vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
			)?;

			ray_device.cmd_build_acceleration_structures(build_command_buffer, &[build_info], &[&[build_range_info]]);
			self.device.end_command_buffer(build_command_buffer)?;

			let queue = self.queue.lock();
			self.device
				.queue_submit(
					*queue,
					&[ash::vk::SubmitInfo::default().command_buffers(&[build_command_buffer])],
					ash::vk::Fence::null(),
				)
				.expect("queue submit failed.");
			self.device.queue_wait_idle(*queue)?;
			drop(queue);

			// Cleanup temporary resources
			self.device.free_command_buffers(command_pool, &[build_command_buffer]);
			self.device.destroy_command_pool(command_pool, None);
			self.device.destroy_buffer(scratch_buffer, None);
			self.memory_allocator().free(scratch_alloc).unwrap();
			self.device.destroy_buffer(instance_buffer, None);
			self.memory_allocator().free(instance_buffer_alloc).unwrap();
		}

		Ok(Some(top_as))
	}

	pub unsafe fn update_acceleration_structure_descriptor_set(
		&self,
		set: &AshBindlessRayDescriptorSet,
		top_as: ash::vk::AccelerationStructureKHR,
	) {
		unsafe {
			let accel_structs = [top_as];
			let mut accel_info =
				ash::vk::WriteDescriptorSetAccelerationStructureKHR::default().acceleration_structures(&accel_structs);

			let accel_write = WriteDescriptorSet::default()
				.dst_set(set.ray_query_descriptor_set)
				.dst_binding(0)
				.dst_array_element(0)
				.descriptor_type(DescriptorType::ACCELERATION_STRUCTURE_KHR)
				.push_next(&mut accel_info);

			self.device.update_descriptor_sets(&[accel_write], &[]);
		}
	}
}

impl Deref for Ash {
	type Target = AshCreateInfo;

	fn deref(&self) -> &Self::Target {
		&self.create_info
	}
}

impl Drop for Ash {
	fn drop(&mut self) {
		unsafe {
			// This device_wait_idle is needed as some semaphores may still be in use. Likely due to being waited
			// upon, as that does not hold a strong ref on the semaphore.
			self.device.device_wait_idle().unwrap();
			self.execution_manager.destroy(&self.create_info.device);
		}
	}
}

pub struct AshCreateInfo {
	pub entry: ash::Entry,
	pub instance: ash::Instance,
	pub physical_device: ash::vk::PhysicalDevice,
	pub device: ash::Device,
	pub memory_allocator: Option<Mutex<Allocator>>,
	pub shader_stages: ShaderStageFlags,
	pub queue_family_index: u32,
	pub queue: Mutex<ash::vk::Queue>,
	pub cache: Option<PipelineCache>,
	pub extensions: AshExtensions,
	pub destroy: Option<AshDestroyFn>,
}

pub type AshDestroyFn = Box<dyn FnOnce(&mut AshCreateInfo) + Send + Sync>;

#[derive(Default)]
#[non_exhaustive]
pub struct AshExtensions {
	pub debug_utils: Option<debug_utils::Device>,
	pub mesh_shader: Option<mesh_shader::Device>,
	pub surface: Option<surface::Instance>,
	pub swapchain: Option<swapchain::Device>,
	pub ray_device: Option<ash::khr::acceleration_structure::Device>,
	pub ray_query_enabled: bool,
}

impl AshExtensions {
	pub fn debug_utils(&self) -> &debug_utils::Device {
		self.debug_utils.as_ref().expect("missing ext_debug_utils")
	}

	pub fn mesh_shader(&self) -> &mesh_shader::Device {
		self.mesh_shader.as_ref().expect("missing ext_mesh_shader")
	}

	pub fn surface(&self) -> &surface::Instance {
		self.surface.as_ref().expect("missing khr_surface")
	}

	pub fn swapchain(&self) -> &swapchain::Device {
		self.swapchain.as_ref().expect("missing khr_swapchain")
	}
}

impl AshCreateInfo {
	pub fn memory_allocator(&self) -> MutexGuard<'_, RawMutex, Allocator> {
		self.memory_allocator.as_ref().unwrap().lock()
	}
}

impl Drop for AshCreateInfo {
	fn drop(&mut self) {
		if let Some(destroy) = self.destroy.take() {
			destroy(self);
		}
	}
}

/// Wraps gpu-allocator's MemoryAllocation to be able to [`Option::take`] it on drop, but saving the enum flag byte
/// with [`MaybeUninit`]
///
/// # Safety
/// UnsafeCell: Required to gain mutable access where it is safe to do so, see safety of interface methods.
/// MaybeUninit: The Allocation is effectively always initialized, it only becomes uninit after taking it during drop.
#[derive(Debug)]
pub struct AshMemoryAllocation(UnsafeCell<Option<Allocation>>);

impl AshMemoryAllocation {
	/// Create a `AshMemoryAllocation` from a gpu-allocator Allocation
	///
	/// # Safety
	/// You must [`Self::take`] the Allocation and deallocate manually before dropping self
	pub unsafe fn new(allocation: Allocation) -> Self {
		Self(UnsafeCell::new(Some(allocation)))
	}

	/// Create a `AshMemoryAllocation` without a backing allocation
	pub fn none() -> Self {
		Self(UnsafeCell::new(None))
	}

	/// Get exclusive mutable access to the `AshMemoryAllocation`
	///
	/// # Safety
	/// You must ensure you have exclusive mutable access to the Allocation
	#[allow(clippy::mut_from_ref)]
	pub unsafe fn get_mut(&self) -> &mut Allocation {
		unsafe { (*self.0.get()).as_mut().unwrap() }
	}

	/// Take the `AshMemoryAllocation`
	pub fn take(&self) -> Option<Allocation> {
		unsafe { (*self.0.get()).take() }
	}
}

/// Safety: MemoryAllocation is safety Send and Sync, will only uninit on drop
unsafe impl Send for AshMemoryAllocation {}
unsafe impl Sync for AshMemoryAllocation {}

pub struct AshBuffer {
	pub buffer: ash::vk::Buffer,
	pub allocation: AshMemoryAllocation,
}

pub struct AshImage {
	pub image: ash::vk::Image,
	pub image_view: Option<ash::vk::ImageView>,
	pub allocation: AshMemoryAllocation,
}

#[derive(Copy, Clone, Debug)]
pub struct AshBindlessRayDescriptorSet {
	pub pipeline_layout: PipelineLayout,
	pub pool: DescriptorPool,
	pub set_layout: DescriptorSetLayout,
	pub set: DescriptorSet,
	pub ray_query_descriptor_set_layout: DescriptorSetLayout,
	pub ray_query_descriptor_set: DescriptorSet,
}

#[derive(Copy, Clone, Debug)]
pub struct AshBindlessDescriptorSet {
	pub pipeline_layout: PipelineLayout,
	pub set_layout: DescriptorSetLayout,
	pub pool: DescriptorPool,
	pub set: DescriptorSet,
}

impl Deref for AshBindlessDescriptorSet {
	type Target = DescriptorSet;

	fn deref(&self) -> &Self::Target {
		&self.set
	}
}

#[derive(Error)]
pub enum AshAllocationError {
	#[error("VkResult: {0}")]
	Vk(#[from] ash::vk::Result),
	#[error("gpu-allocator Error: {0}")]
	Allocation(#[from] AllocationError),
}

impl core::fmt::Debug for AshAllocationError {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		core::fmt::Display::fmt(self, f)
	}
}

impl From<AshAllocationError> for BufferAllocationError<Ash> {
	fn from(value: AshAllocationError) -> Self {
		BufferAllocationError::Platform(value)
	}
}

impl From<AshAllocationError> for ImageAllocationError<Ash> {
	fn from(value: AshAllocationError) -> Self {
		ImageAllocationError::Platform(value)
	}
}

impl From<AshAllocationError> for SamplerAllocationError<Ash> {
	fn from(value: AshAllocationError) -> Self {
		SamplerAllocationError::Platform(value)
	}
}

unsafe impl BindlessPlatform for Ash {
	type PlatformCreateInfo = AshCreateInfo;
	type PlatformCreateError = ash::vk::Result;
	type Buffer = AshBuffer;
	type Image = AshImage;
	type Sampler = ash::vk::Sampler;
	type AllocationError = AshAllocationError;
	type BindlessDescriptorSet = AshBindlessDescriptorSet;
	type PendingExecution = AshPendingExecution;

	unsafe fn create_platform(
		create_info: Self::PlatformCreateInfo,
		bindless_cyclic: &WeakBindless<Self>,
	) -> VkResult<Self> {
		Ash::new(create_info, bindless_cyclic)
	}

	unsafe fn update_after_bind_descriptor_limits(&self) -> DescriptorCounts {
		unsafe {
			let mut vulkan12properties = PhysicalDeviceVulkan12Properties::default();
			let mut properties2 = PhysicalDeviceProperties2::default().push_next(&mut vulkan12properties);
			self.instance
				.get_physical_device_properties2(self.physical_device, &mut properties2);
			DescriptorCounts {
				buffers: vulkan12properties.max_descriptor_set_update_after_bind_storage_buffers,
				image: u32::min(
					vulkan12properties.max_per_stage_descriptor_update_after_bind_storage_images,
					vulkan12properties.max_descriptor_set_update_after_bind_sampled_images,
				),
				samplers: vulkan12properties.max_descriptor_set_update_after_bind_samplers,
			}
		}
	}

	unsafe fn create_descriptor_set(&self, counts: DescriptorCounts) -> Self::BindlessDescriptorSet {
		unsafe {
			let bindings = [
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_BUFFER)
					.descriptor_type(DescriptorType::STORAGE_BUFFER)
					.descriptor_count(counts.buffers)
					.stage_flags(self.shader_stages),
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_STORAGE_IMAGE)
					.descriptor_type(DescriptorType::STORAGE_IMAGE)
					.descriptor_count(counts.image)
					.stage_flags(self.shader_stages),
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_SAMPLED_IMAGE)
					.descriptor_type(DescriptorType::SAMPLED_IMAGE)
					.descriptor_count(counts.image)
					.stage_flags(self.shader_stages),
				ash::vk::DescriptorSetLayoutBinding::default()
					.binding(BINDING_SAMPLER)
					.descriptor_type(DescriptorType::SAMPLER)
					.descriptor_count(counts.samplers)
					.stage_flags(self.shader_stages),
			];
			let binding_flags = [DescriptorBindingFlags::UPDATE_AFTER_BIND
				| DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
				| DescriptorBindingFlags::PARTIALLY_BOUND; 4];
			assert_eq!(bindings.len(), binding_flags.len());

			let set_layout = self
				.device
				.create_descriptor_set_layout(
					&DescriptorSetLayoutCreateInfo::default()
						.flags(DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
						.bindings(&bindings)
						.push_next(
							&mut DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags),
						),
					None,
				)
				.unwrap();

			let pipeline_layout = self
				.device
				.create_pipeline_layout(
					&PipelineLayoutCreateInfo::default()
						.set_layouts(&[set_layout])
						.push_constant_ranges(&[PushConstantRange {
							offset: 0,
							size: size_of::<BindlessPushConstant>() as u32,
							stage_flags: self.shader_stages,
						}]),
					None,
				)
				.unwrap();

			// bindings + ray query bindings
			let pool_sizes = bindings
				.iter()
				.map(|b| {
					DescriptorPoolSize::default()
						.ty(b.descriptor_type)
						.descriptor_count(b.descriptor_count)
				})
				.collect::<Vec<_>>();
			let pool = self
				.device
				.create_descriptor_pool(
					&DescriptorPoolCreateInfo::default()
						.flags(DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
						.pool_sizes(&pool_sizes)
						.max_sets(1),
					None,
				)
				.unwrap();

			let set = self
				.device
				.allocate_descriptor_sets(
					&DescriptorSetAllocateInfo::default()
						.descriptor_pool(pool)
						.set_layouts(&[set_layout]),
				)
				.unwrap()
				.into_iter()
				.next()
				.unwrap();

			AshBindlessDescriptorSet {
				pipeline_layout,
				set_layout,
				pool,
				set,
			}
		}
	}

	unsafe fn bindless_initialized(&self, bindless: &Bindless<Self>) {
		self.execution_manager.start_wait_semaphore_thread(bindless);
	}

	unsafe fn bindless_shutdown(&self, _bindless: &Bindless<Self>) {
		self.execution_manager.graceful_shutdown().unwrap();
	}

	unsafe fn update_descriptor_set(
		&self,
		set: &Self::BindlessDescriptorSet,
		mut buffers: DrainFlushQueue<BufferInterface<Self>>,
		mut images: DrainFlushQueue<ImageInterface<Self>>,
		mut samplers: DrainFlushQueue<SamplerInterface<Self>>,
	) {
		unsafe {
			let (buffer_table, buffers) = buffers.into_inner();
			let mut storage_buffers = DescriptorIndexRangeSet::new(buffer_table, RangeSet::new());
			for buffer_id in buffers {
				let buffer = buffer_table.get_slot_unchecked(buffer_id);
				if buffer.usage.contains(BindlessBufferUsage::STORAGE_BUFFER) {
					storage_buffers.insert(buffer_id);
				}
			}

			let buffer_infos = storage_buffers
				.iter()
				.map(|(_, buffer)| {
					DescriptorBufferInfo::default()
						.buffer(buffer.buffer)
						.offset(0)
						.range(buffer.size)
				})
				.collect::<Vec<_>>();
			let mut buffer_info_index = 0;
			let buffers = storage_buffers.iter_ranges().map(|(range, _)| {
				let count = range.end.to_usize() - range.start.to_usize();
				WriteDescriptorSet::default()
					.dst_set(set.set)
					.dst_binding(BINDING_BUFFER)
					.descriptor_type(DescriptorType::STORAGE_BUFFER)
					.dst_array_element(range.start.to_u32())
					.descriptor_count(count as u32)
					.buffer_info({
						let buffer_info_start = buffer_info_index;
						buffer_info_index += count;
						&buffer_infos[buffer_info_start..buffer_info_start + count]
					})
			});

			let (image_table, images) = images.into_inner();
			let mut storage_images = DescriptorIndexRangeSet::new(image_table, RangeSet::new());
			let mut sampled_images = DescriptorIndexRangeSet::new(image_table, RangeSet::new());
			for image_id in images {
				let image = image_table.get_slot_unchecked(image_id);
				if image.usage.contains(BindlessImageUsage::STORAGE) {
					storage_images.insert(image_id);
				}
				if image.usage.contains(BindlessImageUsage::SAMPLED) {
					sampled_images.insert(image_id);
				}
			}

			let storage_image_infos = storage_images
				.iter()
				.map(|(_, storage_image)| {
					DescriptorImageInfo::default()
						.image_view(*storage_image.image_view.as_ref().unwrap())
						.image_layout(ImageLayout::GENERAL)
				})
				.collect::<Vec<_>>();
			let mut storage_image_info_index = 0;
			let storage_images = storage_images.iter_ranges().map(|(range, _)| {
				let count = range.end.to_usize() - range.start.to_usize();
				WriteDescriptorSet::default()
					.dst_set(set.set)
					.dst_binding(BINDING_STORAGE_IMAGE)
					.descriptor_type(DescriptorType::STORAGE_IMAGE)
					.dst_array_element(range.start.to_u32())
					.descriptor_count(count as u32)
					.image_info({
						let storage_image_info_start = storage_image_info_index;
						storage_image_info_index += count;
						&storage_image_infos[storage_image_info_start..storage_image_info_start + count]
					})
			});

			let sampled_image_infos = sampled_images
				.iter()
				.map(|(_, sampled_image)| {
					DescriptorImageInfo::default()
						.image_view(*sampled_image.image_view.as_ref().unwrap())
						.image_layout(ImageLayout::SHADER_READ_ONLY_OPTIMAL)
				})
				.collect::<Vec<_>>();
			let mut sampled_image_info_index = 0;
			let sampled_images = sampled_images.iter_ranges().map(|(range, _)| {
				let count = range.end.to_usize() - range.start.to_usize();
				WriteDescriptorSet::default()
					.dst_set(set.set)
					.dst_binding(BINDING_SAMPLED_IMAGE)
					.descriptor_type(DescriptorType::SAMPLED_IMAGE)
					.dst_array_element(range.start.to_u32())
					.descriptor_count(count as u32)
					.image_info({
						let sampled_image_info_start = sampled_image_info_index;
						sampled_image_info_index += count;
						&sampled_image_infos[sampled_image_info_start..sampled_image_info_start + count]
					})
			});

			let samplers = samplers.into_range_set();
			let sampler_infos = samplers
				.iter()
				.map(|(_, sampler)| DescriptorImageInfo::default().sampler(*sampler))
				.collect::<Vec<_>>();
			let mut sampler_info_index = 0;
			let samplers = samplers.iter_ranges().map(|(range, _)| {
				let count = range.end.to_usize() - range.start.to_usize();
				WriteDescriptorSet::default()
					.dst_set(set.set)
					.dst_binding(BINDING_SAMPLER)
					.descriptor_type(DescriptorType::SAMPLER)
					.dst_array_element(range.start.to_u32())
					.descriptor_count(count as u32)
					.image_info({
						let sampler_info_start = sampler_info_index;
						sampler_info_index += count;
						&sampler_infos[sampler_info_start..sampler_info_start + count]
					})
			});

			let writes = buffers
				.chain(storage_images)
				.chain(sampled_images)
				.chain(samplers)
				.collect::<Vec<_>>();
			self.device.update_descriptor_sets(&writes, &[]);
		}
	}

	unsafe fn destroy_descriptor_set(&self, set: Self::BindlessDescriptorSet) {
		unsafe {
			// descriptor sets allocated from pool are freed implicitly
			self.device.destroy_descriptor_pool(set.pool, None);
			self.device.destroy_pipeline_layout(set.pipeline_layout, None);
			self.device.destroy_descriptor_set_layout(set.set_layout, None);
		}
	}

	unsafe fn alloc_buffer(
		&self,
		create_info: &BindlessBufferCreateInfo,
		size: u64,
	) -> Result<Self::Buffer, Self::AllocationError> {
		unsafe {
			let buffer = self.device.create_buffer(
				&ash::vk::BufferCreateInfo::default()
					.usage(create_info.usage.to_ash_buffer_usage_flags())
					.size(size)
					.sharing_mode(SharingMode::EXCLUSIVE),
				None,
			)?;
			self.set_debug_object_name(buffer, create_info.name)?;
			let requirements = self.device.get_buffer_memory_requirements(buffer);
			let memory_allocation = self.memory_allocator().allocate(&AllocationCreateDesc {
				requirements,
				name: create_info.name,
				location: create_info.usage.to_gpu_allocator_memory_location(),
				allocation_scheme: create_info.allocation_scheme.to_gpu_allocator_buffer(buffer),
				linear: true,
			})?;
			self.device
				.bind_buffer_memory(buffer, memory_allocation.memory(), memory_allocation.offset())?;
			Ok(AshBuffer {
				buffer,
				allocation: AshMemoryAllocation::new(memory_allocation),
			})
		}
	}

	unsafe fn alloc_image<T: ImageType>(
		&self,
		create_info: &BindlessImageCreateInfo<T>,
	) -> Result<Self::Image, Self::AllocationError> {
		unsafe {
			let image_type = bindless_image_type_to_vk_image_type::<T>().expect("Unsupported ImageType");
			let image = self.device.create_image(
				&ash::vk::ImageCreateInfo::default()
					.flags(ash::vk::ImageCreateFlags::empty())
					.image_type(image_type)
					.format(create_info.format)
					.extent(create_info.extent.into())
					.mip_levels(create_info.mip_levels)
					.array_layers(create_info.array_layers)
					.samples(create_info.samples.to_ash_sample_count_flags())
					.tiling(ImageTiling::OPTIMAL)
					.usage(create_info.usage.to_ash_image_usage_flags())
					.sharing_mode(SharingMode::EXCLUSIVE)
					.initial_layout(ImageLayout::UNDEFINED),
				None,
			)?;
			self.set_debug_object_name(image, create_info.name)?;
			let requirements = self.device.get_image_memory_requirements(image);
			let memory_allocation = self.memory_allocator().allocate(&AllocationCreateDesc {
				requirements,
				name: create_info.name,
				location: create_info.usage.to_gpu_allocator_memory_location(),
				allocation_scheme: create_info.allocation_scheme.to_gpu_allocator_image(image),
				linear: true,
			})?;
			self.device
				.bind_image_memory(image, memory_allocation.memory(), memory_allocation.offset())?;
			let image_view = self.create_image_view(image, create_info)?;
			Ok(AshImage {
				image,
				image_view,
				allocation: AshMemoryAllocation::new(memory_allocation),
			})
		}
	}

	unsafe fn alloc_sampler(
		&self,
		create_info: &BindlessSamplerCreateInfo,
	) -> Result<Self::Sampler, Self::AllocationError> {
		unsafe {
			Ok(self.device.create_sampler(
				&SamplerCreateInfo::default()
					.mag_filter(create_info.mag_filter.to_ash_filter())
					.min_filter(create_info.min_filter.to_ash_filter())
					.mipmap_mode(create_info.mipmap_mode.to_ash_mipmap_mode())
					.address_mode_u(create_info.address_mode_u.to_ash_address_mode())
					.address_mode_v(create_info.address_mode_v.to_ash_address_mode())
					.address_mode_w(create_info.address_mode_w.to_ash_address_mode())
					.anisotropy_enable(create_info.max_anisotropy.is_some())
					.max_anisotropy(create_info.max_anisotropy.unwrap_or(1.0))
					.min_lod(create_info.min_lod)
					.max_lod(create_info.max_lod.unwrap_or(LOD_CLAMP_NONE))
					.border_color(create_info.border_color.to_ash_border_color(false)),
				None,
			)?)
		}
	}

	unsafe fn mapped_buffer_to_slab(buffer: &BufferSlot<Self>) -> &mut (impl Slab + '_) {
		unsafe { buffer.allocation.get_mut() }
	}

	unsafe fn destroy_buffers<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		buffers: impl DescriptorIndexIterator<'a, BufferInterface<Self>>,
	) {
		unsafe {
			let mut allocator = self.memory_allocator();
			for (_, buffer) in buffers.into_iter() {
				// Safety: We have exclusive access to BufferSlot in this method. The MemoryAllocation will no longer
				// we accessed by anything nor dropped due to being wrapped in MaybeUninit, so we can safely read and drop
				// it ourselves.
				if let Some(allocation) = buffer.allocation.take() {
					allocator.free(allocation).unwrap();
				}
				self.device.destroy_buffer(buffer.buffer, None);
			}
		}
	}

	unsafe fn destroy_images<'a>(
		&self,
		_global_descriptor_set: &Self::BindlessDescriptorSet,
		images: impl DescriptorIndexIterator<'a, ImageInterface<Self>>,
	) {
		unsafe {
			let mut allocator = self.memory_allocator();
			for (_, image) in images.into_iter() {
				// Safety: We have exclusive access to BufferSlot in this method. The MemoryAllocation will no longer
				// we accessed by anything nor dropped due to being wrapped in MaybeUninit, so we can safely read and drop
				// it ourselves.
				if let Some(allocation) = image.allocation.take() {
					allocator.free(allocation).unwrap();
				}
				if let Some(imageview) = image.image_view {
					self.device.destroy_image_view(imageview, None);
				}
				// do not destroy swapchain images
				if !image.usage.contains(BindlessImageUsage::SWAPCHAIN) {
					self.device.destroy_image(image.image, None);
				}
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
				self.device.destroy_sampler(*sampler, None);
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// we want our [`BindlessBufferUsage`] bits to equal ash's [`BufferUsageFlags`] so the conversion can mostly be
	/// optimized away
	#[test]
	fn test_buffer_usage_to_ash_same_bits() {
		for usage in [
			BindlessBufferUsage::TRANSFER_SRC,
			BindlessBufferUsage::TRANSFER_DST,
			BindlessBufferUsage::UNIFORM_BUFFER,
			BindlessBufferUsage::STORAGE_BUFFER,
			BindlessBufferUsage::INDEX_BUFFER,
			BindlessBufferUsage::VERTEX_BUFFER,
			BindlessBufferUsage::INDIRECT_BUFFER,
		] {
			assert_eq!(
				Some(usage),
				BindlessBufferUsage::from_bits(usage.to_ash_buffer_usage_flags().as_raw() as u64)
			)
		}
	}
}
