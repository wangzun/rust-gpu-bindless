use crate::descriptor::Bindless;
use crate::pipeline::{
	GraphicsPipelineCreateInfo, MeshGraphicsPipelineCreateInfo, Recording, RecordingError, RenderPassFormat,
};
use crate::platform::BindlessPipelinePlatform;
use crate::platform::wgpu_hal::bindless::{WgpuHal, WgpuHalBindlessDescriptorSet, ash_format_to_wgpu};
use crate::platform::wgpu_hal::recording::{
	WgpuHalRecordingContext, WgpuHalRecordingError, WgpuHalRecordingResourceContext, wgpu_hal_record_and_execute,
};
use crate::platform::wgpu_hal::rendering::WgpuHalRenderingContext;
use rust_gpu_bindless_shaders::buffer_content::BufferStruct;
use rust_gpu_bindless_shaders::shader::BindlessShader;
use rust_gpu_bindless_shaders::shader_type::{
	ComputeShader, FragmentShader, MeshShader, ShaderType, TaskShader, VertexShader,
};
use std::ffi::CStr;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::LazyLock;
use wgpu_hal::{Api, Device as HalDevice, ShaderInput};
use wgpu_types as wgt;

static EMPTY_PIPELINE_CONSTANTS: LazyLock<naga::back::PipelineConstants> =
	LazyLock::new(naga::back::PipelineConstants::default);

// ----- Pipeline wrapper types -----

pub struct WgpuHalComputePipeline<A: Api>(pub WgpuHalPipeline<A>);
pub struct WgpuHalGraphicsPipeline<A: Api>(pub WgpuHalPipeline<A>);
pub struct WgpuHalMeshGraphicsPipeline<A: Api>(pub WgpuHalPipeline<A>);

pub enum WgpuHalPipelineInner<A: Api> {
	Compute(A::ComputePipeline),
	Render(A::RenderPipeline),
}

pub struct WgpuHalPipeline<A: Api> {
	pub bindless: Bindless<WgpuHal<A>>,
	pub pipeline: WgpuHalPipelineInner<A>,
}

impl<A: Api> Drop for WgpuHalPipeline<A> {
	fn drop(&mut self) {
		// TODO Pipelines need to be kept alive while executing. Put in TableSync?
	}
}

// ----- Shader module helper -----

struct WgpuHalShaderModule<'a, A: Api, S: ShaderType, T: BufferStruct> {
	bindless: Bindless<WgpuHal<A>>,
	module: ManuallyDrop<A::ShaderModule>,
	entry_point_name: &'a CStr,
	_phantom: PhantomData<(S, T)>,
}

impl<'a, A: Api, S: ShaderType, T: BufferStruct> WgpuHalShaderModule<'a, A, S, T> {
	fn new(
		bindless: &Bindless<WgpuHal<A>>,
		shader: &'a impl BindlessShader<ShaderType = S, ParamConstant = T>,
	) -> Result<Self, WgpuHalPipelineError> {
		unsafe {
			let device = &bindless.platform.create_info.device;
			let spirv = shader.spirv_binary();
			let module = device
				.create_shader_module(
					&wgpu_hal::ShaderModuleDescriptor {
						label: Some(&spirv.entry_point_name.to_string_lossy()),
						runtime_checks: wgt::ShaderRuntimeChecks::unchecked(),
					},
					ShaderInput::SpirV(spirv.binary),
				)
				.map_err(|_| WgpuHalPipelineError::ShaderCreation)?;
			Ok(Self {
				bindless: bindless.clone(),
				module: ManuallyDrop::new(module),
				entry_point_name: spirv.entry_point_name,
				_phantom: PhantomData,
			})
		}
	}

	fn to_programmable_stage(&self) -> wgpu_hal::ProgrammableStage<'_, A::ShaderModule> {
		wgpu_hal::ProgrammableStage {
			module: &self.module,
			entry_point: self.entry_point_name.to_str().unwrap_or("main"),
			constants: &EMPTY_PIPELINE_CONSTANTS,
			zero_initialize_workgroup_memory: false,
		}
	}
}

impl<A: Api, S: ShaderType, T: BufferStruct> Drop for WgpuHalShaderModule<'_, A, S, T> {
	fn drop(&mut self) {
		unsafe {
			let module = ManuallyDrop::take(&mut self.module);
			self.bindless
				.platform
				.create_info
				.device
				.destroy_shader_module(module);
		}
	}
}

// ----- Error type -----

#[derive(Debug, thiserror::Error)]
pub enum WgpuHalPipelineError {
	#[error("failed to create shader module")]
	ShaderCreation,
	#[error("failed to create pipeline: {0}")]
	Pipeline(#[from] wgpu_hal::PipelineError),
}

// ----- BindlessPipelinePlatform implementation -----

unsafe impl<A: Api> BindlessPipelinePlatform for WgpuHal<A> {
	type PipelineCreationError = WgpuHalPipelineError;
	type ComputePipeline = WgpuHalComputePipeline<A>;
	type RecordingResourceContext = WgpuHalRecordingResourceContext<A>;
	type RecordingContext<'a> = WgpuHalRecordingContext<'a, A>;
	type RecordingError = WgpuHalRecordingError;

	unsafe fn create_compute_pipeline<T: BufferStruct>(
		bindless: &Bindless<Self>,
		compute_shader: &impl BindlessShader<ShaderType = ComputeShader, ParamConstant = T>,
	) -> Result<Self::ComputePipeline, Self::PipelineCreationError> {
		unsafe {
			let compute = WgpuHalShaderModule::<A, _, _>::new(bindless, compute_shader)?;
			let device = &bindless.platform.create_info.device;
			let desc_set = bindless.global_descriptor_set();

			let pipeline = device.create_compute_pipeline(&wgpu_hal::ComputePipelineDescriptor {
				label: Some("bindless_compute"),
				layout: &desc_set.pipeline_layout,
				stage: compute.to_programmable_stage(),
				cache: None,
			})?;

			Ok(WgpuHalComputePipeline(WgpuHalPipeline {
				bindless: bindless.clone(),
				pipeline: WgpuHalPipelineInner::Compute(pipeline),
			}))
		}
	}

	unsafe fn record_and_execute<R: Send + Sync>(
		bindless: &Bindless<Self>,
		f: impl FnOnce(&mut Recording<'_, Self>) -> Result<R, RecordingError<Self>>,
	) -> Result<R, RecordingError<Self>> {
		unsafe { wgpu_hal_record_and_execute(bindless, f) }
	}

	type GraphicsPipeline = WgpuHalGraphicsPipeline<A>;
	type MeshGraphicsPipeline = WgpuHalMeshGraphicsPipeline<A>;
	type RenderingContext<'a: 'b, 'b> = WgpuHalRenderingContext<'a, 'b, A>;

	unsafe fn create_graphics_pipeline<T: BufferStruct>(
		bindless: &Bindless<Self>,
		render_pass: &RenderPassFormat,
		_create_info: &GraphicsPipelineCreateInfo,
		vertex_shader: &impl BindlessShader<ShaderType = VertexShader, ParamConstant = T>,
		fragment_shader: &impl BindlessShader<ShaderType = FragmentShader, ParamConstant = T>,
	) -> Result<Self::GraphicsPipeline, Self::PipelineCreationError> {
		unsafe {
			let vertex = WgpuHalShaderModule::<A, _, _>::new(bindless, vertex_shader)?;
			let fragment = WgpuHalShaderModule::<A, _, _>::new(bindless, fragment_shader)?;
			let device = &bindless.platform.create_info.device;
			let desc_set = bindless.global_descriptor_set();

			let color_targets: Vec<wgt::ColorTargetState> = render_pass
				.color_attachments
				.iter()
				.map(|&format| wgt::ColorTargetState {
					format: ash_format_to_wgpu(format),
					blend: Some(wgt::BlendState::REPLACE),
					write_mask: wgt::ColorWrites::ALL,
				})
				.collect();

			let depth_stencil = render_pass.depth_attachment.map(|format| wgt::DepthStencilState {
				format: ash_format_to_wgpu(format),
				depth_write_enabled: Some(true),
				depth_compare: Some(wgt::CompareFunction::Less),
				stencil: wgt::StencilState::default(),
				bias: wgt::DepthBiasState::default(),
			});

			let color_targets_opt: Vec<Option<wgt::ColorTargetState>> = color_targets.into_iter().map(Some).collect();

			let pipeline = device.create_render_pipeline(&wgpu_hal::RenderPipelineDescriptor {
				label: Some("bindless_graphics"),
				layout: &desc_set.pipeline_layout,
				vertex_processor: wgpu_hal::VertexProcessor::Standard {
					vertex_stage: vertex.to_programmable_stage(),
					vertex_buffers: &[],
				},
				fragment_stage: Some(fragment.to_programmable_stage()),
				color_targets: &color_targets_opt,
				depth_stencil,
				multisample: wgt::MultisampleState {
					count: 1,
					mask: !0,
					alpha_to_coverage_enabled: false,
				},
				multiview_mask: None,
				primitive: wgt::PrimitiveState {
					topology: wgt::PrimitiveTopology::TriangleList,
					strip_index_format: None,
					front_face: wgt::FrontFace::Ccw,
					cull_mode: None,
					unclipped_depth: false,
					polygon_mode: wgt::PolygonMode::Fill,
					conservative: false,
				},
				cache: None,
			})?;

			Ok(WgpuHalGraphicsPipeline(WgpuHalPipeline {
				bindless: bindless.clone(),
				pipeline: WgpuHalPipelineInner::Render(pipeline),
			}))
		}
	}

	unsafe fn create_mesh_graphics_pipeline<T: BufferStruct>(
		bindless: &Bindless<Self>,
		render_pass: &RenderPassFormat,
		_create_info: &MeshGraphicsPipelineCreateInfo,
		task_shader: Option<&impl BindlessShader<ShaderType = TaskShader, ParamConstant = T>>,
		mesh_shader: &impl BindlessShader<ShaderType = MeshShader, ParamConstant = T>,
		fragment_shader: &impl BindlessShader<ShaderType = FragmentShader, ParamConstant = T>,
	) -> Result<Self::MeshGraphicsPipeline, Self::PipelineCreationError> {
		// Mesh shaders are not directly supported in wgpu-hal's standard RenderPipelineDescriptor.
		// The Vulkan backend has mesh shader support, but it's not exposed in the generic API.
		// For now, this is unsupported and returns an error.
		Err(WgpuHalPipelineError::ShaderCreation)
	}
}
