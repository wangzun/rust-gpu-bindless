use crate::descriptor::Bindless;
use crate::pipeline::{
	BindlessGraphicsPipeline, BindlessMeshGraphicsPipeline, ColorAttachment, DepthStencilAttachment,
	DrawIndexedIndirectCommand, DrawIndirectCommand, HasResourceContext, IndexReadable, IndexTypeTrait,
	IndirectCommandReadable, MutOrSharedBuffer, RecordingError, RenderPassFormat, RenderingAttachment,
};
use crate::platform::RenderingContext;
use crate::platform::wgpu_hal::bindless::WgpuHal;
use crate::platform::wgpu_hal::recording::{WgpuHalRecordingContext, WgpuHalRecordingError, WgpuHalRecordingResourceContext};
use glam::UVec2;
use rust_gpu_bindless_shaders::buffer_content::BufferStruct;
use rust_gpu_bindless_shaders::descriptor::TransientAccess;
use rust_gpu_bindless_shaders::utils::rect::IRect2;
use rust_gpu_bindless_shaders::utils::viewport::Viewport;
use std::ops::{Deref, DerefMut};
use wgpu_hal::Api;

pub struct WgpuHalRenderingContext<'a, 'b, A: Api> {
	recording: &'b mut WgpuHalRecordingContext<'a, A>,
	graphics_bind_descriptors: bool,
	viewport: Viewport,
	scissor: IRect2,
	set_viewport: bool,
	set_scissor: bool,
}

impl<'a, A: Api> Deref for WgpuHalRenderingContext<'a, '_, A> {
	type Target = WgpuHalRecordingContext<'a, A>;

	fn deref(&self) -> &Self::Target {
		self.recording
	}
}

impl<A: Api> DerefMut for WgpuHalRenderingContext<'_, '_, A> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		self.recording
	}
}

impl<'a, 'b, A: Api> WgpuHalRenderingContext<'a, 'b, A> {
	pub unsafe fn new(recording: &'b mut WgpuHalRecordingContext<'a, A>) -> Self {
		Self {
			recording,
			graphics_bind_descriptors: true,
			viewport: Viewport::default(),
			scissor: IRect2::default(),
			set_viewport: true,
			set_scissor: true,
		}
	}
}

unsafe impl<'a, A: Api> TransientAccess<'a> for WgpuHalRenderingContext<'a, '_, A> {}

unsafe impl<'a, A: Api> HasResourceContext<'a, WgpuHal<A>> for WgpuHalRenderingContext<'a, '_, A> {
	#[inline]
	fn bindless(&self) -> &Bindless<WgpuHal<A>> {
		self.recording.bindless()
	}

	#[inline]
	fn resource_context(&self) -> &'a WgpuHalRecordingResourceContext<A> {
		self.recording.resource_context()
	}
}

unsafe impl<'a, 'b, A: Api> RenderingContext<'a, 'b, WgpuHal<A>> for WgpuHalRenderingContext<'a, 'b, A> {
	unsafe fn begin_rendering(
		recording: &'b mut WgpuHalRecordingContext<'a, A>,
		_format: RenderPassFormat,
		_render_area: UVec2,
		_color_attachments: &[RenderingAttachment<WgpuHal<A>, ColorAttachment>],
		_depth_attachment: Option<RenderingAttachment<WgpuHal<A>, DepthStencilAttachment>>,
	) -> Result<Self, WgpuHalRecordingError> {
		unsafe {
			// TODO: Begin render pass using wgpu_hal::CommandEncoder::begin_render_pass
			// Requires building wgpu_hal::RenderPassDescriptor from attachments
			Ok(Self::new(recording))
		}
	}

	unsafe fn end_rendering(&mut self) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			// TODO: encoder.end_render_pass()
			Ok(())
		}
	}

	unsafe fn set_viewport(&mut self, viewport: Viewport) {
		self.viewport = viewport;
		self.set_viewport = true;
	}

	unsafe fn set_scissor(&mut self, scissor: IRect2) {
		self.scissor = scissor;
		self.set_scissor = true;
	}

	unsafe fn draw<T: BufferStruct>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<WgpuHal<A>, T>,
		count: DrawIndirectCommand,
		param: T,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			self.recording.push_param(param);
			// TODO: bind pipeline, set_bind_group, set_viewport, set_scissor, draw
			let _ = (pipeline, count);
			Ok(())
		}
	}

	unsafe fn draw_indexed<T: BufferStruct, IT: IndexTypeTrait, AIR: IndexReadable>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<WgpuHal<A>, T>,
		index_buffer: impl MutOrSharedBuffer<WgpuHal<A>, [IT], AIR>,
		count: DrawIndexedIndirectCommand,
		param: T,
	) -> Result<(), RecordingError<WgpuHal<A>>> {
		unsafe {
			self.recording.push_param(param);
			let _index_slot = index_buffer.inner_slot();
			// TODO: bind pipeline, bind index buffer, draw_indexed
			let _ = (pipeline, count);
			Ok(())
		}
	}

	unsafe fn draw_indirect<T: BufferStruct, AIC: IndirectCommandReadable>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<WgpuHal<A>, T>,
		indirect: impl MutOrSharedBuffer<WgpuHal<A>, DrawIndirectCommand, AIC>,
		param: T,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			self.recording.push_param(param);
			let _indirect_slot = indirect.inner_slot();
			// TODO: bind pipeline, draw_indirect
			let _ = pipeline;
			Ok(())
		}
	}

	unsafe fn draw_indexed_indirect<
		T: BufferStruct,
		IT: IndexTypeTrait,
		AIR: IndexReadable,
		AIC: IndirectCommandReadable,
	>(
		&mut self,
		pipeline: &BindlessGraphicsPipeline<WgpuHal<A>, T>,
		index_buffer: impl MutOrSharedBuffer<WgpuHal<A>, [IT], AIR>,
		indirect: impl MutOrSharedBuffer<WgpuHal<A>, DrawIndexedIndirectCommand, AIC>,
		param: T,
	) -> Result<(), WgpuHalRecordingError> {
		unsafe {
			self.recording.push_param(param);
			let _index_slot = index_buffer.inner_slot();
			let _indirect_slot = indirect.inner_slot();
			// TODO: bind pipeline, bind index buffer, draw_indexed_indirect
			let _ = pipeline;
			Ok(())
		}
	}

	unsafe fn draw_mesh_tasks<T: BufferStruct>(
		&mut self,
		_pipeline: &BindlessMeshGraphicsPipeline<WgpuHal<A>, T>,
		_group_counts: [u32; 3],
		_param: T,
	) -> Result<(), WgpuHalRecordingError> {
		// Mesh shaders not supported in generic wgpu-hal
		Err(WgpuHalRecordingError::BarrierWhileRendering)
	}

	unsafe fn draw_mesh_tasks_indirect<T: BufferStruct, AIC: IndirectCommandReadable>(
		&mut self,
		_pipeline: &BindlessMeshGraphicsPipeline<WgpuHal<A>, T>,
		_indirect: impl MutOrSharedBuffer<WgpuHal<A>, [u32; 3], AIC>,
		_param: T,
	) -> Result<(), WgpuHalRecordingError> {
		// Mesh shaders not supported in generic wgpu-hal
		Err(WgpuHalRecordingError::BarrierWhileRendering)
	}
}
