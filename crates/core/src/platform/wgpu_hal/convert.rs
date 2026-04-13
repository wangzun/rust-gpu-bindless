use crate::descriptor::{
	AddressMode, BindlessBufferUsage, BindlessImageUsage, BorderColor, Extent, Filter, SampleCount,
};
use crate::pipeline::{ImageAccessType, LoadOp, RenderingAttachment, StoreOp};
use rust_gpu_bindless_shaders::descriptor::ImageType;
use rust_gpu_bindless_shaders::shader_type::Shader;
use spirv_std::image::{Arrayed, Dimensionality};
use wgpu_hal::Api;
use wgpu_types as wgt;

impl BindlessBufferUsage {
	pub fn to_wgpu_buffer_uses(&self) -> wgt::BufferUses {
		let mut out = wgt::BufferUses::empty();
		if self.contains(BindlessBufferUsage::TRANSFER_SRC) {
			out |= wgt::BufferUses::COPY_SRC;
		}
		if self.contains(BindlessBufferUsage::TRANSFER_DST) {
			out |= wgt::BufferUses::COPY_DST;
		}
		if self.contains(BindlessBufferUsage::MAP_READ) {
			out |= wgt::BufferUses::MAP_READ;
		}
		if self.contains(BindlessBufferUsage::MAP_WRITE) {
			out |= wgt::BufferUses::MAP_WRITE;
		}
		if self.contains(BindlessBufferUsage::UNIFORM_BUFFER) {
			out |= wgt::BufferUses::UNIFORM;
		}
		if self.contains(BindlessBufferUsage::STORAGE_BUFFER) {
			out |= wgt::BufferUses::STORAGE_READ_WRITE;
		}
		if self.contains(BindlessBufferUsage::INDEX_BUFFER) {
			out |= wgt::BufferUses::INDEX;
		}
		if self.contains(BindlessBufferUsage::VERTEX_BUFFER) {
			out |= wgt::BufferUses::VERTEX;
		}
		if self.contains(BindlessBufferUsage::INDIRECT_BUFFER) {
			out |= wgt::BufferUses::INDIRECT;
		}
		out
	}
}

impl BindlessImageUsage {
	pub fn to_wgpu_texture_uses(&self) -> wgt::TextureUses {
		let mut out = wgt::TextureUses::empty();
		if self.contains(BindlessImageUsage::TRANSFER_SRC) {
			out |= wgt::TextureUses::COPY_SRC;
		}
		if self.contains(BindlessImageUsage::TRANSFER_DST) {
			out |= wgt::TextureUses::COPY_DST;
		}
		if self.contains(BindlessImageUsage::SAMPLED) {
			out |= wgt::TextureUses::RESOURCE;
		}
		if self.contains(BindlessImageUsage::STORAGE) {
			out |= wgt::TextureUses::STORAGE_READ_WRITE;
		}
		if self.contains(BindlessImageUsage::COLOR_ATTACHMENT) {
			out |= wgt::TextureUses::COLOR_TARGET;
		}
		if self.contains(BindlessImageUsage::DEPTH_STENCIL_ATTACHMENT) {
			out |= wgt::TextureUses::DEPTH_STENCIL_READ | wgt::TextureUses::DEPTH_STENCIL_WRITE;
		}
		if self.contains(BindlessImageUsage::SWAPCHAIN) {
			out |= wgt::TextureUses::PRESENT;
		}
		out
	}

	pub fn has_texture_view(&self) -> bool {
		self.intersects(
			BindlessImageUsage::SAMPLED
				| BindlessImageUsage::STORAGE
				| BindlessImageUsage::COLOR_ATTACHMENT
				| BindlessImageUsage::DEPTH_STENCIL_ATTACHMENT,
		)
	}
}

pub fn bindless_image_type_to_wgpu_dimension<T: ImageType>() -> Option<wgt::TextureDimension> {
	match T::dimensionality() {
		Dimensionality::OneD => Some(wgt::TextureDimension::D1),
		Dimensionality::TwoD | Dimensionality::Cube | Dimensionality::Rect => Some(wgt::TextureDimension::D2),
		Dimensionality::ThreeD => Some(wgt::TextureDimension::D3),
		_ => None,
	}
}

pub fn bindless_image_type_to_wgpu_view_dimension<T: ImageType>() -> Option<wgt::TextureViewDimension> {
	match (T::dimensionality(), T::arrayed()) {
		(Dimensionality::OneD, Arrayed::False) => Some(wgt::TextureViewDimension::D1),
		(Dimensionality::TwoD, Arrayed::False) => Some(wgt::TextureViewDimension::D2),
		(Dimensionality::TwoD, Arrayed::True) => Some(wgt::TextureViewDimension::D2Array),
		(Dimensionality::ThreeD, Arrayed::False) => Some(wgt::TextureViewDimension::D3),
		(Dimensionality::Cube, Arrayed::False) => Some(wgt::TextureViewDimension::Cube),
		(Dimensionality::Cube, Arrayed::True) => Some(wgt::TextureViewDimension::CubeArray),
		(Dimensionality::Rect, Arrayed::False) => Some(wgt::TextureViewDimension::D2),
		(Dimensionality::Rect, Arrayed::True) => Some(wgt::TextureViewDimension::D2Array),
		_ => None,
	}
}

impl Extent {
	pub fn to_wgpu(&self) -> wgt::Extent3d {
		wgt::Extent3d {
			width: self.width,
			height: self.height,
			depth_or_array_layers: self.depth,
		}
	}
}

impl SampleCount {
	pub fn to_wgpu(&self) -> u32 {
		match self {
			SampleCount::Sample1 => 1,
			SampleCount::Sample2 => 2,
			SampleCount::Sample4 => 4,
			SampleCount::Sample8 => 8,
			SampleCount::Sample16 => 16,
			SampleCount::Sample32 => 32,
			SampleCount::Sample64 => 64,
		}
	}
}

impl Filter {
	pub fn to_wgpu(&self) -> wgt::FilterMode {
		match self {
			Filter::Nearest => wgt::FilterMode::Nearest,
			Filter::Linear => wgt::FilterMode::Linear,
		}
	}

	pub fn to_wgpu_mipmap(&self) -> wgt::MipmapFilterMode {
		match self {
			Filter::Nearest => wgt::MipmapFilterMode::Nearest,
			Filter::Linear => wgt::MipmapFilterMode::Linear,
		}
	}
}

impl AddressMode {
	pub fn to_wgpu(&self) -> wgt::AddressMode {
		match self {
			AddressMode::ClampToEdge => wgt::AddressMode::ClampToEdge,
			AddressMode::Repeat => wgt::AddressMode::Repeat,
			AddressMode::MirrorRepeat => wgt::AddressMode::MirrorRepeat,
			AddressMode::ClampToBorder => wgt::AddressMode::ClampToBorder,
		}
	}
}

impl BorderColor {
	pub fn to_wgpu(&self) -> wgt::SamplerBorderColor {
		match self {
			BorderColor::TransparentBlack => wgt::SamplerBorderColor::TransparentBlack,
			BorderColor::OpaqueBlack => wgt::SamplerBorderColor::OpaqueBlack,
			BorderColor::OpaqueWhite => wgt::SamplerBorderColor::OpaqueWhite,
		}
	}
}

pub fn shader_to_wgpu_shader_stage(shader: &Shader) -> wgt::ShaderStages {
	match shader {
		Shader::VertexShader => wgt::ShaderStages::VERTEX,
		Shader::FragmentShader => wgt::ShaderStages::FRAGMENT,
		Shader::ComputeShader => wgt::ShaderStages::COMPUTE,
		// Mesh/Task shaders don't have wgt::ShaderStages equivalents; use ALL for now
		Shader::MeshShader | Shader::TaskShader => wgt::ShaderStages::all(),
		_ => wgt::ShaderStages::all(),
	}
}

pub fn load_op_to_wgpu(load_op: &LoadOp) -> wgpu_hal::AttachmentOps {
	match load_op {
		LoadOp::Load => wgpu_hal::AttachmentOps::LOAD,
		LoadOp::Clear(_) => wgpu_hal::AttachmentOps::empty(),
		LoadOp::DontCare => wgpu_hal::AttachmentOps::empty(),
	}
}

pub fn store_op_to_wgpu(store_op: &StoreOp) -> wgpu_hal::AttachmentOps {
	match store_op {
		StoreOp::Store => wgpu_hal::AttachmentOps::STORE,
		StoreOp::DontCare => wgpu_hal::AttachmentOps::empty(),
	}
}
