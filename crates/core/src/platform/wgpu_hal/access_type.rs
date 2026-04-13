use crate::pipeline::{BufferAccess, ImageAccess};
use wgpu_types as wgt;

impl BufferAccess {
	pub fn to_wgpu_buffer_uses(&self) -> wgt::BufferUses {
		match self {
			BufferAccess::Undefined => wgt::BufferUses::empty(),
			BufferAccess::General => wgt::BufferUses::STORAGE_READ_WRITE,
			BufferAccess::GeneralRead => wgt::BufferUses::STORAGE_READ_WRITE,
			BufferAccess::GeneralWrite => wgt::BufferUses::STORAGE_READ_WRITE,
			BufferAccess::TransferRead => wgt::BufferUses::COPY_SRC,
			BufferAccess::TransferWrite => wgt::BufferUses::COPY_DST,
			BufferAccess::ShaderRead => wgt::BufferUses::STORAGE_READ_WRITE,
			BufferAccess::ShaderWrite => wgt::BufferUses::STORAGE_READ_WRITE,
			BufferAccess::ShaderReadWrite => wgt::BufferUses::STORAGE_READ_WRITE,
			BufferAccess::HostAccess => wgt::BufferUses::MAP_READ | wgt::BufferUses::MAP_WRITE,
			BufferAccess::IndirectCommandRead => wgt::BufferUses::INDIRECT,
			BufferAccess::IndexRead => wgt::BufferUses::INDEX,
			BufferAccess::VertexAttributeRead => wgt::BufferUses::VERTEX,
		}
	}
}

impl ImageAccess {
	pub fn to_wgpu_texture_uses(&self) -> wgt::TextureUses {
		match self {
			ImageAccess::Undefined => wgt::TextureUses::UNINITIALIZED,
			ImageAccess::General => wgt::TextureUses::STORAGE_READ_WRITE,
			ImageAccess::GeneralRead => wgt::TextureUses::RESOURCE,
			ImageAccess::GeneralWrite => wgt::TextureUses::STORAGE_READ_WRITE,
			ImageAccess::TransferRead => wgt::TextureUses::COPY_SRC,
			ImageAccess::TransferWrite => wgt::TextureUses::COPY_DST,
			ImageAccess::StorageRead => wgt::TextureUses::STORAGE_READ_WRITE,
			ImageAccess::StorageWrite => wgt::TextureUses::STORAGE_READ_WRITE,
			ImageAccess::StorageReadWrite => wgt::TextureUses::STORAGE_READ_WRITE,
			ImageAccess::SampledRead => wgt::TextureUses::RESOURCE,
			ImageAccess::ColorAttachment => wgt::TextureUses::COLOR_TARGET,
			ImageAccess::DepthStencilAttachment => wgt::TextureUses::DEPTH_STENCIL_WRITE,
			ImageAccess::Present => wgt::TextureUses::PRESENT,
		}
	}
}
