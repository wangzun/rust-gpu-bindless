use rust_gpu_bindless_shader_builder::{ShaderSymbolsBuilder, spirv_builder::Capability};

fn main() -> anyhow::Result<()> {
	ShaderSymbolsBuilder::new("integration-test-shader", "spirv-unknown-vulkan1.2")?
		.capability(Capability::RayQueryKHR)
		.extension("SPV_KHR_ray_query")
		.build()?;
	Ok(())
}
