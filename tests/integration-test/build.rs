use rust_gpu_bindless_shader_builder::ShaderSymbolsBuilder;

fn main() -> anyhow::Result<()> {
	println!("cargo:rerun-if-changed=../integration-test-shader/src");
	println!("cargo:rerun-if-changed=../integration-test-shader/Cargo.toml");
	ShaderSymbolsBuilder::new("integration-test-shader", "spirv-unknown-vulkan1.2")?.build()?;
	Ok(())
}
