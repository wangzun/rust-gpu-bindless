use anyhow::{Context, Result};
use ash::vk::{
	ColorComponentFlags, CullModeFlags, FrontFace, PipelineColorBlendAttachmentState,
	PipelineColorBlendStateCreateInfo, PolygonMode, PrimitiveTopology,
};
use bevy::app::AppExit;
use bevy::camera::{Camera3d, PerspectiveProjection, Projection};
use bevy::camera_controller::free_camera::{FreeCamera, FreeCameraPlugin};
use bevy::ecs::message::{MessageReader, MessageWriter};
use bevy::input::ButtonInput;
use bevy::input::keyboard::KeyCode;
use bevy::prelude::*;
use bevy::time::Real;
use bevy::window::{Window, WindowCloseRequested};
use bevy::winit::{DisplayHandleWrapper, RawWinitWindowEvent, WINIT_WINDOWS};
use egui::Context as EguiContext;
use glam::{Mat4, UVec2};
use integration_test_shader::terrain::{BufferAParam, BufferBParam, Param};
use rust_gpu_bindless_bevy::{BindlessPlugin, BindlessRenderManagerState};
use rust_gpu_bindless_core::descriptor::{
	AddressMode, Bindless, BindlessImageCreateInfo, BindlessImageUsage, BindlessSamplerCreateInfo, Extent, Filter,
	Format, Image, Image2d, ImageDescExt, MutDesc, MutDescExt, MutImage, RCDesc, RCDescExt, Sampler, UnsafeDesc,
};
use rust_gpu_bindless_core::pipeline::DrawIndirectCommand;
use rust_gpu_bindless_core::pipeline::{
	BindlessGraphicsPipeline, ClearValue, ColorAttachment, GraphicsPipelineCreateInfo, LoadOp, MutImageAccessExt,
	PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineRasterizationStateCreateInfo,
	Present, RenderPassFormat, RenderingAttachment, SampledRead, StoreOp,
};
use rust_gpu_bindless_core::platform::BindlessPipelinePlatform;
use rust_gpu_bindless_core::platform::ash::Ash;
use rust_gpu_bindless_egui::renderer::{
	EguiRenderContext, EguiRenderOutput, EguiRenderPipeline, EguiRenderer, EguiRenderingOptions,
};
use smallvec::SmallVec;
use winit::window::WindowId;

fn main() {
	App::new()
		.add_plugins(DefaultPlugins.set(WindowPlugin {
			primary_window: Some(Window {
				title: "swapchain triangle".into(),
				..Default::default()
			}),
				..Default::default()
			}))
			.add_plugins((BindlessPlugin, FreeCameraPlugin))
			.insert_resource(DemoRenderState::default())
			.add_systems(Startup, spawn_camera)
			.add_systems(Update, (handle_close_requests, render_frame).chain())
			.run();
}

#[derive(Resource, Default)]
struct DemoRenderState {
	renderer: Option<TriangleRenderer<Ash>>,
	egui: Option<BevyEguiContext<Ash>>,
	egui_pipeline: Option<EguiRenderPipeline<Ash>>,
	fps: FpsCounter,
}

fn handle_close_requests(mut close_requests: MessageReader<WindowCloseRequested>, mut exit: MessageWriter<AppExit>) {
	if close_requests.read().next().is_some() {
		exit.write(AppExit::Success);
	}
}

#[derive(Component)]
struct DemoCamera;

const CAMERA_FOV_Y_RADIANS: f32 = 11.0 * std::f32::consts::PI / 180.0;

fn spawn_camera(mut commands: Commands) {
	commands.spawn((
		Camera3d::default(),
		Projection::Perspective(PerspectiveProjection {
			fov: CAMERA_FOV_Y_RADIANS,
			..default()
		}),
		Transform {
			translation: bevy::math::Vec3::new(-0.91287905, 1.7548301, -2.8095529),
			rotation: bevy::math::Quat::from_euler(bevy::math::EulerRot::ZYX, 0.0, -2.8274333, -0.43),
			..default()
		},
		FreeCamera {
			sensitivity: 0.2,
			friction: 40.0,
			walk_speed: 5.0,
			run_speed: 15.0,
			scroll_factor: 0.04879016,
			..default()
		},
		DemoCamera,
	));
}

fn render_frame(
	mut bindless_state: NonSendMut<BindlessRenderManagerState>,
	mut demo_state: ResMut<DemoRenderState>,
	camera_query: Query<(&Transform, &Projection), With<DemoCamera>>,
	time: Res<Time<Real>>,
	display_handle: Res<DisplayHandleWrapper>,
	key_input: Res<ButtonInput<KeyCode>>,
	mut raw_events: MessageReader<RawWinitWindowEvent>,
) {
		if let Err(err) = render_frame_inner(
			&mut bindless_state,
			&mut demo_state,
			&camera_query,
			&time,
			&display_handle,
			&key_input,
		&mut raw_events,
	) {
		panic!("{err:#}");
	}
}

fn render_frame_inner(
	bindless_state: &mut BindlessRenderManagerState,
	demo_state: &mut DemoRenderState,
	camera_query: &Query<(&Transform, &Projection), With<DemoCamera>>,
	time: &Time<Real>,
	display_handle: &DisplayHandleWrapper,
	key_input: &ButtonInput<KeyCode>,
	raw_events: &mut MessageReader<RawWinitWindowEvent>,
) -> Result<()> {
	let Some(manager) = bindless_state.get_mut() else {
		return Ok(());
	};
	let Ok((camera_transform, camera_projection)) = camera_query.single() else {
		return Ok(());
	};
	let camera_world = Mat4::from_cols_array(&camera_transform.to_matrix().to_cols_array());
	let camera_fov_y_radians = match camera_projection {
		Projection::Perspective(perspective) => perspective.fov,
		_ => return Ok(()),
	};

	let DemoRenderState {
		renderer,
		egui,
		egui_pipeline,
		fps,
	} = demo_state;

	if renderer.is_none() {
		*renderer = Some(TriangleRenderer::new(manager.bindless(), manager.params().format)?);
	}
	if egui.is_none() || egui_pipeline.is_none() {
		let egui_renderer = EguiRenderer::new(manager.bindless().clone());
		*egui_pipeline = Some(EguiRenderPipeline::new(
			egui_renderer.clone(),
			Some(manager.params().format),
			None,
		));
		*egui = Some(BevyEguiContext::new(
			egui_renderer,
			EguiContext::default(),
			display_handle,
			manager.window_entity(),
		)?);
	}

	let renderer = renderer.as_mut().expect("renderer should be initialized");
	let egui = egui.as_mut().expect("egui context should be initialized");
	let egui_pipeline = egui_pipeline.as_ref().expect("egui pipeline should be initialized");

	for event in raw_events.read() {
		egui.on_raw_window_event(event)?;
	}

	if key_input.just_pressed(KeyCode::Space) {
		renderer.toggle_display_mode();
	}

	fps.update(time.delta_secs());
		let fps_value = fps.fps();
		let frame_ms = fps.frame_ms();
		let display_mode = renderer.display_mode_name();

		let (egui_render, egui_scale) = egui.run(|ctx| draw_perf_overlay(ctx, fps_value, frame_ms, display_mode))?;
		let rt = manager.acquire_image(None)?;
		let rt = renderer.draw(rt, camera_world, camera_fov_y_radians)?;
		let rt = manager.bindless().execute(|cmd| {
			let mut rt = rt.access_dont_care::<ColorAttachment>(cmd)?;
			egui_render
			.draw(
				egui_pipeline,
				cmd,
				Some(&mut rt),
				None,
				EguiRenderingOptions {
					render_scale: egui_scale,
					..EguiRenderingOptions::default()
				},
			)
			.expect("egui overlay draw failed");
		Ok(rt.transition::<Present>()?.into_desc())
	})?;
	manager.present_image(rt)?;

	Ok(())
}

fn draw_perf_overlay(ctx: &egui::Context, fps_value: f32, frame_ms: f32, display_mode: &str) {
	egui::Area::new("perf_overlay".into())
		.anchor(egui::Align2::LEFT_TOP, egui::vec2(12.0, 12.0))
		.interactable(false)
		.show(ctx, |ui| {
			egui::Frame::none()
				.fill(egui::Color32::from_black_alpha(176))
				.stroke(egui::Stroke::new(1.0, egui::Color32::from_white_alpha(32)))
				.rounding(egui::Rounding::same(6.0))
				.inner_margin(egui::Margin::same(10.0))
				.show(ui, |ui| {
					ui.set_min_width(120.0);
					ui.label(
						egui::RichText::new(format!("FPS {:.1}", fps_value))
							.monospace()
							.strong(),
					);
					ui.label(egui::RichText::new(format!("{frame_ms:.2} ms")).monospace());
					ui.label(egui::RichText::new(display_mode).monospace());
				});
		});
}

fn with_winit_window<R>(window_entity: Entity, f: impl FnOnce(&winit::window::Window) -> Result<R>) -> Result<R> {
	WINIT_WINDOWS.with_borrow(|winit_windows| {
		let window = winit_windows
			.get_window(window_entity)
			.context("missing primary Bevy winit window")?;
		f(window)
	})
}

struct BevyEguiContext<P: rust_gpu_bindless_egui::platform::EguiBindlessPlatform> {
	window_entity: Entity,
	window_id: WindowId,
	render_ctx: EguiRenderContext<P>,
	winit_state: egui_winit::State,
}

impl<P: rust_gpu_bindless_egui::platform::EguiBindlessPlatform> BevyEguiContext<P> {
	fn new(
		renderer: EguiRenderer<P>,
		ctx: EguiContext,
		display_handle: &DisplayHandleWrapper,
		window_entity: Entity,
	) -> Result<Self> {
		let max_texture_side = unsafe { renderer.bindless().platform.max_image_dimensions_2d() };
		let (window_id, scale_factor, theme) = with_winit_window(window_entity, |window| {
			Ok((window.id(), window.scale_factor() as f32, window.theme()))
		})?;
		let mut slf = Self {
			window_entity,
			window_id,
			render_ctx: EguiRenderContext::new(renderer, ctx.clone()),
			winit_state: egui_winit::State::new(
				ctx,
				egui::ViewportId::ROOT,
				&display_handle.0,
				Some(scale_factor),
				theme,
				Some(max_texture_side as usize),
			),
		};
		slf.update_viewport_info(true)?;
		Ok(slf)
	}

	fn on_raw_window_event(&mut self, event: &RawWinitWindowEvent) -> Result<()> {
		if event.window_id != self.window_id {
			return Ok(());
		}
		with_winit_window(self.window_entity, |window| {
			let _ = self.winit_state.on_window_event(window, &event.event);
			Ok(())
		})
	}

	fn run(&mut self, run_ui: impl FnMut(&egui::Context)) -> Result<(EguiRenderOutput<'_, P>, f32)> {
		let scale = self.update_viewport_info(false)?.recip();
		let raw_input = with_winit_window(self.window_entity, |window| {
			Ok(self.winit_state.take_egui_input(window))
		})?;
		let (render, platform_output) = self.render_ctx.run(raw_input, run_ui)?;
		with_winit_window(self.window_entity, |window| {
			self.winit_state.handle_platform_output(window, platform_output);
			Ok(())
		})?;
		Ok((render, scale))
	}

	fn update_viewport_info(&mut self, is_init: bool) -> Result<f32> {
		with_winit_window(self.window_entity, |window| {
			let raw_input = self.winit_state.egui_input_mut();
			let viewport_info = raw_input.viewports.entry(raw_input.viewport_id).or_default();
			egui_winit::update_viewport_info(viewport_info, &self.render_ctx, window, is_init);
			Ok(viewport_info.native_pixels_per_point.unwrap())
		})
	}
}

#[derive(Default)]
struct FpsCounter {
	accum_time: f32,
	accum_frames: u32,
	fps: f32,
	frame_ms: f32,
}

impl FpsCounter {
	fn update(&mut self, dt: f32) {
		self.accum_time += dt;
		self.accum_frames += 1;
		self.frame_ms = dt * 1000.0;
		if self.accum_time >= 0.25 {
			self.fps = self.accum_frames as f32 / self.accum_time.max(f32::EPSILON);
			self.accum_time = 0.0;
			self.accum_frames = 0;
		}
	}

	fn fps(&self) -> f32 {
		self.fps
	}

	fn frame_ms(&self) -> f32 {
		self.frame_ms
	}
}

#[derive(Debug, Copy, Clone)]
pub struct TriangleRendererRTFormat {
	rt_format: Format,
}

impl TriangleRendererRTFormat {
	pub fn to_render_pass_format(&self) -> RenderPassFormat {
		RenderPassFormat {
			color_attachments: SmallVec::from_slice(&[self.rt_format]),
			depth_attachment: None,
		}
	}
}

pub struct TriangleRenderer<P: BindlessPipelinePlatform> {
	bindless: Bindless<P>,
	rt_format: TriangleRendererRTFormat,
	buffer_a_pipeline: BindlessGraphicsPipeline<P, BufferAParam>,
	buffer_b_pipeline: BindlessGraphicsPipeline<P, BufferBParam>,
	pipeline: BindlessGraphicsPipeline<P, Param>,
	map_image: Option<MutDesc<P, MutImage<Image2d>>>,
	detail_image: Option<MutDesc<P, MutImage<Image2d>>>,
	sampler: RCDesc<P, Sampler>,
	map_extent: UVec2,
	terrain_time: f32,
	maps_initialized: bool,
	display_mode: u32,
}

const VERTEX_CNT: usize = 3;
const MAP_RESOLUTION: u32 = 4096;
const MAP_FORMAT: Format = Format::R16G16B16A16_SFLOAT;
const DETAIL_FORMAT: Format = Format::R16G16B16A16_SFLOAT;

impl<P: BindlessPipelinePlatform> TriangleRenderer<P> {
	pub fn new(bindless: &Bindless<P>, rt_format: Format) -> Result<Self> {
		let rt_format = TriangleRendererRTFormat { rt_format };
		let map_extent = UVec2::splat(MAP_RESOLUTION);
		let color_blend_attachments =
			[PipelineColorBlendAttachmentState::default().color_write_mask(ColorComponentFlags::RGBA)];
		let map_render_pass = RenderPassFormat {
			color_attachments: SmallVec::from_slice(&[MAP_FORMAT]),
			depth_attachment: None,
		};
		let detail_render_pass = RenderPassFormat {
			color_attachments: SmallVec::from_slice(&[DETAIL_FORMAT]),
			depth_attachment: None,
		};

		let pipeline_ci = GraphicsPipelineCreateInfo {
			input_assembly_state: PipelineInputAssemblyStateCreateInfo::default()
				.topology(PrimitiveTopology::TRIANGLE_LIST),
			rasterization_state: PipelineRasterizationStateCreateInfo::default()
				.polygon_mode(PolygonMode::FILL)
				.front_face(FrontFace::COUNTER_CLOCKWISE)
				.cull_mode(CullModeFlags::NONE),
			depth_stencil_state: PipelineDepthStencilStateCreateInfo::default(),
			color_blend_state: PipelineColorBlendStateCreateInfo::default().attachments(&color_blend_attachments),
		};

		let buffer_a_pipeline = bindless.create_graphics_pipeline(
			&map_render_pass,
			&pipeline_ci,
			integration_test::shader::terrain::terrain_buffer_a_vertex::new(),
			integration_test::shader::terrain::terrain_buffer_a_fragment::new(),
		)?;

		let buffer_b_pipeline = bindless.create_graphics_pipeline(
			&detail_render_pass,
			&pipeline_ci,
			integration_test::shader::terrain::terrain_buffer_b_vertex::new(),
			integration_test::shader::terrain::terrain_buffer_b_fragment::new(),
		)?;

		let pipeline = bindless.create_graphics_pipeline(
			&rt_format.to_render_pass_format(),
			&pipeline_ci,
			integration_test::shader::terrain::terrain_vertex::new(),
			integration_test::shader::terrain::terrain_fragment::new(),
		)?;

		let map_image = bindless.image().alloc::<Image2d>(&BindlessImageCreateInfo {
			name: "terrain_map_buffer_a",
			format: MAP_FORMAT,
			extent: Extent::from(map_extent),
			usage: BindlessImageUsage::COLOR_ATTACHMENT | BindlessImageUsage::SAMPLED,
			..BindlessImageCreateInfo::default()
		})?;

		let detail_image = bindless.image().alloc::<Image2d>(&BindlessImageCreateInfo {
			name: "terrain_map_buffer_b",
			format: DETAIL_FORMAT,
			extent: Extent::from(map_extent),
			usage: BindlessImageUsage::COLOR_ATTACHMENT | BindlessImageUsage::SAMPLED,
			..BindlessImageCreateInfo::default()
		})?;

		let sampler = bindless.sampler().alloc(&BindlessSamplerCreateInfo {
			mag_filter: Filter::Linear,
			min_filter: Filter::Linear,
			address_mode_u: AddressMode::ClampToEdge,
			address_mode_v: AddressMode::ClampToEdge,
			address_mode_w: AddressMode::ClampToEdge,
			..BindlessSamplerCreateInfo::default()
		})?;

		Ok(Self {
			bindless: bindless.clone(),
			rt_format,
			buffer_a_pipeline,
			buffer_b_pipeline,
			pipeline,
			map_image: Some(map_image),
			detail_image: Some(detail_image),
			sampler,
			map_extent,
			terrain_time: 0.0,
			maps_initialized: false,
			display_mode: 0,
		})
	}

	pub fn toggle_display_mode(&mut self) {
		self.display_mode = (self.display_mode + 1) % 3;
	}

	pub fn display_mode_name(&self) -> &'static str {
		match self.display_mode {
			1 => "map_image",
			2 => "detail_image",
			_ => "result",
		}
	}

	pub fn draw(
		&mut self,
		rt: MutDesc<P, MutImage<Image2d>>,
		camera_world: Mat4,
		camera_fov_y_radians: f32,
	) -> Result<MutDesc<P, MutImage<Image2d>>> {
		let map_resolution = self.map_extent.as_vec2();
		let frame_time = self.terrain_time;
		let rt_resolution = UVec2::from(rt.extent()).as_vec2();
		let bindless = self.bindless.clone();
		let buffer_a_pipeline = &self.buffer_a_pipeline;
		let buffer_b_pipeline = &self.buffer_b_pipeline;
		let pipeline = &self.pipeline;
		let sampler = self.sampler.to_strong();
		let map_render_pass = RenderPassFormat {
			color_attachments: SmallVec::from_slice(&[MAP_FORMAT]),
			depth_attachment: None,
		};
		let detail_render_pass = RenderPassFormat {
			color_attachments: SmallVec::from_slice(&[DETAIL_FORMAT]),
			depth_attachment: None,
		};

		if !self.maps_initialized {
			let map_desc = self.map_image.take().expect("terrain map image missing");
			let detail_desc = self.detail_image.take().expect("terrain detail image missing");
			let (map_image, detail_image) = bindless.execute(|cmd| {
				let mut map_image = map_desc.access_dont_care::<ColorAttachment>(cmd)?;
				cmd.begin_rendering(
					map_render_pass,
					&[RenderingAttachment {
						image: &mut map_image,
						load_op: LoadOp::DontCare,
						store_op: StoreOp::Store,
					}],
					None,
					|rp| {
						rp.draw(
							buffer_a_pipeline,
							DrawIndirectCommand {
								vertex_count: VERTEX_CNT as u32,
								instance_count: 1,
								..DrawIndirectCommand::default()
							},
							BufferAParam {
								resolution: map_resolution,
								time: frame_time,
							},
						)?;
						Ok(())
					},
				)?;
				let map_image = map_image.transition::<SampledRead>()?;

				let mut detail_image = detail_desc.access_dont_care::<ColorAttachment>(cmd)?;
				cmd.begin_rendering(
					detail_render_pass,
					&[RenderingAttachment {
						image: &mut detail_image,
						load_op: LoadOp::DontCare,
						store_op: StoreOp::Store,
					}],
					None,
					|rp| {
						rp.draw(
							buffer_b_pipeline,
							DrawIndirectCommand {
								vertex_count: VERTEX_CNT as u32,
								instance_count: 1,
								..DrawIndirectCommand::default()
							},
							BufferBParam {
								resolution: map_resolution,
								time: frame_time,
							},
						)?;
						Ok(())
					},
				)?;
				let detail_image = detail_image.transition::<SampledRead>()?;

				Ok((map_image.into_desc(), detail_image.into_desc()))
			})?;
			self.map_image = Some(map_image);
			self.detail_image = Some(detail_image);
			self.maps_initialized = true;
		}

		let map_id = self.map_image.as_ref().expect("terrain map image missing").id();
		let detail_id = self.detail_image.as_ref().expect("terrain detail image missing").id();
		let rt = bindless.execute(|cmd| {
			let mut rt = rt.access_dont_care::<ColorAttachment>(cmd)?;
			cmd.begin_rendering(
				self.rt_format.to_render_pass_format(),
				&[RenderingAttachment {
					image: &mut rt,
					load_op: LoadOp::Clear(ClearValue::ColorF([0.0, 0.0, 0.0, 1.0])),
					store_op: StoreOp::Store,
				}],
				None,
				|rp| {
					rp.draw(
						pipeline,
						DrawIndirectCommand {
							vertex_count: VERTEX_CNT as u32,
							instance_count: 1,
							..DrawIndirectCommand::default()
						},
							Param {
								map_image: unsafe { UnsafeDesc::<Image<Image2d>>::new(map_id) },
								detail_image: unsafe { UnsafeDesc::<Image<Image2d>>::new(detail_id) },
								sampler,
								resolution: rt_resolution,
								map_resolution,
								time: frame_time,
								display_mode: self.display_mode,
								camera_fov_y_radians,
								camera_world,
							},
						)?;
						Ok(())
				},
			)?;

			Ok(rt.into_desc())
		})?;
		Ok(rt)
	}
}
