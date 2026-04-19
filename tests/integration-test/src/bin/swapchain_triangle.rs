use ash::vk::{
	ColorComponentFlags, CullModeFlags, FrontFace, PipelineColorBlendAttachmentState,
	PipelineColorBlendStateCreateInfo, PolygonMode, PrimitiveTopology,
};
use glam::{EulerRot, Mat4, Quat, UVec2, Vec2, Vec3};
use integration_test::debugger;
use integration_test_shader::terrain::{BufferAParam, BufferBParam, Param};
use rust_gpu_bindless_core::descriptor::{
	AddressMode, Bindless, BindlessImageCreateInfo, BindlessImageUsage, BindlessInstance, BindlessSamplerCreateInfo,
	DescriptorCounts, Extent, Filter, Format, Image, Image2d, ImageDescExt, MutDesc, MutDescExt, MutImage, RCDesc,
	RCDescExt, Sampler, UnsafeDesc,
};
use rust_gpu_bindless_core::pipeline::DrawIndirectCommand;
use rust_gpu_bindless_core::pipeline::{
	BindlessGraphicsPipeline, ClearValue, ColorAttachment, GraphicsPipelineCreateInfo, LoadOp, MutImageAccessExt,
	PipelineDepthStencilStateCreateInfo, PipelineInputAssemblyStateCreateInfo, PipelineRasterizationStateCreateInfo,
	Present, RenderPassFormat, RenderingAttachment, SampledRead, StoreOp,
};
use rust_gpu_bindless_core::platform::BindlessPipelinePlatform;
use rust_gpu_bindless_core::platform::ash::Debuggers;
use rust_gpu_bindless_core::platform::ash::{Ash, AshSingleGraphicsQueueCreateInfo, ash_init_single_graphics_queue};
use rust_gpu_bindless_egui::renderer::{EguiRenderPipeline, EguiRenderer, EguiRenderingOptions};
use rust_gpu_bindless_egui::winit_integration::EguiWinitContext;
use rust_gpu_bindless_winit::ash::{
	AshSwapchain, AshSwapchainParams, SwapchainImageFormatPreference, ash_enumerate_required_extensions,
};
use rust_gpu_bindless_winit::event_loop::{EventLoopExecutor, event_loop_init};
use rust_gpu_bindless_winit::window_ref::WindowRef;
use smallvec::SmallVec;
use std::collections::HashSet;
use std::f32::consts::PI;
use std::sync::Arc;
use std::sync::mpsc::Receiver;
use std::time::Instant;
use winit::event::{DeviceEvent, ElementState, Event, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::raw_window_handle::HasDisplayHandle;
use winit::window::WindowAttributes;

pub fn main() {
	event_loop_init(|event_loop, events| async {
		main_loop(event_loop, events).await.unwrap();
	});
}

pub async fn main_loop(event_loop: EventLoopExecutor, events: Receiver<Event<()>>) -> anyhow::Result<()> {
	if matches!(debugger(), Debuggers::RenderDoc) {
		unsafe {
			// renderdoc does not yet support wayland
			std::env::remove_var("WAYLAND_DISPLAY");
			std::env::set_var("ENABLE_VULKAN_RENDERDOC_CAPTURE", "1");
		}
	}

	let (window, window_extensions) = event_loop
		.spawn(|e| {
			let window = e.create_window(WindowAttributes::default().with_title("swapchain triangle"))?;
			let extensions = ash_enumerate_required_extensions(e.display_handle()?.as_raw())?;
			Ok::<_, anyhow::Error>((WindowRef::new(Arc::new(window)), extensions))
		})
		.await?;

	let bindless = unsafe {
		BindlessInstance::<Ash>::new(
			ash_init_single_graphics_queue(AshSingleGraphicsQueueCreateInfo {
				instance_extensions: window_extensions,
				extensions: &[ash::khr::swapchain::NAME],
				debug: debugger(),
				..AshSingleGraphicsQueueCreateInfo::default()
			})?,
			DescriptorCounts::REASONABLE_DEFAULTS,
		)
	};

	let mut swapchain = unsafe {
		let bindless2 = bindless.clone();
		AshSwapchain::new(&bindless, &event_loop, window.clone(), move |surface, _| {
			AshSwapchainParams::automatic_best(
				&bindless2,
				surface,
				BindlessImageUsage::COLOR_ATTACHMENT,
				SwapchainImageFormatPreference::SRGB,
			)
		})
	}
	.await?;

	let egui_renderer = EguiRenderer::new(bindless.clone());
	let egui_pipeline = EguiRenderPipeline::new(egui_renderer.clone(), Some(swapchain.params().format), None);
	let mut egui_ctx = event_loop
		.spawn(move |e| {
			EguiWinitContext::new(
				egui_renderer.clone(),
				egui::Context::default(),
				e,
				window.get(e).clone(),
			)
		})
		.await;

	let mut renderer = TriangleRenderer::new(&bindless, swapchain.params().format)?;
	let mut camera = FreeCameraController::new(Vec3::new(-0.91287905, 1.7548301, -2.8095529), -0.43, -2.8274333);
	let mut fps_counter = FpsCounter::default();
	let mut last_frame = Instant::now();

	'outer: loop {
		let now = Instant::now();
		let dt = (now - last_frame).as_secs_f32();
		last_frame = now;

		for event in events.try_iter() {
			swapchain.handle_input(&event);
			egui_ctx.on_event(&event);
			camera.handle_event(&event);
			if let Event::WindowEvent {
				event: WindowEvent::KeyboardInput { event, .. },
				..
			} = &event
			{
				if event.state == ElementState::Pressed
					&& matches!(event.physical_key, PhysicalKey::Code(KeyCode::Space))
				{
					renderer.toggle_display_mode();
				}
			}
			if let Event::WindowEvent {
				event: WindowEvent::CloseRequested,
				..
			} = &event
			{
				break 'outer;
			}
		}

		camera.update(dt);
		fps_counter.update(dt);
		let fps_value = fps_counter.fps();
		let frame_ms = fps_counter.frame_ms();
		let display_mode = renderer.display_mode_name();

		let egui_render = egui_ctx.run(|ctx| {
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
		})?;

		let rt = swapchain.acquire_image(None).await?;
		let rt = renderer.draw(rt, &camera).await?;
		let rt = bindless.execute(|cmd| {
			let mut rt = rt.access_dont_care::<ColorAttachment>(cmd)?;
			egui_render
				.draw(
					&egui_pipeline,
					cmd,
					Some(&mut rt),
					None,
					EguiRenderingOptions::default(),
				)
				.expect("egui overlay draw failed");
			Ok(rt.transition::<Present>()?.into_desc())
		})?;
		swapchain.present_image(rt)?;
	}

	Ok(())
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
const CAMERA_FOV_Y_RADIANS: f32 = 11.0 * PI / 180.0;

impl<P: BindlessPipelinePlatform> TriangleRenderer<P> {
	pub fn new(bindless: &Bindless<P>, rt_format: Format) -> anyhow::Result<Self> {
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

	pub async fn draw(
		&mut self,
		rt: MutDesc<P, MutImage<Image2d>>,
		camera: &FreeCameraController,
	) -> anyhow::Result<MutDesc<P, MutImage<Image2d>>> {
		let (cam_right, cam_up, cam_forward) = camera.basis();
		let camera_world = Mat4::from_cols(
			cam_right.extend(0.0),
			cam_up.extend(0.0),
			(-cam_forward).extend(0.0),
			camera.position.extend(1.0),
		);
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
								camera_fov_y_radians: CAMERA_FOV_Y_RADIANS,
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

// ---------------------------------------------------------------------------
// Free camera controller (ported from Bevy's FreeCameraController)
// ---------------------------------------------------------------------------

const RADIANS_PER_DOT: f32 = 1.0 / 180.0;

pub struct FreeCameraController {
	// State
	pub position: Vec3,
	pub pitch: f32,
	pub yaw: f32,
	pub velocity: Vec3,
	pub speed_multiplier: f32,

	// Input accumulation
	keys_held: HashSet<KeyCode>,
	mouse_delta: Vec2,
	scroll_delta: f32,
	cursor_grabbed: bool,
	right_mouse_held: bool,

	// Config (matching Bevy defaults)
	pub sensitivity: f32,
	pub walk_speed: f32,
	pub run_speed: f32,
	pub scroll_factor: f32,
	pub friction: f32,
}

impl FreeCameraController {
	pub fn new(position: Vec3, pitch: f32, yaw: f32) -> Self {
		Self {
			position,
			pitch,
			yaw,
			velocity: Vec3::ZERO,
			speed_multiplier: 1.0,
			keys_held: HashSet::new(),
			mouse_delta: Vec2::ZERO,
			scroll_delta: 0.0,
			cursor_grabbed: false,
			right_mouse_held: false,
			sensitivity: 0.2,
			walk_speed: 5.0,
			run_speed: 15.0,
			scroll_factor: 0.04879016,
			friction: 40.0,
		}
	}

	/// Compute camera basis vectors (right, up, forward) from pitch/yaw.
	pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
		let rotation = Quat::from_euler(EulerRot::ZYX, 0.0, self.yaw, self.pitch);
		let forward = rotation * Vec3::NEG_Z;
		let right = rotation * Vec3::X;
		let up = rotation * Vec3::Y;
		(right, up, forward)
	}

	pub fn handle_event(&mut self, event: &Event<()>) {
		match event {
			Event::WindowEvent { event, .. } => match event {
				WindowEvent::KeyboardInput { event, .. } => {
					if let PhysicalKey::Code(code) = event.physical_key {
						match event.state {
							ElementState::Pressed => {
								self.keys_held.insert(code);
							}
							ElementState::Released => {
								self.keys_held.remove(&code);
							}
						}
					}
				}
				WindowEvent::MouseInput { state, button, .. } => {
					if *button == MouseButton::Right {
						self.right_mouse_held = *state == ElementState::Pressed;
						self.cursor_grabbed = self.right_mouse_held;
					}
				}
				WindowEvent::MouseWheel { delta, .. } => {
					let lines = match delta {
						MouseScrollDelta::LineDelta(_, y) => *y,
						MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 28.0,
					};
					self.scroll_delta += lines;
				}
				_ => {}
			},
			Event::DeviceEvent {
				event: DeviceEvent::MouseMotion { delta },
				..
			} => {
				self.mouse_delta.x += delta.0 as f32;
				self.mouse_delta.y += delta.1 as f32;
			}
			_ => {}
		}
	}

	pub fn update(&mut self, dt: f32) {
		// Scroll → speed multiplier (exponential, same as Bevy)
		if self.scroll_delta != 0.0 {
			self.speed_multiplier *= (self.scroll_factor * self.scroll_delta).exp();
			self.speed_multiplier = self.speed_multiplier.clamp(f32::EPSILON, f32::MAX);
			self.scroll_delta = 0.0;
		}

		// Mouse look (only when cursor is grabbed)
		if self.cursor_grabbed && self.mouse_delta != Vec2::ZERO {
			self.pitch =
				(self.pitch - self.mouse_delta.y * RADIANS_PER_DOT * self.sensitivity).clamp(-PI / 2.0, PI / 2.0);
			self.yaw -= self.mouse_delta.x * RADIANS_PER_DOT * self.sensitivity;
		}
		self.mouse_delta = Vec2::ZERO;

		// Keyboard → axis input
		let mut axis = Vec3::ZERO;
		if self.keys_held.contains(&KeyCode::KeyW) {
			axis.z += 1.0;
		}
		if self.keys_held.contains(&KeyCode::KeyS) {
			axis.z -= 1.0;
		}
		if self.keys_held.contains(&KeyCode::KeyD) {
			axis.x += 1.0;
		}
		if self.keys_held.contains(&KeyCode::KeyA) {
			axis.x -= 1.0;
		}
		if self.keys_held.contains(&KeyCode::KeyE) {
			axis.y += 1.0;
		}
		if self.keys_held.contains(&KeyCode::KeyQ) {
			axis.y -= 1.0;
		}

		// Velocity update (same logic as Bevy free camera)
		if axis != Vec3::ZERO {
			let max_speed = if self.keys_held.contains(&KeyCode::ShiftLeft) {
				self.run_speed
			} else {
				self.walk_speed
			} * self.speed_multiplier;
			self.velocity = axis.normalize() * max_speed;
		} else {
			// Exponential friction decay
			let decay = (-self.friction * dt).exp();
			self.velocity *= decay;
			if self.velocity.length_squared() < 1e-6 {
				self.velocity = Vec3::ZERO;
			}
		}

		// Apply movement in camera-local space
		if self.velocity != Vec3::ZERO {
			let (right, _up, forward) = self.basis();
			self.position +=
				self.velocity.x * dt * right + self.velocity.y * dt * Vec3::Y + self.velocity.z * dt * forward;
		}
	}
}
