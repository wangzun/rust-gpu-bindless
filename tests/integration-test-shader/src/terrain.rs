use glam::{Vec2, Vec2Swizzles, Vec3, Vec3Swizzles, Vec4, Vec4Swizzles};
use rust_gpu_bindless_macros::{BufferStruct, bindless};
use rust_gpu_bindless_shaders::descriptor::{Descriptors, Image, Image2d, ImageType, Sampler, StrongDesc, UnsafeDesc};
#[cfg(target_arch = "spirv")]
use spirv_std::num_traits::Float;

type SampledImage2d = <Image2d as ImageType>::SampledSpvImage;

#[derive(Copy, Clone, BufferStruct)]
pub struct BufferAParam {
	pub resolution: Vec2,
	pub time: f32,
}

#[derive(Copy, Clone, BufferStruct)]
pub struct BufferBParam {
	pub resolution: Vec2,
	pub time: f32,
}

#[derive(Copy, Clone, BufferStruct)]
pub struct Param {
	pub map_image: UnsafeDesc<Image<Image2d>>,
	pub detail_image: UnsafeDesc<Image<Image2d>>,
	pub sampler: StrongDesc<Sampler>,
	pub resolution: Vec2,
	pub map_resolution: Vec2,
	pub time: f32,
	pub display_mode: u32,
	pub camera_pos: Vec3,
	pub camera_right: Vec3,
	pub camera_up: Vec3,
	pub camera_forward: Vec3,
}

#[derive(Copy, Clone)]
struct TerrainParams {
	time_scroll_offset: Vec2,
	water_height: f32,
	erosion_scale: f32,
	erosion_strength: f32,
	erosion_detail: f32,
	erosion_rounding: Vec4,
}

#[derive(Copy, Clone)]
struct HeightSample {
	height: f32,
	erosion: f32,
	ridgemap: f32,
	trees: f32,
}

#[derive(Copy, Clone)]
struct BoxIntersection {
	hit: bool,
	t_near: f32,
	t_far: f32,
	normal: Vec3,
}

#[derive(Copy, Clone)]
struct MarchResult {
	hit: bool,
	t: f32,
	normal: Vec3,
	material: i32,
	shadow_term: f32,
}

const PI: f32 = 3.14159265358979;
const TAU: f32 = 6.28318530717959;

const M_GROUND: i32 = 0;
const M_STRATA: i32 = 1;
const M_WATER: i32 = 2;

const BOX_SIZE: Vec3 = Vec3::new(0.5, 1.0, 0.5);

const CLIFF_COLOR: Vec3 = Vec3::new(0.22, 0.20, 0.20);
const DIRT_COLOR: Vec3 = Vec3::new(0.60, 0.50, 0.40);
const TREE_COLOR: Vec3 = Vec3::new(0.12, 0.26, 0.10);
const GRASS_COLOR1: Vec3 = Vec3::new(0.15, 0.30, 0.10);
const GRASS_COLOR2: Vec3 = Vec3::new(0.40, 0.50, 0.20);
const SAND_COLOR: Vec3 = Vec3::new(0.80, 0.70, 0.60);
const WATER_COLOR: Vec3 = Vec3::new(0.00, 0.05, 0.10);
const WATER_SHORE_COLOR: Vec3 = Vec3::new(0.00, 0.25, 0.25);
const SUN_COLOR: Vec3 = Vec3::new(2.0, 1.96, 1.90);
const AMBIENT_COLOR: Vec3 = Vec3::new(0.03, 0.05, 0.07);

const GRASS_HEIGHT: f32 = 0.465;
const DRAINAGE_WIDTH: f32 = 0.3;
const RAYMARCH_QUALITY: f32 = 1.6;
const CAMERA_FOV_DEGREES: f32 = 11.0;

const HEIGHT_FREQUENCY: f32 = 3.0;
const HEIGHT_AMP: f32 = 0.125;
const HEIGHT_OCTAVES: usize = 3;
const HEIGHT_LACUNARITY: f32 = 2.0;
const HEIGHT_GAIN: f32 = 0.1;
const TERRAIN_HEIGHT_OFFSET: Vec2 = Vec2::new(-0.65, 0.0);

const EROSION_GULLY_WEIGHT: f32 = 0.5;
const EROSION_CELL_SCALE: f32 = 0.7;
const EROSION_NORMALIZATION: f32 = 0.5;
const EROSION_OCTAVES: usize = 5;
const EROSION_LACUNARITY: f32 = 2.0;
const EROSION_GAIN: f32 = 0.5;
const EROSION_ONSET: Vec4 = Vec4::new(1.25, 1.25, 2.8, 1.5);
const EROSION_ASSUMED_SLOPE: Vec2 = Vec2::new(0.7, 1.0);

const C_RAYLEIGH: Vec3 = Vec3::new(5.802e-6, 13.558e-6, 33.100e-6);
const C_MIE: Vec3 = Vec3::new(3.996e-6, 3.996e-6, 3.996e-6);

#[inline(always)]
fn clamp01(x: f32) -> f32 {
	x.clamp(0.0, 1.0)
}

#[inline(always)]
fn quantize_scroll_offset(scroll_offset: Vec2, resolution: Vec2) -> Vec2 {
	Vec2::new(
		(scroll_offset.x * resolution.x).round() / resolution.x,
		(scroll_offset.y * resolution.y).round() / resolution.y,
	)
}

#[inline(always)]
fn scroll_offset_parts(scroll_offset: Vec2, resolution: Vec2) -> (Vec2, Vec2) {
	let scroll_offset_int = quantize_scroll_offset(scroll_offset, resolution);
	(scroll_offset_int, scroll_offset - scroll_offset_int)
}

#[inline(always)]
fn exp3(v: Vec3) -> Vec3 {
	Vec3::new(v.x.exp(), v.y.exp(), v.z.exp())
}

#[inline(always)]
fn camera_focal_scale(fov_degrees: f32) -> f32 {
	((90.0 - fov_degrees * 0.5) * PI / 180.0).tan()
}

#[inline(always)]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
	let t = clamp01((x - edge0) / (edge1 - edge0));
	t * t * (3.0 - 2.0 * t)
}

#[inline(always)]
fn mix_f32(x: f32, y: f32, a: f32) -> f32 {
	x * (1.0 - a) + y * a
}

#[inline(always)]
fn mix_vec2(x: Vec2, y: Vec2, a: f32) -> Vec2 {
	x * (1.0 - a) + y * a
}

#[inline(always)]
fn mix_vec3(x: Vec3, y: Vec3, a: f32) -> Vec3 {
	x * (1.0 - a) + y * a
}

#[inline(always)]
fn animate_wave_to(mut current: f32, target: f32, mut time: f32) -> f32 {
	time = clamp01(time);
	current = mix_f32(current, target, 0.5 - 0.5 * (3.0 * time * PI).cos());
	current
}

#[inline(always)]
fn animate_lo_hi(mut current: f32, lo: f32, hi: f32, time: f32) -> f32 {
	let original = current;
	current = mix_f32(current, lo, smoothstep(0.0, 1.0, time));
	current = mix_f32(current, hi, smoothstep(0.0, 1.0, time - 2.0));
	current = mix_f32(current, original, smoothstep(0.0, 1.0, time - 4.0));
	current
}

#[inline(always)]
fn rem_euclid_f32(x: f32, rhs: f32) -> f32 {
	let r = x % rhs;
	if r < 0.0 { r + rhs.abs() } else { r }
}

#[inline(always)]
fn hash(mut x: Vec2) -> Vec2 {
	let k = Vec2::new(0.3183099, 0.3678794);
	x = x * k + k.yx();
	Vec2::splat(-1.0) + 2.0 * (16.0 * k * Vec2::splat((x.x * x.y * (x.x + x.y)).fract())).fract()
}

#[inline(always)]
fn noised(p: Vec2) -> Vec3 {
	let i = p.floor();
	let f = p.fract();

	let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
	let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

	let ga = hash(i + Vec2::new(0.0, 0.0));
	let gb = hash(i + Vec2::new(1.0, 0.0));
	let gc = hash(i + Vec2::new(0.0, 1.0));
	let gd = hash(i + Vec2::new(1.0, 1.0));

	let va = ga.dot(f - Vec2::new(0.0, 0.0));
	let vb = gb.dot(f - Vec2::new(1.0, 0.0));
	let vc = gc.dot(f - Vec2::new(0.0, 1.0));
	let vd = gd.dot(f - Vec2::new(1.0, 1.0));

	let value = va + u.x * (vb - va) + u.y * (vc - va) + u.x * u.y * (va - vb - vc + vd);
	let deriv = ga
		+ u.x * (gb - ga)
		+ u.y * (gc - ga)
		+ u.x * u.y * (ga - gb - gc + gd)
		+ du * (u.yx() * (va - vb - vc + vd) + Vec2::new(vb, vc) - Vec2::splat(va));

	Vec3::new(value, deriv.x, deriv.y)
}

#[inline(always)]
fn phacelle_noise(p: Vec2, norm_dir: Vec2, freq: f32, mut offset: f32, normalization: f32) -> Vec4 {
	let side_dir = norm_dir.yx() * Vec2::new(-1.0, 1.0) * freq * TAU;
	offset *= TAU;

	let p_int = p.floor();
	let p_frac = p.fract();
	let mut phase_dir = Vec2::ZERO;
	let mut weight_sum = 0.0;

	for i in -1..=2 {
		for j in -1..=2 {
			let grid_offset = Vec2::new(i as f32, j as f32);
			let grid_point = p_int + grid_offset;
			let random_offset = hash(grid_point) * 0.5;
			let vector_from_cell_point = p_frac - grid_offset - random_offset;
			let sqr_dist = vector_from_cell_point.dot(vector_from_cell_point);
			let mut weight = (-sqr_dist * 2.0).exp();
			weight = (weight - 0.01111).max(0.0);
			weight_sum += weight;

			let wave_input = vector_from_cell_point.dot(side_dir) + offset;
			phase_dir += Vec2::new(wave_input.cos(), wave_input.sin()) * weight;
		}
	}

	let interpolated = phase_dir / weight_sum.max(1e-6);
	let magnitude = interpolated.dot(interpolated).sqrt().max(1.0 - normalization);
	Vec4::new(
		interpolated.x / magnitude,
		interpolated.y / magnitude,
		side_dir.x,
		side_dir.y,
	)
}

#[inline(always)]
fn pow_inv(t: f32, power: f32) -> f32 {
	1.0 - (1.0 - clamp01(t)).powf(power)
}

#[inline(always)]
fn ease_out(t: f32) -> f32 {
	let v = 1.0 - clamp01(t);
	1.0 - v * v
}

#[inline(always)]
fn smooth_start(t: f32, smoothing: f32) -> f32 {
	if t >= smoothing {
		t - 0.5 * smoothing
	} else {
		0.5 * t * t / smoothing.max(1e-6)
	}
}

#[inline(always)]
fn safe_normalize(n: Vec2) -> Vec2 {
	let len = n.length();
	if len.abs() > 1e-10 { n / len } else { n }
}

#[inline(always)]
fn fractal_noise(p: Vec2, freq: f32, octaves: usize, lacunarity: f32, gain: f32) -> Vec3 {
	let mut n = Vec3::ZERO;
	let mut nf = freq;
	let mut na = 1.0;
	for _ in 0..octaves {
		n += noised(p * nf) * na * Vec3::new(1.0, nf, nf);
		na *= gain;
		nf *= lacunarity;
	}
	n
}

#[inline(always)]
fn get_trees_amount(height: f32, normal_y: f32, occlusion: f32, ridge_map: f32, water_height: f32) -> f32 {
	((smoothstep(
		GRASS_HEIGHT + 0.05,
		GRASS_HEIGHT + 0.01,
		height + 0.01 + (occlusion - 0.8) * 0.05,
	) * smoothstep(0.0, 0.4, occlusion)
		* smoothstep(0.95, 1.0, normal_y)
		* smoothstep(-1.4, 0.0, ridge_map)
		* smoothstep(water_height + 0.000, water_height + 0.007, height))
		- 0.5) / 0.6
}

#[inline(always)]
fn sky_color(rd: Vec3, sun: Vec3) -> Vec3 {
	let costh = rd.dot(sun);
	AMBIENT_COLOR * PI * (1.0 - costh.abs() * 0.8)
}

#[inline(always)]
fn tonemap_aces(x: Vec3) -> Vec3 {
	let a = 2.51;
	let b = 0.03;
	let c = 2.43;
	let d = 0.59;
	let e = 0.14;
	(x * (a * x + b)) / (x * (c * x + d) + e)
}

#[inline(always)]
fn d_ggx(linear_roughness: f32, no_h: f32) -> f32 {
	let one_minus_no_h_squared = 1.0 - no_h * no_h;
	let a = no_h * linear_roughness;
	let k = linear_roughness / (one_minus_no_h_squared + a * a);
	k * k * (1.0 / PI)
}

#[inline(always)]
fn v_smith_ggx_correlated(linear_roughness: f32, no_v: f32, no_l: f32) -> f32 {
	let a2 = linear_roughness * linear_roughness;
	let ggx_v = no_l * ((no_v - a2 * no_v) * no_v + a2).sqrt();
	let ggx_l = no_v * ((no_l - a2 * no_l) * no_l + a2).sqrt();
	0.5 / (ggx_v + ggx_l).max(1e-5)
}

#[inline(always)]
fn pow5(x: f32) -> f32 {
	let x2 = x * x;
	x2 * x2 * x
}

#[inline(always)]
fn f_schlick_vec(f0: Vec3, vo_h: f32) -> Vec3 {
	f0 + (Vec3::ONE - f0) * pow5(1.0 - vo_h)
}

#[inline(always)]
fn f_schlick_scalar(f0: f32, f90: f32, vo_h: f32) -> f32 {
	f0 + (f90 - f0) * pow5(1.0 - vo_h)
}

#[inline(always)]
fn fd_burley(linear_roughness: f32, no_v: f32, no_l: f32, lo_h: f32) -> f32 {
	let f90 = 0.5 + 2.0 * linear_roughness * lo_h * lo_h;
	let light_scatter = f_schlick_scalar(1.0, f90, no_l);
	let view_scatter = f_schlick_scalar(1.0, f90, no_v);
	light_scatter * view_scatter * (1.0 / PI)
}

#[inline(always)]
fn fd_lambert() -> f32 {
	1.0 / PI
}

#[inline(always)]
fn shade(diffuse: Vec3, f0: Vec3, smoothness: f32, n: Vec3, v: Vec3, l: Vec3, lc: Vec3) -> Vec3 {
	let h = (v + l).normalize();
	let no_v = n.dot(v).abs() + 1e-5;
	let no_l = clamp01(n.dot(l));
	let no_h = clamp01(n.dot(h));
	let lo_h = clamp01(l.dot(h));
	let roughness = 1.0 - smoothness;
	let linear_roughness = roughness * roughness;
	let d = d_ggx(linear_roughness, no_h);
	let vis = v_smith_ggx_correlated(linear_roughness, no_v, no_l);
	let fresnel = f_schlick_vec(f0, lo_h);
	let fr = (d * vis) * fresnel;
	let fd = diffuse * fd_burley(linear_roughness, no_v, no_l, lo_h);
	(fd + fr) * lc * no_l
}

#[inline(always)]
fn phase_rayleigh(costh: f32) -> f32 {
	3.0 * (1.0 + costh * costh) / (16.0 * PI)
}

#[inline(always)]
fn phase_mie(costh: f32, mut g: f32) -> f32 {
	g = g.min(0.9381);
	let k = 1.55 * g - 0.55 * g * g * g;
	let kcosth = k * costh;
	(1.0 - k * k) / ((4.0 * PI) * (1.0 - kcosth) * (1.0 - kcosth))
}

#[inline(always)]
fn box_intersection(ro: Vec3, rd: Vec3, box_size: Vec3) -> BoxIntersection {
	let inv = Vec3::new(1.0 / rd.x, 1.0 / rd.y, 1.0 / rd.z);
	let n = inv * ro;
	let k = inv.abs() * box_size;
	let t1 = -n - k;
	let t2 = -n + k;
	let t_near = t1.x.max(t1.y).max(t1.z);
	let t_far = t2.x.min(t2.y).min(t2.z);
	if t_near > t_far || t_far < 0.0 {
		return BoxIntersection {
			hit: false,
			t_near: -1.0,
			t_far: -1.0,
			normal: Vec3::ZERO,
		};
	}

	let normal = if t1.x >= t1.y && t1.x >= t1.z {
		Vec3::new(-rd.x.signum(), 0.0, 0.0)
	} else if t1.y >= t1.z {
		Vec3::new(0.0, -rd.y.signum(), 0.0)
	} else {
		Vec3::new(0.0, 0.0, -rd.z.signum())
	};

	BoxIntersection {
		hit: true,
		t_near,
		t_far,
		normal,
	}
}

#[inline(always)]
fn erosion_filter(
	p: Vec2,
	mut height_and_slope: Vec3,
	mut fade_target: f32,
	strength: f32,
	gully_weight: f32,
	detail: f32,
	rounding: Vec4,
	onset: Vec4,
	assumed_slope: Vec2,
	scale: f32,
	octaves: usize,
	lacunarity: f32,
	gain: f32,
	cell_scale: f32,
	normalization: f32,
) -> (Vec4, f32, f32) {
	let mut strength = strength * scale;
	fade_target = fade_target.clamp(-1.0, 1.0);

	let input_height_and_slope = height_and_slope;
	let mut freq = 1.0 / (scale * cell_scale);
	let slope_length = height_and_slope.yz().length().max(1e-10);
	let mut magnitude = 0.0;
	let mut rounding_mult = 1.0;

	let rounding_for_input = mix_f32(rounding.y, rounding.x, clamp01(fade_target + 0.5)) * rounding.z;
	let mut combi_mask = ease_out(smooth_start(slope_length * onset.x, rounding_for_input * onset.x));

	let mut ridge_map_combi_mask = ease_out(slope_length * onset.z);
	let mut ridge_map_fade_target = fade_target;

	let mut gully_slope = mix_vec2(
		height_and_slope.yz(),
		height_and_slope.yz() / slope_length * assumed_slope.x,
		assumed_slope.y,
	);

	for _ in 0..octaves {
		let mut phacelle = phacelle_noise(p * freq, safe_normalize(gully_slope), cell_scale, 0.25, normalization);
		let phacelle_deriv = Vec2::new(phacelle.z, phacelle.w) * -freq;
		phacelle.z = phacelle_deriv.x;
		phacelle.w = phacelle_deriv.y;

		let sloping = phacelle.y.abs();
		gully_slope += phacelle.y.signum() * phacelle_deriv * strength * gully_weight;

		let gullies = Vec3::new(phacelle.x, phacelle.y * phacelle_deriv.x, phacelle.y * phacelle_deriv.y);
		let faded_gullies = mix_vec3(Vec3::new(fade_target, 0.0, 0.0), gullies * gully_weight, combi_mask);
		height_and_slope += faded_gullies * strength;
		magnitude += strength;
		fade_target = faded_gullies.x;

		let rounding_for_octave = mix_f32(rounding.y, rounding.x, clamp01(phacelle.x + 0.5)) * rounding_mult;
		let new_mask = ease_out(smooth_start(sloping * onset.y, rounding_for_octave * onset.y));
		combi_mask = pow_inv(combi_mask, detail) * new_mask;

		ridge_map_fade_target = mix_f32(ridge_map_fade_target, gullies.x, ridge_map_combi_mask);
		let new_ridge_map_mask = ease_out(sloping * onset.w);
		ridge_map_combi_mask *= new_ridge_map_mask;

		strength *= gain;
		freq *= lacunarity;
		rounding_mult *= rounding.w;
	}

	let ridge_map = ridge_map_fade_target * (1.0 - ridge_map_combi_mask);
	let debug = fade_target;
	(
		(height_and_slope - input_height_and_slope).extend(magnitude),
		ridge_map,
		debug,
	)
}

#[inline(always)]
fn terrain_params(time: f32) -> TerrainParams {
	let terrain_time = rem_euclid_f32(time * 0.5, 30.0);
	let erosion_scale = animate_lo_hi(0.15, 0.08, 0.25, terrain_time - 7.0);
	let erosion_strength = animate_lo_hi(0.22, 0.01, 0.10, terrain_time - 1.0);
	let erosion_detail = animate_lo_hi(1.5, 3.0, 0.7, terrain_time - 13.0);

	let mut ridge_rounding = 0.1;
	let mut crease_rounding = 0.0;
	crease_rounding = animate_wave_to(crease_rounding, 1.0, terrain_time - 19.0);
	ridge_rounding = animate_wave_to(ridge_rounding, 1.0, terrain_time - 21.0);
	crease_rounding = animate_wave_to(crease_rounding, 0.0, terrain_time - 23.0);
	ridge_rounding = animate_wave_to(ridge_rounding, 0.1, terrain_time - 25.0);

	let water_cycle = rem_euclid_f32(time, 120.0);
	let water_height = 0.36 + 0.1 * (smoothstep(54.0, 60.0, water_cycle) - smoothstep(114.0, 120.0, water_cycle));

	let phase = time / 60.0 * TAU;
	let time_scroll_offset = Vec2::new(phase.cos() * 2.0, -phase.sin() * 0.1);

	TerrainParams {
		time_scroll_offset,
		water_height,
		erosion_scale,
		erosion_strength,
		erosion_detail,
		erosion_rounding: Vec4::new(ridge_rounding, crease_rounding, 0.1, 2.0),
	}
}

#[inline(always)]
fn height_sample(params: &TerrainParams, p: Vec2) -> HeightSample {
	let mut n =
		fractal_noise(p, HEIGHT_FREQUENCY, HEIGHT_OCTAVES, HEIGHT_LACUNARITY, HEIGHT_GAIN) * HEIGHT_AMP * Vec3::ONE;

	let fade_target = (n.x / (HEIGHT_AMP * 0.6)).clamp(-1.0, 1.0);
	n = Vec3::new(n.x * 0.5 + 0.5, n.y * 0.5, n.z * 0.5);

	let (h, ridge_map, _debug) = erosion_filter(
		p,
		n,
		fade_target,
		params.erosion_strength,
		EROSION_GULLY_WEIGHT,
		params.erosion_detail,
		params.erosion_rounding,
		EROSION_ONSET,
		EROSION_ASSUMED_SLOPE,
		params.erosion_scale,
		EROSION_OCTAVES,
		EROSION_LACUNARITY,
		EROSION_GAIN,
		EROSION_CELL_SCALE,
		EROSION_NORMALIZATION,
	);

	let offset = mix_f32(TERRAIN_HEIGHT_OFFSET.x, -fade_target, TERRAIN_HEIGHT_OFFSET.y) * h.w;
	let mut eroded = n.x + h.x + offset;
	let deriv = n.yz() + h.yz();

	let normal_y = 1.0 / (1.0 + deriv.dot(deriv)).sqrt();
	let erosion = if h.w > 1e-6 { (h.x / h.w).clamp(-1.0, 1.0) } else { 0.0 };

	let trees_amount = get_trees_amount(eroded, normal_y, erosion + 0.5, ridge_map, params.water_height);
	let tree_noise = noised((p + Vec2::splat(0.5)) * 200.0).x * 0.5 + 0.5;
	let tree_field = (1.0 - tree_noise.powf(2.0) - 1.0 + trees_amount) * 1.5;
	if tree_field > 0.0 {
		eroded += tree_field / 300.0;
	}

	HeightSample {
		height: eroded,
		erosion,
		ridgemap: ridge_map,
		trees: clamp01(tree_field * 0.5 + 0.5),
	}
}

#[inline(always)]
fn detail_texture(uv: Vec2) -> Vec3 {
	let mut color = Vec3::ZERO;
	let mut amplitude = 0.5;
	let mut freq = 2.0;
	for _ in 0..8 {
		color += noised(uv * freq) * amplitude;
		amplitude *= 0.95;
		freq *= 2.0;
	}
	color
}

#[inline(always)]
fn map_uv_from_world(p: Vec3, map_resolution: Vec2) -> Vec2 {
	let pixel = Vec2::ONE / map_resolution;
	let uv = p.xz() * (Vec2::ONE - pixel * 2.0) + Vec2::splat(0.5);
	uv.clamp(pixel, Vec2::ONE - pixel)
}

#[inline(always)]
fn scrolled_map_uv_from_world(p: Vec3, map_resolution: Vec2, scroll_offset_frac: Vec2) -> Vec2 {
	map_uv_from_world(p, map_resolution) + scroll_offset_frac
}

#[inline(always)]
fn sample_map_raw(img: &SampledImage2d, sampler: Sampler, uv: Vec2) -> Vec4 {
	img.sample(sampler, uv)
}

#[inline(always)]
fn sample_detail(img: &SampledImage2d, sampler: Sampler, uv: Vec2) -> Vec3 {
	sample_map_raw(img, sampler, uv).xyz()
}

#[inline(always)]
fn sample_map(img: &SampledImage2d, sampler: Sampler, map_resolution: Vec2, uv: Vec2) -> (f32, Vec3, f32, f32, f32) {
	let texel = Vec2::new(1.0 / map_resolution.x, 1.0 / map_resolution.y);
	let sample = sample_map_raw(img, sampler, uv);
	let height = sample.x;
	let erosion = sample.y;
	let ridgemap = sample.z;
	let trees = sample.w;

	let h1 = sample_map_raw(img, sampler, (uv + Vec2::new(texel.x, 0.0)).min(Vec2::ONE - texel)).x;
	let h2 = sample_map_raw(img, sampler, (uv + Vec2::new(0.0, texel.y)).min(Vec2::ONE - texel)).x;
	let v1 = Vec3::new(texel.x, 0.0, h1 - height);
	let v2 = Vec3::new(0.0, texel.y, h2 - height);
	let normal = v1.cross(v2).normalize().xzy();

	(height, normal, erosion, ridgemap, trees)
}

#[inline(always)]
fn map_height(img: &SampledImage2d, sampler: Sampler, uv: Vec2) -> f32 {
	sample_map_raw(img, sampler, uv).x
}

#[inline(always)]
fn march(
	img: &SampledImage2d,
	sampler: Sampler,
	map_resolution: Vec2,
	scroll_offset_frac: Vec2,
	ro: Vec3,
	rd: Vec3,
	water_height: f32,
) -> MarchResult {
	let box_hit = box_intersection(ro, rd, BOX_SIZE);
	if !box_hit.hit {
		return MarchResult {
			hit: false,
			t: -1.0,
			normal: Vec3::ZERO,
			material: M_GROUND,
			shadow_term: 9999.0,
		};
	}

	let t_start = box_hit.t_near.max(0.0) + 1e-2;
	let t_end = box_hit.t_far - 1e-2;

	let mut material = M_GROUND;
	let mut normal = Vec3::ZERO;
	let mut shadow_term: f32 = 9999.0;
	let mut step_size = 0.0;
	let mut step_scale = 1.0 / RAYMARCH_QUALITY;
	let samples = (44.0 * RAYMARCH_QUALITY) as usize;
	let mut t = t_start;

	for i in 0..samples {
		let pos = ro + rd * t;
		let h = map_height(
			img,
			sampler,
			scrolled_map_uv_from_world(pos, map_resolution, scroll_offset_frac),
		);
		let altitude = pos.y - h;

		if t > 1e-3 {
			shadow_term = shadow_term.min((altitude / t).max(0.0));
		}

		if altitude < 0.0 {
			if i == 0 {
				if pos.y < 0.35 {
					return MarchResult {
						hit: false,
						t: -1.0,
						normal: Vec3::ZERO,
						material,
						shadow_term: 9999.0,
					};
				}
				normal = box_hit.normal;
				material = M_STRATA;
				break;
			}

			step_scale *= 0.5;
			t -= step_size * step_scale;
		} else {
			step_size = altitude.abs() + (altitude.abs() * 0.01).min(1e-2);
			t += step_size * step_scale;
		}

		if t > t_end {
			break;
		}
	}

	if t > t_end {
		t = -1.0;
	}

	let water_hit = box_intersection(ro, rd, Vec3::new(BOX_SIZE.x, water_height, BOX_SIZE.z));
	if water_hit.hit {
		let water_t = water_hit.t_near.max(0.0);
		if (t < 0.0 || water_t < t) && material != M_STRATA {
			return MarchResult {
				hit: true,
				t: water_t,
				normal: water_hit.normal,
				material: M_WATER,
				shadow_term,
			};
		}
	}

	if t < 0.0 {
		MarchResult {
			hit: false,
			t: -1.0,
			normal: Vec3::ZERO,
			material,
			shadow_term,
		}
	} else {
		MarchResult {
			hit: true,
			t,
			normal,
			material,
			shadow_term,
		}
	}
}

#[inline(always)]
fn fullscreen_vertex(vertex_index: u32, out_position: &mut Vec4, out_uv: &mut Vec2) {
	let uv = Vec2::new(((vertex_index << 1) & 2) as f32, (vertex_index & 2) as f32);
	let pos = 2.0 * uv - Vec2::ONE;
	*out_position = pos.extend(0.0).extend(1.0);
	*out_uv = uv;
}

#[bindless(vertex())]
pub fn terrain_buffer_a_vertex(
	#[bindless(descriptors)] _: Descriptors<'_>,
	#[bindless(param)] _: &BufferAParam,
	#[spirv(vertex_index)] vertex_index: u32,
	#[spirv(position)] out_position: &mut Vec4,
	out_uv: &mut Vec2,
) {
	fullscreen_vertex(vertex_index, out_position, out_uv);
}

#[bindless(vertex())]
pub fn terrain_buffer_b_vertex(
	#[bindless(descriptors)] _: Descriptors<'_>,
	#[bindless(param)] _: &BufferBParam,
	#[spirv(vertex_index)] vertex_index: u32,
	#[spirv(position)] out_position: &mut Vec4,
	out_uv: &mut Vec2,
) {
	fullscreen_vertex(vertex_index, out_position, out_uv);
}

#[bindless(vertex())]
pub fn terrain_vertex(
	#[bindless(descriptors)] _: Descriptors<'_>,
	#[bindless(param)] _: &Param,
	#[spirv(vertex_index)] vertex_index: u32,
	#[spirv(position)] out_position: &mut Vec4,
	out_uv: &mut Vec2,
) {
	fullscreen_vertex(vertex_index, out_position, out_uv);
}

#[bindless(fragment())]
pub fn terrain_buffer_a_fragment(#[bindless(param)] param: &BufferAParam, in_uv: Vec2, out_color: &mut Vec4) {
	let params = terrain_params(param.time);
	let screen_uv = Vec2::new(in_uv.x, 1.0 - in_uv.y);
	let (scroll_offset_int, _) = scroll_offset_parts(params.time_scroll_offset, param.resolution);
	let p = screen_uv + scroll_offset_int;
	let sample = height_sample(&params, p);
	*out_color = Vec4::new(sample.height, sample.erosion, sample.ridgemap * 0.5 + 0.5, sample.trees);
}

#[bindless(fragment())]
pub fn terrain_buffer_b_fragment(#[bindless(param)] param: &BufferBParam, in_uv: Vec2, out_color: &mut Vec4) {
	let params = terrain_params(param.time);
	let screen_uv = Vec2::new(in_uv.x, 1.0 - in_uv.y);
	let (scroll_offset_int, _) = scroll_offset_parts(params.time_scroll_offset, param.resolution);
	let uv = screen_uv + scroll_offset_int;
	*out_color = detail_texture(uv).extend(1.0);
}

#[bindless(fragment())]
pub fn terrain_fragment(
	#[bindless(descriptors)] descriptors: Descriptors<'_>,
	#[bindless(param)] param: &Param,
	in_uv: Vec2,
	out_color: &mut Vec4,
) {
	let map_image = unsafe { param.map_image.to_transient_unchecked(&descriptors) };
	let detail_image = unsafe { param.detail_image.to_transient_unchecked(&descriptors) };
	let map_img = map_image.access(&descriptors);
	let detail_img = detail_image.access(&descriptors);
	let sampler = param.sampler.access(&descriptors);
	let res = param.resolution;
	let terrain = terrain_params(param.time);
	let (_, scroll_offset_frac) = scroll_offset_parts(terrain.time_scroll_offset, param.map_resolution);
	let screen_uv = Vec2::new(in_uv.x, 1.0 - in_uv.y);

	if param.display_mode == 1 {
		let map = sample_map_raw(map_img, sampler, screen_uv);
		*out_color = Vec4::new(map.x, clamp01(map.y * 0.5 + 0.5), clamp01(map.z), 1.0);
		return;
	}

	if param.display_mode == 2 {
		let detail = sample_detail(detail_img, sampler, screen_uv);
		*out_color = (detail * 0.5 + Vec3::splat(0.5)).extend(1.0);
		return;
	}

	let s = (-1.0 + 2.0 * screen_uv) * Vec2::new(res.x / res.y, 1.0);
	let camera_focal = camera_focal_scale(CAMERA_FOV_DEGREES);

	// Camera from host
	let ro = param.camera_pos;
	let cr = param.camera_right;
	let cu = param.camera_up;
	let cd = param.camera_forward;
	let rd = (s.x * cr + s.y * cu + camera_focal * cd).normalize();

	let sun = Vec3::new(-1.0, 0.4, 0.05).normalize();
	let fog_color = Vec3::ONE - exp3(sky_color(rd, sun) * -2.0);

	let hit = march(
		map_img,
		sampler,
		param.map_resolution,
		scroll_offset_frac,
		ro,
		rd,
		terrain.water_height,
	);
	let mut color = if hit.hit {
		let pos = ro + rd * hit.t;
		let (height, sampled_normal, erosion, ridgemap, trees) = sample_map(
			map_img,
			sampler,
			param.map_resolution,
			scrolled_map_uv_from_world(pos, param.map_resolution, scroll_offset_frac),
		);
		let mut normal = if hit.material == M_GROUND {
			sampled_normal
		} else {
			hit.normal
		};
		let diff = pos.y - height;

		let breakup = sample_detail(
			detail_img,
			sampler,
			scrolled_map_uv_from_world(pos, param.map_resolution, scroll_offset_frac),
		);
		if hit.material == M_WATER {
			normal.x += breakup.y * 0.1;
			normal.z += breakup.z * 0.1;
			normal = normal.normalize();
		}

		let drainage = clamp01((1.0 - clamp01(ridgemap / DRAINAGE_WIDTH)) * 1.5);
		let f0 = Vec3::splat(0.04);
		let mut smoothness = 0.0;
		let mut occlusion = 1.0;

		let diffuse_color = if hit.material == M_GROUND {
			occlusion = clamp01(erosion + 0.5);
			let mut diffuse_color = CLIFF_COLOR * smoothstep(0.4, 0.52, pos.y);
			diffuse_color = mix_vec3(
				diffuse_color,
				DIRT_COLOR,
				smoothstep(0.6, 0.0, occlusion + breakup.x * 1.5),
			);
			diffuse_color = mix_vec3(diffuse_color, Vec3::ONE, smoothstep(0.53, 0.6, pos.y + breakup.x * 0.1));
			diffuse_color = mix_vec3(
				diffuse_color,
				SAND_COLOR,
				smoothstep(
					terrain.water_height + 0.005,
					terrain.water_height,
					pos.y + breakup.x * 0.01,
				),
			);

			let grass_mix = mix_vec3(
				GRASS_COLOR1,
				GRASS_COLOR2,
				smoothstep(0.4, 0.6, pos.y - erosion * 0.05 + breakup.x * 0.3),
			);
			diffuse_color = mix_vec3(
				diffuse_color,
				grass_mix,
				smoothstep(
					GRASS_HEIGHT + 0.05,
					GRASS_HEIGHT + 0.02,
					pos.y + 0.01 + (occlusion - 0.8) * 0.05 - breakup.x * 0.02,
				) * smoothstep(0.8, 1.0, 1.0 - (1.0 - normal.y) * (1.0 - trees) + breakup.x * 0.1),
			);

			diffuse_color = mix_vec3(
				diffuse_color,
				TREE_COLOR * trees.powf(8.0),
				clamp01(trees * 2.2 - 0.8) * 0.6,
			);
			diffuse_color *= 1.0 + breakup.x * 0.5;
			mix_vec3(diffuse_color, Vec3::ONE, drainage)
		} else if hit.material == M_STRATA {
			let strata = Vec3::new((diff * 130.0).cos(), (diff * 190.0).cos(), (diff * 250.0).cos());
			let strata = Vec3::new(
				smoothstep(0.0, 1.0, strata.x),
				smoothstep(0.0, 1.0, strata.y),
				smoothstep(0.0, 1.0, strata.z),
			);
			let mut diffuse_color = Vec3::splat(0.3);
			diffuse_color = mix_vec3(diffuse_color, Vec3::splat(0.50), strata.x);
			diffuse_color = mix_vec3(diffuse_color, Vec3::splat(0.55), strata.y);
			diffuse_color = mix_vec3(diffuse_color, Vec3::splat(0.60), strata.z);
			diffuse_color * (diff * 10.0).exp() * Vec3::new(1.0, 0.9, 0.7)
		} else {
			let shore = if normal.y > 1e-2 { (-diff * 60.0).exp() } else { 0.0 };
			let foam = if normal.y > 1e-2 {
				smoothstep(0.005, 0.0, diff + breakup.x * 0.005)
			} else {
				0.0
			};
			smoothness = 0.95;
			let diffuse_color = mix_vec3(WATER_COLOR, WATER_SHORE_COLOR, shore);
			mix_vec3(diffuse_color, Vec3::ONE, foam)
		};

		let mut shadow = 1.0;
		if hit.material != M_STRATA {
			let shadow_hit = march(
				map_img,
				sampler,
				param.map_resolution,
				scroll_offset_frac,
				pos + Vec3::new(0.0, 1.0e-4, 0.0),
				sun,
				terrain.water_height,
			);
			shadow = 1.0 - (-shadow_hit.shadow_term * 20.0).exp();
		}

		let mut color = diffuse_color * sky_color(normal, sun) * fd_lambert();
		color *= occlusion;
		color += shade(diffuse_color, f0, smoothness, normal, -rd, sun, SUN_COLOR * shadow);
		color +=
			diffuse_color * SUN_COLOR * (normal.dot(sun * Vec3::new(1.0, -1.0, 1.0)) * 0.5 + 0.5) * fd_lambert() / PI;
		color += sky_color(rd.reflect(normal), sun) * 4.0 * f_schlick_vec(f0, clamp01((-rd).dot(normal)));

		let air_box = box_intersection(ro, rd, BOX_SIZE);
		if air_box.hit {
			let costh = rd.dot(sun);
			let phase_r = phase_rayleigh(costh);
			let phase_m = phase_mie(costh, 0.6);
			let ray_end = if hit.t > 0.0 { hit.t } else { air_box.t_far };
			let ray_length = (ray_end - air_box.t_near).max(0.0);
			let step_size = ray_length / 16.0;
			let mut optical_depth = Vec2::ZERO;
			let mut transmittance = Vec3::ONE;
			let mut scatter = Vec3::ZERO;

			for i in 0..16 {
				let p = ro + rd * (air_box.t_near + (i as f32 + 0.5) * step_size);
				let mut density = 1.0 - clamp01((p.y - 0.35).max(0.0) / 0.2);
				if p.y < 0.35 {
					density = 0.0;
				}

				let density_r = density * 1e5;
				let density_m = density * 1e5;
				optical_depth += step_size * Vec2::new(density_r, density_m);
				transmittance = exp3(-(optical_depth.x * C_RAYLEIGH + optical_depth.y * C_MIE));
				scatter += transmittance * C_RAYLEIGH * phase_r * density_r * step_size;
				scatter += transmittance * C_MIE * phase_m * density_m * step_size;
			}

			color = color * transmittance + scatter * 10.0;
		} else {
			let fog = 1.0 - (-hit.t * 0.12).exp();
			color = mix_vec3(color, fog_color, fog);
		}

		color
	} else {
		fog_color * (1.0 + (in_uv.y).powf(3.0) * 3.0) * 0.5
	};

	color = tonemap_aces(color);
	let gamma = Vec3::new(
		color.x.powf(1.0 / 2.2),
		color.y.powf(1.0 / 2.2),
		color.z.powf(1.0 / 2.2),
	);
	*out_color = gamma.extend(1.0);
}
