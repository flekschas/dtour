// Point renderer — instanced quads with SDF circle anti-aliasing
//
// Projects raw ND data to 2D inline using pre-adjusted basis weights and
// biases (normalization folded into the basis on CPU). This eliminates the
// separate compute-projection pass and its GPU barrier.
//
// Trade-off: the projection loop runs once per vertex (4x per point) since
// vertex shaders can't share results across a strip. This is a net win
// because eliminating the compute dispatch + barrier + projected-buffer
// read costs more than the redundant ALU, and GPUs are throughput-oriented.
// For very high dim counts (>200) this balance may shift — profile first.
//
// Applies 2D camera transform (pan/zoom/aspect). Each point is rendered as
// a unit quad (triangle strip, 4 vertices) scaled by point_size (NDC units).
//
// Color mapping: instead of a pre-computed N-element color buffer, the shader
// reads raw data values and looks up colors from a tiny LUT at render time.
// This eliminates the 20MB color buffer for 5M points and makes color changes
// instant (just swap the ~100-byte LUT + update uniforms).

struct Uniforms {
  point_size: f32,
  opacity: f32,
  // Color mode: 0 = uniform color, 1 = continuous, 2 = categorical
  color_mode: u32,
  useSelectionMask: f32,
  color: vec4f,
  useSubtractive: f32,
  num_points: u32,
  num_dims: u32,
  max_points: u32,
  bias: vec2f,
  // Color mapping params (used when color_mode > 0)
  color_column_offset: u32,  // columnIndex * num_points (continuous only)
  color_min: f32,
  color_range: f32,
  color_num_stops: u32,      // LUT size
  bias_z: f32,               // z-axis bias (3D mode only)
  // 2D colormap params (used when color_mode == 3)
  color_column_offset_v: u32, // second column index * num_points
  color_min_v: f32,
  color_range_v: f32,
  color2d_map_index: u32,     // 0-5 = LUT-based, 6 = oklab_polar
}

struct Camera {
  pan_x: f32,
  pan_y: f32,
  zoom: f32,
  aspect: f32,
  viewport_height: f32,
  // NDC-space Y offset applied after zoom — used to shift the viewport
  // down when a toolbar covers the top of the canvas.
  inset_offset_y: f32,
  // Zoom multiplier for the inset — scales content to fit in the visible
  // area below the toolbar (typically (H - toolbar) / H).
  inset_zoom: f32,
  // 3D camera rotation: 0.0 = 2D mode (skip z projection), 1.0 = 3D mode
  use_3d: f32,
  // 3×3 rotation matrix (column-major, each column padded to vec4f for alignment)
  rot_col0: vec3f,
  _pad0: f32,
  rot_col1: vec3f,
  _pad1: f32,
  rot_col2: vec3f,
  _pad2: f32,
}

@group(0) @binding(0) var<uniform> uni: Uniforms;
// Raw ND data (column-major): dim0[0..N], dim1[0..N], ..., dimP[0..N]
@group(0) @binding(1) var<storage, read> data: array<f32>;
@group(0) @binding(2) var<uniform> camera: Camera;
// Colormap/palette LUT — packed RGBA8 u32 entries (25 for continuous, up to 256 for categorical)
@group(0) @binding(3) var<storage, read> colorLut: array<u32>;
// Bit-packed selection mask (1 bit per point, 32 per u32)
@group(0) @binding(4) var<storage, read> selectionMask: array<u32>;
// Adjusted basis weights (normalization folded in):
// first num_dims = x-weights, next num_dims = y-weights
// adj_basis[d] = basis[d] / range[d] * viewport_scale
@group(0) @binding(5) var<storage, read> adj_basis: array<f32>;
// Categorical index buffer (per-point category index, used when color_mode == 2)
@group(0) @binding(6) var<storage, read> catIndices: array<u32>;

struct VertOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) point_color: vec4f,
  @location(2) @interpolate(flat) selected: u32,
  @location(3) @interpolate(flat) effective_opacity: f32,
}

// Triangle strip: 4 vertices forming a unit quad [-1, 1]²
fn quad_vertex(vi: u32) -> vec2f {
  switch vi {
    case 0u: { return vec2f(-1.0, -1.0); }
    case 1u: { return vec2f( 1.0, -1.0); }
    case 2u: { return vec2f(-1.0,  1.0); }
    default: { return vec2f( 1.0,  1.0); }
  }
}

// Unpack RGBA8 from u32 (0xAABBGGRR little-endian)
fn unpackColor(packed: u32) -> vec4f {
  let r = f32(packed & 0xFFu) / 255.0;
  let g = f32((packed >> 8u) & 0xFFu) / 255.0;
  let b = f32((packed >> 16u) & 0xFFu) / 255.0;
  let a = f32((packed >> 24u) & 0xFFu) / 255.0;
  return vec4f(r, g, b, a);
}

// PCG hash — fast, high-quality deterministic random from instance index.
// Used for vertex-shader decimation when num_points > max_points.
fn pcg_hash(input: u32) -> u32 {
  var state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn random_float(seed: u32) -> f32 {
  return f32(pcg_hash(seed)) / 4294967295.0;
}

// ─── 2D Colormap helpers ──────────────────────────────────────────────────

// Read f32 from the colorLut buffer (which stores u32). Used for 2D colormap LUT curves.
fn lut_f32(index: u32) -> f32 {
  return bitcast<f32>(colorLut[index]);
}

// Linearly interpolate a 16-entry LUT curve stored in colorLut at the given base offset.
fn lut_interp(base: u32, t: f32) -> f32 {
  let pos = t * 15.0;
  let lo = min(u32(floor(pos)), 14u);
  let hi = lo + 1u;
  return mix(lut_f32(base + lo), lut_f32(base + hi), fract(pos));
}

// Reconstruct a 2D colormap color from SVD rank-1 LUT curves stored in colorLut.
// Layout: R_X[16] R_Y[16] G_X[16] G_Y[16] B_X[16] B_Y[16] B_X2[16] B_Y2[16]
fn colormap2d_lut(u: f32, v: f32) -> vec3f {
  let r = lut_interp(0u, u) * lut_interp(16u, v);
  let g = lut_interp(32u, u) * lut_interp(48u, v);
  let b = lut_interp(64u, u) * lut_interp(80u, v)
        + lut_interp(96u, u) * lut_interp(112u, v); // rank-2 correction
  return clamp(vec3f(r, g, b), vec3f(0.0), vec3f(1.0));
}

// Oklab → linear sRGB conversion
fn oklab_to_linear(L: f32, a: f32, b: f32) -> vec3f {
  let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
  let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
  let s_ = L - 0.0894841775 * a - 1.2914855480 * b;
  let l = l_ * l_ * l_;
  let m = m_ * m_ * m_;
  let s = s_ * s_ * s_;
  return vec3f(
     4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );
}

// Oklab polar 2D colormap: hue = angle, chroma = radius, lightness varies
fn colormap2d_oklab_polar(u: f32, v: f32) -> vec3f {
  let cx = u - 0.5;
  let cy = v - 0.5;
  let radius = min(length(vec2f(cx, cy)) * 2.0, 1.0);
  let angle = atan2(cy, cx);
  let chroma = radius * 0.25;
  let lightness = 0.80 - radius * 0.30;
  let a_val = chroma * cos(angle);
  let b_val = chroma * sin(angle);
  let linear = oklab_to_linear(lightness, a_val, b_val);
  // Linear to sRGB gamma (approximate)
  return pow(clamp(linear, vec3f(0.0), vec3f(1.0)), vec3f(1.0 / 2.2));
}

// Dispatch 2D colormap by index
fn colormap2d(map_index: u32, u: f32, v: f32) -> vec3f {
  if map_index == 6u {
    return colormap2d_oklab_polar(u, v);
  }
  return colormap2d_lut(u, v);
}

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertOut {
  // Vertex-shader decimation: skip expensive projection for excess points.
  // Early exit before the ND projection loop saves ~50 cycles per rejected point.
  if (uni.max_points > 0u && uni.num_points > uni.max_points) {
    let threshold = f32(uni.max_points) / f32(uni.num_points);
    if (random_float(ii) > threshold) {
      var out: VertOut;
      out.clip_pos = vec4f(0.0, 0.0, 0.0, 0.0);
      out.uv = vec2f(0.0);
      out.point_color = vec4f(0.0);
      out.selected = 0u;
      out.effective_opacity = 0.0;
      return out;
    }
  }

  // Inline ND → 2D projection with pre-adjusted basis.
  // CPU folds normalization into the basis weights and biases so the
  // hot loop is just: x += raw * adjBasisX[d], y += raw * adjBasisY[d]
  // Must produce the same result as compute-projection.wgsl (used for lasso).
  let N = uni.num_points;
  let p = uni.num_dims;
  var x = uni.bias.x;
  var y = uni.bias.y;
  for (var d = 0u; d < p; d++) {
    let raw = data[d * N + ii];
    x += raw * adj_basis[d];
    y += raw * adj_basis[p + d];
  }

  // 3D camera rotation: project z-axis, rotate, then discard z (orthographic).
  if (camera.use_3d > 0.5) {
    var z = uni.bias_z;
    for (var d = 0u; d < p; d++) {
      z += data[d * N + ii] * adj_basis[2u * p + d];
    }
    let pos3 = vec3f(x, y, z);
    let rot = mat3x3f(camera.rot_col0, camera.rot_col1, camera.rot_col2);
    let rotated = rot * pos3;
    x = rotated.x;
    y = rotated.y;
    // z discarded — orthographic projection
  }

  // Apply camera: translate then scale, with aspect correction on x.
  // inset_zoom shrinks the scene to fit below the toolbar;
  // inset_offset_y shifts it down so it's centered in the visible area.
  let iz = camera.inset_zoom;
  let center = vec2f(
    (x + camera.pan_x) * camera.zoom * iz / camera.aspect,
    (y + camera.pan_y) * camera.zoom * iz + camera.inset_offset_y,
  );

  let q = quad_vertex(vi);
  // Clamp point size to [2, 32] pixels (physical).
  // min_ndc ensures points are always at least 2px — 1px quads can miss pixel
  // centers entirely depending on sub-pixel alignment, producing no fragments.
  let min_ndc = 4.0 / camera.viewport_height;
  let max_ndc = 64.0 / camera.viewport_height;
  let point_size = clamp(uni.point_size, min_ndc, max_ndc);
  // Aspect-correct the offset so circles stay circular on non-square canvases
  let offset = vec2f(q.x / camera.aspect, q.y) * (point_size * 0.5);

  // Scale opacity by zoom² to keep constant visual fill density (Reusser).
  // Zooming out compresses points → more overlap → must reduce opacity.
  let z = camera.zoom * camera.inset_zoom;
  let eff_opacity = uni.opacity * z * z;

  // Resolve per-point color from LUT
  var col: vec4f;
  if uni.color_mode == 1u {
    // Continuous: read raw value from color column, normalize, interpolate LUT
    let raw = data[uni.color_column_offset + ii];
    let inv_range = select(0.0, 1.0 / uni.color_range, uni.color_range > 0.0);
    let t = clamp((raw - uni.color_min) * inv_range, 0.0, 1.0);
    let stops = f32(uni.color_num_stops - 1u);
    let pos = t * stops;
    let idx = min(u32(floor(pos)), uni.color_num_stops - 2u);
    let frac = pos - f32(idx);
    let c0 = unpackColor(colorLut[idx]);
    let c1 = unpackColor(colorLut[idx + 1u]);
    col = mix(c0, c1, frac);
  } else if uni.color_mode == 2u {
    // Categorical: read category index, direct palette lookup
    let catIdx = catIndices[ii];
    col = unpackColor(colorLut[catIdx % uni.color_num_stops]);
  } else if uni.color_mode == 3u {
    // 2D colormap: read two columns, normalize to UV, look up from 2D colormap
    let raw_u = data[uni.color_column_offset + ii];
    let raw_v = data[uni.color_column_offset_v + ii];
    let inv_range_u = select(0.0, 1.0 / uni.color_range, uni.color_range > 0.0);
    let inv_range_v = select(0.0, 1.0 / uni.color_range_v, uni.color_range_v > 0.0);
    let u_val = clamp((raw_u - uni.color_min) * inv_range_u, 0.0, 1.0);
    let v_val = clamp((raw_v - uni.color_min_v) * inv_range_v, 0.0, 1.0);
    let rgb = colormap2d(uni.color2d_map_index, u_val, v_val);
    col = vec4f(rgb, 1.0);
  } else {
    col = uni.color;
  }

  // Read selection
  var sel: u32 = 1u;
  if uni.useSelectionMask > 0.5 {
    sel = (selectionMask[ii / 32u] >> (ii % 32u)) & 1u;
  }

  var out: VertOut;
  out.clip_pos = vec4f(center + offset, 0.0, 1.0);
  out.uv = q;
  out.point_color = col;
  out.selected = sel;
  out.effective_opacity = eff_opacity;
  return out;
}

// Shared fragment helper — computes edge, selection factor, and intensity.
fn fragment_intensity(
  uv: vec2f,
  point_alpha: f32,
  selected: u32,
  effective_opacity: f32,
) -> f32 {
  let dist = length(uv);
  // Smooth anti-aliased edge — naturally 0 for dist >= 1.0, so no discard needed.
  // Avoiding discard preserves early-Z and SIMD wavefront occupancy.
  let edge = 1.0 - smoothstep(0.75, 1.0, dist);

  // Selection: boost selected, dim unselected
  var sel_factor = 1.0;
  if uni.useSelectionMask > 0.5 {
    if selected == 0u {
      sel_factor = 0.1;
    } else {
      sel_factor = 1.0 / max(effective_opacity, 0.01);
    }
  }

  return edge * effective_opacity * point_alpha * sel_factor;
}

// ─── Fragment: single-target (additive / subtractive) ──────────────────────

@fragment
fn fs_main(
  @location(0) uv: vec2f,
  @location(1) point_color: vec4f,
  @location(2) @interpolate(flat) selected: u32,
  @location(3) @interpolate(flat) effective_opacity: f32,
) -> @location(0) vec4f {
  let intensity = fragment_intensity(uv, point_color.a, selected, effective_opacity);
  // Subtractive mode (Reusser): output complement color so that
  // reverse-subtract blend (dst - src) on a white bg yields the correct hue.
  let rgb = select(point_color.rgb, vec3f(1.0) - point_color.rgb, uni.useSubtractive > 0.5);
  return vec4f(rgb * intensity, intensity);
}
