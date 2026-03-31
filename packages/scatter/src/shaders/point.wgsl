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
