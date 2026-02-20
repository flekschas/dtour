// Point renderer — instanced quads with SDF circle discard
//
// Reads pre-projected 2D positions from a storage buffer (output of the
// compute-projection shader). Applies 2D camera transform (pan/zoom/aspect).
// Each point is rendered as a unit quad scaled by point_size (NDC units).

struct Uniforms {
  point_size: f32,
  opacity: f32,
  _pad0: f32,
  _pad1: f32,
  color: vec4f,
}

struct Camera {
  pan_x: f32,
  pan_y: f32,
  zoom: f32,
  aspect: f32,
}

@group(0) @binding(0) var<uniform> uni: Uniforms;
// projected[i*2] = x, projected[i*2+1] = y — output from compute shader
@group(0) @binding(1) var<storage, read> projected: array<f32>;
@group(0) @binding(2) var<uniform> camera: Camera;

struct VertOut {
  @builtin(position) clip_pos: vec4f,
  @location(0) uv: vec2f,
}

// Two triangles making a unit square [-1, 1]^2
fn quad_vertex(vi: u32) -> vec2f {
  switch vi {
    case 0u: { return vec2f(-1.0, -1.0); }
    case 1u: { return vec2f( 1.0, -1.0); }
    case 2u: { return vec2f(-1.0,  1.0); }
    case 3u: { return vec2f(-1.0,  1.0); }
    case 4u: { return vec2f( 1.0, -1.0); }
    default: { return vec2f( 1.0,  1.0); }
  }
}

@vertex
fn vs_main(
  @builtin(vertex_index) vi: u32,
  @builtin(instance_index) ii: u32,
) -> VertOut {
  let px = projected[ii * 2u];
  let py = projected[ii * 2u + 1u];

  // Apply camera: translate then scale, with aspect correction on x
  let center = vec2f(
    (px + camera.pan_x) * camera.zoom / camera.aspect,
    (py + camera.pan_y) * camera.zoom,
  );

  let q = quad_vertex(vi);
  // Aspect-correct the offset so circles stay circular on non-square canvases
  let offset = vec2f(q.x / camera.aspect, q.y) * (uni.point_size * 0.5);

  var out: VertOut;
  out.clip_pos = vec4f(center + offset, 0.0, 1.0);
  out.uv = q;
  return out;
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let dist = length(uv);
  if dist > 1.0 {
    discard;
  }
  // Smooth anti-aliased edge
  let edge = 1.0 - smoothstep(0.75, 1.0, dist);
  let alpha = edge * uni.opacity * uni.color.a;
  // Premultiplied alpha output
  return vec4f(uni.color.rgb * alpha, alpha);
}
