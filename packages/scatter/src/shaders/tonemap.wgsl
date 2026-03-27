// Fullscreen tone-map pass — reads an HDR rgba32float texture and maps
// to [0,1] for the canvas.  Uses a single oversized triangle (3 vertices,
// no vertex buffer) to cover the viewport.
//
// Mode 0 (additive): per-channel exponential compression 1-exp(-x).
//   Preserves hue ratios, dense regions desaturate toward white.
// Mode 1 (over / subtractive): simple clamp to [0,1].
//   Values are already bounded; tone mapping is identity.

struct Params {
  mode: f32,
}

@group(0) @binding(0) var hdrTexture: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  // Oversized triangle covering the full clip space:
  //   vi=0 → (-1, -1)   vi=1 → (3, -1)   vi=2 → (-1, 3)
  let x = f32(i32(vi) / 2) * 4.0 - 1.0;
  let y = f32(i32(vi) % 2) * 4.0 - 1.0;
  return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
  let hdr = textureLoad(hdrTexture, vec2u(pos.xy), 0);

  var mapped: vec3f;
  if params.mode < 0.5 {
    // Additive: exponential compression preserving density structure
    mapped = 1.0 - exp(-hdr.rgb);
  } else {
    // Over / subtractive: values already in [0,1], just clamp
    mapped = clamp(hdr.rgb, vec3f(0.0), vec3f(1.0));
  }
  return vec4f(mapped, 1.0);
}
