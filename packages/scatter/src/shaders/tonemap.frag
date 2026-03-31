#version 300 es
// Fullscreen tone-map pass — reads HDR float texture and maps to [0,1].
// Same logic as tonemap.wgsl.
//
// Mode 0 (additive): per-channel exponential compression 1-exp(-x).
//   Preserves hue ratios, dense regions desaturate toward white.
// Mode 1 (normal / subtractive): simple clamp to [0,1].

precision highp float;

uniform sampler2D u_hdrTexture;
uniform float u_mode;

layout(location = 0) out vec4 fragColor;

void main() {
  vec4 hdr = texelFetch(u_hdrTexture, ivec2(gl_FragCoord.xy), 0);

  vec3 mapped;
  if (u_mode < 0.5) {
    // Additive: exponential compression preserving density structure
    mapped = 1.0 - exp(-hdr.rgb);
  } else {
    // Normal / Subtractive: values already in [0,1], just clamp
    mapped = clamp(hdr.rgb, vec3(0.0), vec3(1.0));
  }
  fragColor = vec4(mapped, 1.0);
}
