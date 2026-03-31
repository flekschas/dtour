#version 300 es
// Point fragment shader — SDF circle anti-aliasing
//
// WebGL2 equivalent of point.wgsl fs_main. Uses gl_PointCoord for the
// SDF distance (maps [0,1] in gl.POINTS, remap to [-1,1]).

precision highp float;
precision highp int;

flat in float v_effOpacity;
flat in vec4 v_color;
flat in uint v_selected;
flat in float v_useSubtractive;
flat in float v_useSelectionMask;

layout(location = 0) out vec4 fragColor;

void main() {
  // gl_PointCoord is [0,1], remap to [-1,1] for SDF
  vec2 uv = gl_PointCoord * 2.0 - 1.0;
  float dist = length(uv);

  // Smooth anti-aliased edge — naturally 0 for dist >= 1.0, so no discard needed.
  // Avoiding discard preserves early-Z and SIMD wavefront occupancy.
  float edge = 1.0 - smoothstep(0.75, 1.0, dist);

  // Selection: boost selected, dim unselected
  float sel_factor = 1.0;
  if (v_useSelectionMask > 0.5) {
    if (v_selected == 0u) {
      sel_factor = 0.1;
    } else {
      sel_factor = 1.0 / max(v_effOpacity, 0.01);
    }
  }

  float intensity = edge * v_effOpacity * v_color.a * sel_factor;

  // Subtractive mode (Reusser): output complement color so that
  // reverse-subtract blend (dst - src) on a white bg yields the correct hue.
  vec3 rgb = v_color.rgb;
  if (v_useSubtractive > 0.5) {
    rgb = vec3(1.0) - rgb;
  }

  fragColor = vec4(rgb * intensity, intensity);
}
