#version 300 es
// Fullscreen tone-map pass — oversized triangle (3 vertices, no vertex buffer).
// Same geometry as tonemap.wgsl.

void main() {
  // Oversized triangle covering the full clip space:
  //   vi=0 -> (-1, -1)   vi=1 -> (3, -1)   vi=2 -> (-1, 3)
  float x = float(gl_VertexID / 2) * 4.0 - 1.0;
  float y = float(gl_VertexID % 2) * 4.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
