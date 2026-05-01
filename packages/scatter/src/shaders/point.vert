#version 300 es
// Point renderer — gl.POINTS with SDF circle anti-aliasing
//
// WebGL2 equivalent of point.wgsl. Projects raw ND data to 2D inline using
// pre-adjusted basis weights and biases (normalization folded into the basis
// on CPU). Uses gl.POINTS with gl_PointSize for native point expansion —
// 1 vertex per point instead of 4 (instanced quads in WebGPU).
//
// Data is stored in a tiled R32F texture to handle point counts exceeding
// MAX_TEXTURE_SIZE. Layout: width=texWidth, height=numDims*tileRows.
// For point idx: col = idx % texWidth, row = d * tileRows + idx / texWidth.
//
// Color mapping: instead of a pre-computed N-element color texture, the shader
// reads raw data values and looks up colors from a tiny LUT at render time.

precision highp float;
precision highp int;
precision highp sampler2D;
precision highp usampler2D;

// Raw ND data (column-major, tiled).
// R32F texture: width = texWidth, height = numDims * tileRows.
uniform sampler2D u_data;

// Adjusted basis weights (normalization folded in).
// Max 128 dims — for higher, fall back to a basis texture.
uniform float u_adjBasisX[128];
uniform float u_adjBasisY[128];
uniform vec2 u_bias;

// Style
uniform float u_pointSize;    // NDC units
uniform float u_opacity;
uniform vec4 u_color;          // uniform point color (RGBA 0-1)
uniform float u_useSubtractive;
uniform float u_scaleOpacityByZoom; // 1.0 = auto (zoom² density scaling), 0.0 = manual

// Camera
uniform vec2 u_pan;
uniform float u_zoom;
uniform float u_aspect;
uniform float u_viewportHeight;
uniform float u_insetOffsetY;
uniform float u_insetZoom;

// Data dimensions
uniform int u_numPoints;
uniform int u_numDims;

// Decimation: 0 = disabled
uniform int u_maxPoints;

// Tiling: texture width (min(numPoints, MAX_TEXTURE_SIZE))
uniform int u_texWidth;

// Color mapping — tiny LUT replaces N-element packed color texture
// 0 = uniform color, 1 = continuous, 2 = categorical, 3 = 2D colormap
uniform int u_colorMode;
uniform int u_colorColumnIndex;   // which data column to read (continuous only)
uniform float u_colorMin;
uniform float u_colorRange;
uniform int u_colorNumStops;      // LUT size
uniform usampler2D u_colorLutTex; // packed RGBA u32 LUT (width=numStops, height=1)
uniform usampler2D u_catIndexTex; // categorical indices (R32UI, tiled like data tex)

// 2D colormap params (used when u_colorMode == 3)
uniform int u_colorColumnIndexV;  // second data column
uniform float u_colorMinV;
uniform float u_colorRangeV;
uniform int u_color2dMapIndex;    // 0-5 = LUT-based, 6 = oklab_polar
uniform float u_color2dLut[128];  // SVD rank-1 LUT curves (8 × 16 entries)

// Selection mask (bit-packed u32 in R32UI texture, width = ceil(N/32), height = 1)
uniform float u_useSelectionMask;
uniform usampler2D u_selectionTex;

flat out float v_effOpacity;
flat out vec4 v_color;
flat out uint v_selected;
flat out float v_useSubtractive;
flat out float v_useSelectionMask;

// Unpack RGBA8 from u32 (0xAABBGGRR little-endian)
vec4 unpackColor(uint packed) {
  float r = float(packed & 0xFFu) / 255.0;
  float g = float((packed >> 8u) & 0xFFu) / 255.0;
  float b = float((packed >> 16u) & 0xFFu) / 255.0;
  float a = float((packed >> 24u) & 0xFFu) / 255.0;
  return vec4(r, g, b, a);
}

// PCG hash — fast, high-quality deterministic random from vertex index.
// Used for vertex-shader decimation when numPoints > maxPoints.
uint pcg_hash(uint v) {
  uint state = v * 747796405u + 2891336453u;
  uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

float random_float(uint seed) {
  return float(pcg_hash(seed)) / 4294967295.0;
}

// ── 2D colormap helpers ──

// Linearly interpolate within a 16-entry LUT curve starting at base offset.
float lut2d_interp(int base, float t) {
  float pos = t * 15.0;
  int lo = min(int(floor(pos)), 14);
  int hi = lo + 1;
  return mix(u_color2dLut[base + lo], u_color2dLut[base + hi], fract(pos));
}

// Reconstruct 2D color from SVD rank-1 LUT curves.
// Layout: R_X[16] R_Y[16] G_X[16] G_Y[16] B_X[16] B_Y[16] B_X2[16] B_Y2[16]
vec3 colormap2d_lut(float u, float v) {
  float r = lut2d_interp(0, u) * lut2d_interp(16, v);
  float g = lut2d_interp(32, u) * lut2d_interp(48, v);
  float b = lut2d_interp(64, u) * lut2d_interp(80, v)
          + lut2d_interp(96, u) * lut2d_interp(112, v);
  return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

// Oklab → linear sRGB conversion
vec3 oklabToLinear(vec3 lab) {
  float l_ = lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z;
  float m_ = lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z;
  float s_ = lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z;
  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;
  return vec3(
     4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );
}

// Oklab polar 2D colormap: hue = angle, chroma = radius, lightness varies
vec3 colormap2d_oklab_polar(float u, float v) {
  float cx = u - 0.5;
  float cy = v - 0.5;
  float radius = min(length(vec2(cx, cy)) * 2.0, 1.0);
  float angle = atan(cy, cx);
  float chroma = radius * 0.25;
  float L = mix(0.3, 0.85, 1.0 - radius);
  float a = chroma * cos(angle);
  float b = chroma * sin(angle);
  vec3 linear = oklabToLinear(vec3(L, a, b));
  return pow(clamp(linear, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
}

vec3 colormap2d(int mapIndex, float u, float v) {
  if (mapIndex == 6) {
    return colormap2d_oklab_polar(u, v);
  }
  return colormap2d_lut(u, v);
}

void main() {
  int idx = gl_VertexID;

  // Vertex-shader decimation: skip expensive projection for excess points.
  if (u_maxPoints > 0 && u_numPoints > u_maxPoints) {
    float threshold = float(u_maxPoints) / float(u_numPoints);
    if (random_float(uint(idx)) > threshold) {
      gl_Position = vec4(0.0, 0.0, 0.0, 0.0);
      gl_PointSize = 0.0;
      v_effOpacity = 0.0;
      v_color = vec4(0.0);
      v_selected = 0u;
      v_useSubtractive = 0.0;
      v_useSelectionMask = 0.0;
      return;
    }
  }

  // Tiled texture coordinates: col = idx % texWidth, tileRow = idx / texWidth
  int tileRows = (u_numPoints + u_texWidth - 1) / u_texWidth;
  int tx = idx % u_texWidth;
  int tileRow = idx / u_texWidth;

  // Inline ND -> 2D projection with pre-adjusted basis.
  // CPU folds normalization into the basis weights and biases so the
  // hot loop is just: x += raw * adjBasisX[d], y += raw * adjBasisY[d]
  float x = u_bias.x;
  float y = u_bias.y;
  for (int d = 0; d < u_numDims; d++) {
    float raw = texelFetch(u_data, ivec2(tx, d * tileRows + tileRow), 0).r;
    x += raw * u_adjBasisX[d];
    y += raw * u_adjBasisY[d];
  }

  // Apply camera: translate then scale, with aspect correction on x.
  float iz = u_insetZoom;
  vec2 center = vec2(
    (x + u_pan.x) * u_zoom * iz / u_aspect,
    (y + u_pan.y) * u_zoom * iz + u_insetOffsetY
  );

  gl_Position = vec4(center, 0.0, 1.0);

  // Point size: clamp to [2, 32] physical pixels (same as WGSL)
  float min_ndc = 4.0 / u_viewportHeight;
  float max_ndc = 64.0 / u_viewportHeight;
  float ps = clamp(u_pointSize, min_ndc, max_ndc);
  // gl_PointSize is in physical pixels, convert from NDC
  gl_PointSize = ps * u_viewportHeight * 0.5;

  // Scale opacity by zoom^2 for constant visual fill density (Reusser).
  // Only applied for auto-computed opacity; manual values are used as-is.
  float z = u_zoom * u_insetZoom;
  float zoomScale = u_scaleOpacityByZoom > 0.5 ? z * z : 1.0;
  v_effOpacity = u_opacity * zoomScale;

  // Resolve per-point color from LUT
  if (u_colorMode == 1) {
    // Continuous: read raw value from color column, normalize, interpolate LUT
    float raw = texelFetch(u_data, ivec2(tx, u_colorColumnIndex * tileRows + tileRow), 0).r;
    float invRange = u_colorRange > 0.0 ? 1.0 / u_colorRange : 0.0;
    float t = clamp((raw - u_colorMin) * invRange, 0.0, 1.0);
    float stops = float(u_colorNumStops - 1);
    float pos = t * stops;
    int idx0 = min(int(floor(pos)), u_colorNumStops - 2);
    float frac = pos - float(idx0);
    vec4 c0 = unpackColor(texelFetch(u_colorLutTex, ivec2(idx0, 0), 0).r);
    vec4 c1 = unpackColor(texelFetch(u_colorLutTex, ivec2(idx0 + 1, 0), 0).r);
    v_color = mix(c0, c1, frac);
  } else if (u_colorMode == 2) {
    // Categorical: read category index from tiled texture, direct palette lookup
    uint catIdx = texelFetch(u_catIndexTex, ivec2(tx, tileRow), 0).r;
    int lutIdx = int(catIdx) % u_colorNumStops;
    v_color = unpackColor(texelFetch(u_colorLutTex, ivec2(lutIdx, 0), 0).r);
  } else if (u_colorMode == 3) {
    // 2D colormap: read two columns, normalize to UV, look up from 2D colormap
    float rawU = texelFetch(u_data, ivec2(tx, u_colorColumnIndex * tileRows + tileRow), 0).r;
    float rawV = texelFetch(u_data, ivec2(tx, u_colorColumnIndexV * tileRows + tileRow), 0).r;
    float invRangeU = u_colorRange > 0.0 ? 1.0 / u_colorRange : 0.0;
    float invRangeV = u_colorRangeV > 0.0 ? 1.0 / u_colorRangeV : 0.0;
    float uVal = clamp((rawU - u_colorMin) * invRangeU, 0.0, 1.0);
    float vVal = clamp((rawV - u_colorMinV) * invRangeV, 0.0, 1.0);
    vec3 rgb2d = colormap2d(u_color2dMapIndex, uVal, vVal);
    v_color = vec4(rgb2d, 1.0);
  } else {
    v_color = u_color;
  }

  // Read selection bit (not tiled — bit-packed width is small)
  v_selected = 1u;
  if (u_useSelectionMask > 0.5) {
    int word_idx = idx / 32;
    uint word = texelFetch(u_selectionTex, ivec2(word_idx, 0), 0).r;
    v_selected = (word >> uint(idx % 32)) & 1u;
  }

  v_useSubtractive = u_useSubtractive;
  v_useSelectionMask = u_useSelectionMask;
}
