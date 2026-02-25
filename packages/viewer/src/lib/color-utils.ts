/** Check if a string is a hex color (e.g. #ff6600, #f60). */
export const isHexColor = (s: string): boolean => /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.test(s);

/** Parse a hex color string to [r, g, b] in 0-1 range. */
export const hexToRgb = (hex: string): [number, number, number] => {
  let h = hex.slice(1);
  if (h.length === 3) {
    h = h[0]! + h[0]! + h[1]! + h[1]! + h[2]! + h[2]!;
  }
  const n = Number.parseInt(h, 16);
  return [(n >> 16) / 255, ((n >> 8) & 0xff) / 255, (n & 0xff) / 255];
};
