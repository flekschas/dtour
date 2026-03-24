export type FileFormat = 'arrow' | 'parquet';

// "ARROW1" = 0x41 52 52 4F 57 31
const ARROW_MAGIC = [0x41, 0x52, 0x52, 0x4f, 0x57, 0x31];
// "PAR1"   = 0x50 41 52 31
const PARQUET_MAGIC = [0x50, 0x41, 0x52, 0x31];

const matchesMagic = (bytes: Uint8Array, magic: number[]): boolean =>
  bytes.length >= magic.length && magic.every((b, i) => bytes[i] === b);

export const detectFormat = (buffer: ArrayBuffer): FileFormat => {
  const bytes = new Uint8Array(buffer, 0, Math.min(8, buffer.byteLength));

  // Arrow file format: starts with "ARROW1\0\0"
  if (matchesMagic(bytes, ARROW_MAGIC)) return 'arrow';

  // Parquet: "PAR1" at the start
  if (matchesMagic(bytes, PARQUET_MAGIC)) return 'parquet';

  // Arrow IPC stream: starts with 0xFFFFFFFF (continuation token) or a direct message length
  // Treat as Arrow as a fallback — flechette handles both file and stream formats
  return 'arrow';
};
