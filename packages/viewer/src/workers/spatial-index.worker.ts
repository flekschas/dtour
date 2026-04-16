/// <reference lib="webworker" />
import KDBush from 'kdbush';

export type BuildRequest = {
  positions: Float32Array;
  nodeSize: number;
  generation: number;
};

export type BuildResult = {
  /** Serialized kdbush tree — reconstruct with KDBush.from(indexData). */
  indexData: ArrayBuffer;
  /** Original projected positions (transferred back for nearest-among-hits lookup). */
  positions: Float32Array;
  generation: number;
};

self.onmessage = (e: MessageEvent<BuildRequest>) => {
  const { positions, nodeSize, generation } = e.data;
  const n = positions.length / 2;

  const index = new KDBush(n, nodeSize);
  for (let i = 0; i < n; i++) {
    index.add(positions[i * 2]!, positions[i * 2 + 1]!);
  }
  index.finish();

  const result: BuildResult = {
    indexData: index.data,
    positions,
    generation,
  };

  // Transfer both buffers — zero-copy back to main thread
  (self as DedicatedWorkerGlobalScope).postMessage(result, [index.data, positions.buffer]);
};
