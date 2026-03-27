"""Minimal anywidget to test custom message round-trip in Marimo."""

import anywidget


class HandshakeWidget(anywidget.AnyWidget):
    _esm = """
    function render({ model, el }) {
      const log = (msg) => {
        const p = document.createElement("p");
        p.textContent = msg;
        el.appendChild(p);
      };

      log("JS: widget mounted");

      // Listen for custom messages from Python
      model.on("msg:custom", (msg, buffers) => {
        log("JS: got msg:custom type=" + msg.type + " buffers=" + buffers.length);
        if (buffers.length > 0) {
          const view = buffers[0];
          const arr = new Float32Array(
            view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
          );
          log("JS: buffer has " + arr.length + " floats, first=" + arr[0]);
        }
      });

      // Send ready signal to Python
      log("JS: sending ready via model.send()");
      model.send({ type: "ready" });
    }
    export default { render };
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.on_msg(self._on_msg)

    def _on_msg(self, widget, msg, buffers):
        print(f"Python: got msg from JS: {msg}")
        if msg.get("content", {}).get("type") == "ready":
            print("Python: JS is ready, sending binary buffer")
            import numpy as np
            buf = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
            self.send({"type": "data"}, buffers=[buf])
            print(f"Python: sent {len(buf)} bytes")


if __name__ == "__main__":
    w = HandshakeWidget()
    w
