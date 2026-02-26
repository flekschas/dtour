import marimo

__generated_with = "0.20.2"
app = marimo.App(width="full")


@app.cell
def _():
    import anywidget
    import traitlets as t
    import numpy as np

    class HandshakeWidget(anywidget.AnyWidget):
        _esm = """
        function render({ model, el }) {
          const log = (msg) => {
            const p = document.createElement("p");
            p.style.fontFamily = "monospace";
            p.style.margin = "4px 0";
            p.textContent = new Date().toISOString().slice(11,23) + " " + msg;
            el.appendChild(p);
          };

          log("JS: widget mounted");

          // Listen for custom messages from Python
          model.on("msg:custom", (msg, buffers) => {
            log("JS: got msg:custom type=" + JSON.stringify(msg.type) + " buffers=" + buffers.length);
            if (buffers.length > 0) {
              const view = buffers[0];
              const arr = new Float32Array(
                view.buffer.slice(view.byteOffset, view.byteOffset + view.byteLength)
              );
              log("JS: buffer has " + arr.length + " floats: [" + Array.from(arr).join(", ") + "]");
            }
          });

          // Send ready signal to Python via model.send()
          log("JS: sending ready");
          model.send({ type: "ready" });
        }
        export default { render };
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._buf = np.array([1.0, 2.0, 3.0], dtype=np.float32).tobytes()
            self.on_msg(self._handle_custom_msg)

        def _handle_custom_msg(self, data, buffers):
            """Correct 2-arg signature for anywidget on_msg callback."""
            self.send({ "type": data })
            if data.get("type") == "ready":
                print(f"Python: JS ready, sending {len(self._buf)} bytes")
                self.send({"type": "data"}, buffers=[self._buf])

    return (HandshakeWidget,)


@app.cell
def _(HandshakeWidget):
    w = HandshakeWidget()
    w
    return


if __name__ == "__main__":
    app.run()
