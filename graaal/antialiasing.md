# Antialiasing

`AntialiasingPass` applies LXAA or FAAA to a single-sample color target.
`AntialiasingSettings` sets the LXAA edge threshold and line check, or the
FAAA edge threshold, quality (1–9), and edge display.

```python
from graaal.AntialiasingPass import AntialiasingPass, AntialiasingSettings

aa = AntialiasingPass(target_format="rgba8unorm", settings=AntialiasingSettings(faaa_quality=5))
aa.resize((width, height))
scene_output = aa.color_attachment()
# Render the scene to scene_output, then record the antialiasing pass.
aa.record(commands, output, "faaa")  # "lxaa" is also available
```

Call `set_settings(settings)` to change the parameters and `resize(size)` when
the output dimensions change. `record` writes to a single-sample color
attachment with the same format as the pass.
