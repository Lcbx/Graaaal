<VertexInput> iRect: vec4f;
<VertexInput> iColor: vec4f;
<VertexOutput> @builtin(position) position: vec4f;
<VertexOutput> color: vec4f;

@vertex
fn vertex(input: VertexInput, @builtin(vertex_index) index: u32) -> VertexOutput {
    let corners = array<vec2f, 6>(
        vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
        vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0),
    );
    let pixel = input.iRect.xy + corners[index] * input.iRect.zw;
    var output: VertexOutput;
    output.position = vec4f(pixel.x * 2.0 - 1.0, 1.0 - pixel.y * 2.0, 0.0, 1.0);
    output.color = input.iColor;
    return output;
}

@fragment
fn fragment(input: VertexOutput) -> @location(0) vec4f {
    return input.color;
}
