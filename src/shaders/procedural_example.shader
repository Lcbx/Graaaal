#include "draw_common.shaderlib"
#include "utils.shaderlib"
#include "instance_transform.shaderlib"

@group(1) @binding(0) var<storage, read> instances: array<MeshInstance>;
@group(1) @binding(1) var<storage, read> visible_instances: array<u32>;

@vertex
fn vertex(@location(0) position: vec3f, @builtin(instance_index) instance: u32) -> @builtin(position) vec4f {
    return vec4f(instance_position(instances[visible_instances[instance]], position), 1.0);
}

@fragment
fn fragment() -> @location(0) vec4f {
    return vec4f(1.0, 0.25, 0.1, 1.0);
}
