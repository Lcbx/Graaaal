#include "draw_common.shaderlib"
#include "utils_transform.shaderlib"
#include "instance_transform.shaderlib"

storage<1, 0, read> instances: array<MeshInstance>;
storage<1, 1, read> visible_instances: array<u32>;

@vertex
fn vertex(@location(0) position: vec3f, @builtin(instance_index) instance: u32) -> @builtin(position) vec4f {
    return vec4f(instance_position(instances[visible_instances[instance]], position), 1.0);
}

@fragment
fn fragment() -> @location(0) vec4f {
    return vec4f(1.0, 0.25, 0.1, 1.0);
}
