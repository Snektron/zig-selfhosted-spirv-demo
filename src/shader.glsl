#version 460

layout(local_size_x = 8, local_size_y = 8) in;

layout(binding = 0, rgba8) uniform image2D render_target;

void main() {
    const uvec2 uv = gl_GlobalInvocationID.xy;
    imageStore(render_target, ivec2(uv), vec4(1, 0, 1, 1));
}
