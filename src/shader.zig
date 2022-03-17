const Image = u32;

export fn shader() callconv(.Unspecified) void {
    const render_target_ptr = asm volatile(
        \\   %F32 = OpTypeFloat 32
        \\   %Img = OpTypeImage %F32 2D 0 0 0 2 Rgba8
        \\%ImgPtr = OpTypePointer UniformConstant %Img
        \\%target = OpVariable %ImgPtr UniformConstant
        \\          OpDecorate %target DescriptorSet 0
        \\          OpDecorate %target Binding 0
        : [_] "target" (->*const addrspace(.uniform) Image),
    );
    const gid_ptr = asm volatile(
        \\     %U32 = OpTypeInt 32 0
        \\   %V3U32 = OpTypeVector %U32 3
        \\%V3U32Ptr = OpTypePointer Input %V3U32
        \\     %gid = OpVariable %V3U32Ptr Input
        \\            OpDecorate %gid BuiltIn GlobalInvocationId
        : [_] "gid" (->*const addrspace(.input) @Vector(3, u32)),
    );

    const uv = @shuffle(u32, gid_ptr.*, undefined, @Vector(2, i32){0, 1});
    const color = @Vector(4, f32){1, 0, 1, 1};

    asm volatile(
        \\   %F32 = OpTypeFloat 32
        \\   %Img = OpTypeImage %F32 2D 0 0 0 2 Rgba8
        \\   %img = OpLoad %Img %img_ptr
        \\          OpImageWrite %img %uv %texel
        :: [_] "img_ptr" (render_target_ptr),
           [_] "uv" (uv),
           [_] "texel" (color),
    );

    asm volatile(
        \\OpEntryPoint GLCompute %entry "main" %render_target %gid
        \\OpExecutionMode %entry LocalSize 8 8 1
        :: [_] "entry" (shader),
           [_] "render_target" (render_target_ptr),
           [_] "gid" (gid_ptr),
    );
}
