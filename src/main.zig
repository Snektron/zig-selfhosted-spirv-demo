const std = @import("std");
const rsrc = @import("resources");
const vk = @import("vk.zig");
const print = std.debug.print;

// Make sure this is divisible by local size
const image_width = 128;
const image_height = 128;

const bindings = [_]vk.DescriptorSetLayoutBinding{
    .{ // layout(binding = 0, rgba8) uniform image2D render_target
        .binding = 0,
        .descriptor_type = .storage_image,
        .descriptor_count = 1,
        .stage_flags = .{.compute_bit = true},
        .p_immutable_samplers = null,
    },
};

pub fn main() void {
    main2() catch |err| {
        print("error: {}\n", .{err});
    };
}

const BaseDispatch = vk.BaseWrapper(.{
    .createInstance = true,
});

const InstanceDispatch = vk.InstanceWrapper(.{
    .destroyInstance = true,
    .enumeratePhysicalDevices = true,
    .getPhysicalDeviceProperties = true,
    .getPhysicalDeviceQueueFamilyProperties = true,
    .createDevice = true,
    .getDeviceProcAddr = true,
    .getPhysicalDeviceMemoryProperties = true,
});

const DeviceDispatch = vk.DeviceWrapper(.{
    .destroyDevice = true,
    .getDeviceQueue = true,
    .createPipelineLayout = true,
    .destroyPipelineLayout = true,
    .createDescriptorSetLayout = true,
    .destroyDescriptorSetLayout = true,
    .createShaderModule = true,
    .destroyShaderModule = true,
    .createComputePipelines = true,
    .destroyPipeline = true,
    .createDescriptorPool = true,
    .destroyDescriptorPool = true,
    .allocateDescriptorSets = true,
    .freeDescriptorSets = true,
    .createCommandPool = true,
    .destroyCommandPool = true,
    .allocateCommandBuffers = true,
    .createImage = true,
    .destroyImage = true,
    .getImageMemoryRequirements = true,
    .allocateMemory = true,
    .freeMemory = true,
    .bindImageMemory = true,
    .createImageView = true,
    .destroyImageView = true,
    .beginCommandBuffer = true,
    .endCommandBuffer = true,
    .cmdBindPipeline = true,
    .cmdBindDescriptorSets = true,
    .cmdDispatch = true,
    .cmdPipelineBarrier = true,
    .mapMemory = true,
    .unmapMemory = true,
    .queueSubmit = true,
    .deviceWaitIdle = true,
    .updateDescriptorSets = true,
});

fn resolve(size: usize) void {
    _ = size;
}

fn main2() !void {
    const allocator = std.heap.page_allocator;

    resolve(@sizeOf(vk.PhysicalDeviceProperties));
    resolve(@sizeOf(vk.PhysicalDeviceMemoryProperties));
    resolve(@sizeOf(vk.QueueFamilyProperties));
    resolve(@sizeOf(vk.ComputePipelineCreateInfo));
    resolve(@sizeOf(vk.BufferImageCopy));

    print("loading libvulkan\n", .{});
    var lib = try std.DynLib.open("libvulkan.so");
    defer lib.close();

    print("creating base dispatch\n", .{});
    const get_instance_proc_address = lib.lookup(vk.PfnGetInstanceProcAddr, "vkGetInstanceProcAddr") orelse return error.InvalidVulkanLibrary;
    const vkb = try BaseDispatch.load(get_instance_proc_address);

    print("creating instance\n", .{});
    const instance = try vkb.createInstance(&.{
        .flags = .{},
        .p_application_info = &.{
            .p_application_name = "Zig self-hosted spirv demo",
            .application_version = vk.makeApiVersion(0, 0, 0, 0),
            .p_engine_name = "Super awesome Zig engine",
            .engine_version = vk.makeApiVersion(0, 0, 0, 0),
            .api_version = vk.API_VERSION_1_2,
        },
        .enabled_layer_count = 0,
        .pp_enabled_layer_names = undefined,
        .enabled_extension_count = 0,
        .pp_enabled_extension_names = undefined,
    }, null);

    const vki = try InstanceDispatch.load(instance, get_instance_proc_address);
    defer vki.destroyInstance(instance, null);

    const pdev = blk: {
        var device_count: u32 = 1;
        var pdev: vk.PhysicalDevice = undefined;
        _ = try vki.enumeratePhysicalDevices(instance, &device_count, @ptrCast([*]vk.PhysicalDevice, &pdev));
        break :blk pdev;
    };

    const props = vki.getPhysicalDeviceProperties(pdev);
    print("using physical device '{s}'\n", .{std.mem.sliceTo(&props.device_name, 0)});

    const queue_family = blk: {
        var family_count: u32 = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(pdev, &family_count, null);

        const families = try allocator.alloc(vk.QueueFamilyProperties, family_count);
        defer allocator.free(families);
        vki.getPhysicalDeviceQueueFamilyProperties(pdev, &family_count, families.ptr);

        for (families) |fam, i| {
            if (fam.queue_flags.compute_bit) {
               break :blk @intCast(u32, i);
            }
        }
        unreachable;
    };

    print("using queue family {}\n", .{queue_family});
    print("creating device\n", .{});

    const dev = blk: {
        const qci = vk.DeviceQueueCreateInfo{
            .flags = .{},
            .queue_family_index = queue_family,
            .queue_count = 1,
            .p_queue_priorities = &[_]f32{1},
        };

        break :blk try vki.createDevice(pdev, &.{
            .flags = .{},
            .queue_create_info_count = 1,
            .p_queue_create_infos = @ptrCast([*]const vk.DeviceQueueCreateInfo, &qci),
            .enabled_layer_count = 0,
            .pp_enabled_layer_names = undefined,
            .enabled_extension_count = 0,
            .pp_enabled_extension_names = undefined,
            .p_enabled_features = null,
        }, null);
    };

    const vkd = try DeviceDispatch.load(dev, vki.dispatch.vkGetDeviceProcAddr);
    defer vkd.destroyDevice(dev, null);

    const queue = vkd.getDeviceQueue(dev, queue_family, 0);

    const mem_props = vki.getPhysicalDeviceMemoryProperties(pdev);

    print("creating pipeline\n", .{});
    const descriptor_set_layout = try vkd.createDescriptorSetLayout(dev, &.{
        .flags = .{},
        .binding_count = @intCast(u32, bindings.len),
        .p_bindings = &bindings,
    }, null);
    defer vkd.destroyDescriptorSetLayout(dev, descriptor_set_layout, null);

    const pipeline_layout = try vkd.createPipelineLayout(dev, &.{
        .flags = .{},
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &descriptor_set_layout),
        .push_constant_range_count = 0,
        .p_push_constant_ranges = undefined,
    }, null);
    defer vkd.destroyPipelineLayout(dev, pipeline_layout, null);

    const pipeline = blk: {
        const shader_module = try vkd.createShaderModule(dev, &.{
            .flags = .{},
            .code_size = rsrc.shader.len,
            .p_code = @ptrCast([*]const u32, &rsrc.shader),
        }, null);
        defer vkd.destroyShaderModule(dev, shader_module, null);

        const cpci = vk.ComputePipelineCreateInfo{
            .flags = .{},
            .stage = .{
                .flags = .{},
                .stage = .{.compute_bit = true},
                .module = shader_module,
                .p_name = "main",
                .p_specialization_info = null,
            },
            .layout = pipeline_layout,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = 0,
        };
        var pipeline: vk.Pipeline = undefined;
        _ = try vkd.createComputePipelines(
            dev,
            .null_handle,
            1,
            @ptrCast([*]const vk.ComputePipelineCreateInfo, &cpci),
            null,
            @ptrCast([*]vk.Pipeline, &pipeline),
        );
        break :blk pipeline;
    };
    defer vkd.destroyPipeline(dev, pipeline, null);

    print("creating descriptor set\n", .{});
    const descriptor_pool = blk: {
        const pools = [_]vk.DescriptorPoolSize{
            .{
                .type = .storage_image,
                .descriptor_count = 1,
            },
        };
        break :blk try vkd.createDescriptorPool(dev, &.{
            .flags = .{},
            .max_sets = 1,
            .pool_size_count = pools.len,
            .p_pool_sizes = &pools,
        }, null);
    };
    defer vkd.destroyDescriptorPool(dev, descriptor_pool, null);

    const descriptor_set = blk: {
        var set: vk.DescriptorSet = undefined;
        try vkd.allocateDescriptorSets(dev, &.{
            .descriptor_pool = descriptor_pool,
            .descriptor_set_count = 1,
            .p_set_layouts = @ptrCast([*]const vk.DescriptorSetLayout, &descriptor_set_layout),
        }, @ptrCast([*]vk.DescriptorSet, &set));
        break :blk set;
    };

    print("creating command buffer\n", .{});
    const cmd_pool = try vkd.createCommandPool(dev, &.{
        .flags = .{},
        .queue_family_index = queue_family,
    }, null);
    defer vkd.destroyCommandPool(dev, cmd_pool, null);

    const cmd_buf = blk: {
        var buf: vk.CommandBuffer = undefined;
        try vkd.allocateCommandBuffers(dev, &.{
            .command_pool = cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast([*]vk.CommandBuffer, &buf));
        break :blk buf;
    };

    print("creating render target\n", .{});
    const img = try vkd.createImage(dev, &.{
        .flags = .{},
        .image_type = .@"2d",
        .format = .r8g8b8a8_unorm,
        .extent = .{.width = image_width, .height = image_height, .depth = 1},
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{.@"1_bit" = true},
        .tiling = .linear,
        .usage = .{.storage_bit = true, .transfer_src_bit = true},
        .sharing_mode = .exclusive,
        .queue_family_index_count = 0,
        .p_queue_family_indices = undefined,
        .initial_layout = .@"undefined",
    }, null);
    defer vkd.destroyImage(dev, img, null);

    const mem = blk: {
        const mem_reqs = vkd.getImageMemoryRequirements(dev, img);
        const flags = vk.MemoryPropertyFlags{.device_local_bit = true, .host_visible_bit = true};
        const heap = for (mem_props.memory_types[0..mem_props.memory_type_count]) |mem_type, i| {
            if (mem_reqs.memory_type_bits & (@as(u32, 1) << @intCast(u5, i)) != 0 and mem_type.property_flags.contains(flags)) {
                break @intCast(u32, i);
            }
        } else return error.NoSuitableMemoryType;
        break :blk try vkd.allocateMemory(dev, &.{
            .allocation_size = mem_reqs.size,
            .memory_type_index = heap,
        }, null);
    };
    defer vkd.freeMemory(dev, mem, null);

    try vkd.bindImageMemory(dev, img, mem, 0);

    const subresource_range = vk.ImageSubresourceRange{
        .aspect_mask = .{.color_bit = true},
        .base_mip_level = 0,
        .level_count = 1,
        .base_array_layer = 0,
        .layer_count = 1,
    };

    const view = try vkd.createImageView(dev, &.{
        .flags = .{},
        .image = img,
        .view_type = .@"2d",
        .format = .r8g8b8a8_unorm,
        .components = .{.r = .identity, .g = .identity, .b = .identity, .a = .identity},
        .subresource_range = subresource_range,
    }, null);
    defer vkd.destroyImageView(dev, view, null);

    const render_target_write = vk.DescriptorImageInfo{
        .sampler = .null_handle,
        .image_view = view,
        .image_layout = .general,
    };
    const writes = [_]vk.WriteDescriptorSet{
        .{
            .dst_set = descriptor_set,
            .dst_binding = bindings[0].binding,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = bindings[0].descriptor_type,
            .p_image_info = @ptrCast([*]const vk.DescriptorImageInfo, &render_target_write),
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    };
    vkd.updateDescriptorSets(dev, @intCast(u32, writes.len), &writes, 0, undefined);

    print("rendering\n", .{});

    try vkd.beginCommandBuffer(cmd_buf, &.{
        .flags = .{.one_time_submit_bit = true},
        .p_inheritance_info = null,
    });

    {
        const barriers = [_]vk.ImageMemoryBarrier{
            .{
                .src_access_mask = .{},
                .dst_access_mask = .{},
                .old_layout = .@"undefined",
                .new_layout = .general,
                .src_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .dst_queue_family_index = vk.QUEUE_FAMILY_IGNORED,
                .image = img,
                .subresource_range = subresource_range,
            },
        };
        vkd.cmdPipelineBarrier(
            cmd_buf,
            .{.top_of_pipe_bit = true},
            .{.compute_shader_bit = true},
            .{},
            0, undefined,
            0, undefined,
            barriers.len,
            &barriers,
        );
    }

    vkd.cmdBindPipeline(cmd_buf, .compute, pipeline);
    vkd.cmdBindDescriptorSets(
        cmd_buf,
        .compute,
        pipeline_layout,
        0,
        1,
        @ptrCast([*]const vk.DescriptorSet, &descriptor_set),
        0,
        undefined,
    );

    vkd.cmdDispatch(
        cmd_buf,
        image_width / 8,
        image_height / 8,
        1,
    );

    try vkd.endCommandBuffer(cmd_buf);

    const submit = vk.SubmitInfo{
        .wait_semaphore_count = 0,
        .p_wait_semaphores = undefined,
        .p_wait_dst_stage_mask = undefined,
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast([*]const vk.CommandBuffer, &cmd_buf),
        .signal_semaphore_count = 0,
        .p_signal_semaphores = undefined,
    };
    try vkd.queueSubmit(queue, 1, @ptrCast([*]const vk.SubmitInfo, &submit), .null_handle);
    try vkd.deviceWaitIdle(dev);

    print("done\n", .{});

    const ptr = try vkd.mapMemory(dev, mem, 0, vk.WHOLE_SIZE, .{});
    const pixels = @ptrCast([*]const u32, @alignCast(@alignOf(u32), ptr.?));

    const file = try std.fs.cwd().createFile("output.ppm", .{});
    defer file.close();

    const writer = file.writer();
    try writer.writeAll("P6\n");
    try writer.print("{} {}\n", .{image_width, image_height});
    try writer.print("255\n", .{});

    for (pixels[0..image_width * image_height]) |pixel| {
        try writer.writeByte(@truncate(u8, pixel));
        try writer.writeByte(@truncate(u8, pixel >> 8));
        try writer.writeByte(@truncate(u8, pixel >> 16));
    }

    defer vkd.unmapMemory(dev, mem);
}
