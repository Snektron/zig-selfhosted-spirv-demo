ZIG = ~/programming/zig/build-dev/zig-out/bin/zig
CACHEDIR = zig-cache

demo: $(CACHEDIR)/main
	cp $(CACHEDIR)/main demo

$(CACHEDIR)/shader.spv: src/shader.zig
	@mkdir -p $(CACHEDIR)
	$(ZIG) build-obj --cache-dir $(CACHEDIR) -femit-bin=$@ -target spirv64-vulkan -ofmt=spirv $<
	spirv-val $@

$(CACHEDIR)/main: src/main.zig $(CACHEDIR)/resources.zig $(CACHEDIR)/shader.spv
	@mkdir -p $(CACHEDIR)
	$(ZIG) build-exe --cache-dir $(CACHEDIR) -femit-bin=$@ -fLLVM -fno-stage1 -lc $< --pkg-begin resources $(CACHEDIR)/resources.zig --pkg-end

$(CACHEDIR)/resources.zig:
	@mkdir -p $(CACHEDIR)
	@echo 'pub const shader align(@alignOf(u32)) = @embedFile("$(abspath $(CACHEDIR))/shader.spv").*;' > $@

clean:
	@rm -rf $(CACHEDIR) demo

.PHONY: clean
