# Feasibility Analysis: wgpu Platform Backend for rust-gpu-bindless

## TL;DR

Implementing a wgpu platform is **technically possible but with severe limitations** that undermine the purpose of using wgpu. The core "bindless" architecture requires features that wgpu only supports on Vulkan (BUFFER_BINDING_ARRAY), making cross-platform support — the main reason to use wgpu — largely unachievable. A wgpu backend would work only when wgpu runs on Vulkan, offering little advantage over the existing ash backend.

## Trait-by-Trait Feasibility

### 1. BindlessPlatform — PARTIALLY BLOCKED

| Method | wgpu Support | Notes |
|--------|-------------|-------|
| `create_platform` | ✅ | wgpu Instance/Adapter/Device creation |
| `alloc_buffer` | ✅ | wgpu::Buffer with usage flags |
| `alloc_image` | ✅ | wgpu::Texture with usage flags |
| `alloc_sampler` | ✅ | wgpu::Sampler |
| `update_after_bind_descriptor_limits` | ❌ BLOCKED | wgpu bind groups are immutable; no update-after-bind |
| `create_descriptor_set` | ⚠️ WORKAROUND | Must use binding arrays via BUFFER_BINDING_ARRAY + TEXTURE_BINDING_ARRAY (Vulkan-only for buffers) |
| `update_descriptor_set` | ❌ BLOCKED | Bind groups cannot be updated; must recreate entirely |
| `mapped_buffer_to_slab` | ⚠️ WORKAROUND | wgpu has BufferView/BufferViewMut, not presser::Slab |
| `destroy_buffers/images/samplers` | ✅ | Drop semantics |

### 2. PendingExecution — BLOCKED

| Requirement | wgpu Support | Notes |
|------------|-------------|-------|
| Timeline semaphores | ❌ | wgpu doesn't expose them |
| Future<Output=()> for GPU completion | ❌ | wgpu has SubmissionIndex + device.poll() but no Future trait |
| Clone + cheap status query | ❌ | No equivalent mechanism |

### 3. BindlessPipelinePlatform — MOSTLY OK

| Method | wgpu Support | Notes |
|--------|-------------|-------|
| `create_compute_pipeline` | ✅ | SPIR-V via spirv feature or PASSTHROUGH_SHADERS |
| `create_graphics_pipeline` | ✅ | Standard wgpu render pipeline |
| `create_mesh_graphics_pipeline` | ✅ | EXPERIMENTAL_MESH_SHADER (Vulkan/DX12/Metal) |
| `record_and_execute` | ✅ | CommandEncoder + queue.submit() |
| Push constants for `ParamConstant` | ✅ | wgpu IMMEDIATES feature (all platforms) |

### 4. RecordingContext — MOSTLY OK

| Method | wgpu Support | Notes |
|--------|-------------|-------|
| `copy_buffer_to_buffer` | ✅ | CommandEncoder::copy_buffer_to_buffer |
| `copy_buffer_to_image` | ✅ | CommandEncoder::copy_buffer_to_texture |
| `dispatch` | ✅ | ComputePass::dispatch_workgroups |
| `dispatch_indirect` | ✅ | ComputePass::dispatch_workgroups_indirect |

### 5. RecordingResourceContext — BLOCKED

| Method | wgpu Support | Notes |
|--------|-------------|-------|
| `transition_buffer` | ❌ | wgpu manages barriers automatically |
| `transition_image` | ❌ | wgpu manages barriers automatically |
| `add_dependency` | ❌ | No timeline semaphore dependency chain |
| `to_pending_execution` | ❌ | No equivalent |

### 6. RenderingContext — MOSTLY OK

| Method | wgpu Support | Notes |
|--------|-------------|-------|
| `begin_rendering` | ✅ | wgpu::RenderPass (not dynamic rendering, but equivalent) |
| `draw` | ✅ | render_pass.draw() |
| `draw_indexed` | ✅ | render_pass.draw_indexed() |
| `draw_indirect` | ✅ | render_pass.draw_indirect() |
| `draw_mesh` | ✅ | EXPERIMENTAL_MESH_SHADER |
| `set_viewport/scissor` | ✅ | render_pass.set_viewport/set_scissor_rect |

## Hard Blockers (Cannot Implement)

1. **Update-after-bind descriptor sets** — The entire descriptor management model (DescriptorTable with deferred updates) assumes descriptors can be written while the GPU is using them. wgpu bind groups are immutable after creation. Workaround: recreate bind groups every frame, but this breaks the A/B double-buffering model and adds significant CPU overhead.

2. **BUFFER_BINDING_ARRAY on non-Vulkan** — The core bindless buffer access pattern uses runtime-sized arrays of storage buffers indexed by DescriptorId. wgpu's `BUFFER_BINDING_ARRAY` is Vulkan-only. On Metal/DX12, buffer arrays are not available, meaning the fundamental bindless buffer access cannot work cross-platform.

3. **Timeline semaphore synchronization** — The PendingExecution trait requires Future-based GPU completion tracking with timeline semaphores. wgpu provides only poll-based synchronization via SubmissionIndex, with no async/Future support or fine-grained dependency chains.

4. **Explicit barrier control** — RecordingResourceContext::transition_buffer/image requires manual barrier insertion. wgpu manages all barriers internally and doesn't expose this API. The existing transition tracking would need to become no-ops, losing correctness guarantees.

## Possible But Requires Redesign

- **Descriptor set → Bind group mapping**: Would need to recreate bind groups on every update instead of mutating in-place
- **PendingExecution**: Could implement using Arc<AtomicBool> + device.poll() polling loop instead of timeline semaphores
- **Buffer mapping**: Use wgpu's BufferView instead of presser::Slab
- **Barriers**: Make transition methods no-ops since wgpu handles this internally
- **Shader loading**: Use wgpu's PASSTHROUGH_SHADERS or spirv feature for SPIR-V consumption

## Recommendation

A wgpu backend is **not recommended** for this project because:
1. The architecture is fundamentally built around Vulkan descriptor indexing concepts
2. The only platform where all required wgpu features work is Vulkan — where ash already works better
3. The workarounds for blocked features would significantly degrade performance and correctness
4. A proper wgpu port would require redesigning the core descriptor/binding model, not just adding a new platform impl

---

# wgpu-hal Feasibility Analysis

## TL;DR

wgpu-hal **resolves all 4 blockers** that made wgpu infeasible. It exposes low-level APIs that closely mirror what the ash backend needs: explicit barriers, Fence with FenceValue (timeline semaphore equivalent), raw SPIR-V shader loading, and `set_immediates` (push constants). The main design change needed is descriptor management: bind groups must be recreated per-frame instead of updated in-place (no UPDATE_AFTER_BIND flag in wgpu-hal), but this is workable with the existing A/B frame model.

**Verdict: Feasible (Vulkan-first, with path to DX12/Metal later)**

## Blocker Resolution

### Blocker 1: Update-after-bind descriptors → WORKAROUND (bind group per-frame)
- `BindGroupLayoutFlags` only has `PARTIALLY_BOUND` — **no UPDATE_AFTER_BIND flag**
- But bind groups can be freely created while old ones are in-flight (fence-protected)
- Strategy: create a new bind group each frame with all current descriptors
- The existing A/B double-buffer model maps well: frame A uses bind-group-A while bind-group-B is being built
- `BindGroupLayoutDescriptor` uses `wgt::BindGroupLayoutEntry` which has `count: Option<NonZeroU32>` for binding arrays
- Cost: wgpu-hal's Vulkan backend internally calls `vkUpdateDescriptorSets` in `create_bind_group`, comparable to current ash impl

### Blocker 2: Buffer binding arrays → RESOLVED (on Vulkan backend)
- `BindGroupLayoutEntry::count` supports binding arrays on all binding types
- `PARTIALLY_BOUND` flag allows shorter arrays than declared in layout
- On Vulkan backend, this maps directly to runtime descriptor arrays
- GLES backend unlikely to support this — Vulkan-first strategy required

### Blocker 3: Timeline semaphore sync → RESOLVED
- `Device::create_fence()` → creates a fence
- `Queue::submit(cmd_bufs, surface_textures, signal_fence: (&mut Fence, FenceValue))` → signals fence at value
- `Device::get_fence_value(&fence) → FenceValue` → polls current value (non-blocking)
- `Device::wait(&fence, value, timeout)` → blocks until fence reaches value
- `FenceValue = u64` — exactly the timeline semaphore model
- `PendingExecution` impl: store Arc<(Fence, FenceValue)>, Future::poll calls get_fence_value

### Blocker 4: Explicit barriers → RESOLVED
- `CommandEncoder::transition_buffers(barriers: Iterator<BufferBarrier>)`
- `CommandEncoder::transition_textures(barriers: Iterator<TextureBarrier>)`
- Direct mapping to the existing `RecordingResourceContext::transition_buffer/image`

## Feature Mapping (wgpu-hal ↔ rust-gpu-bindless)

| rust-gpu-bindless Feature | wgpu-hal Equivalent | Status |
|--------------------------|---------------------|--------|
| VkDescriptorSetLayout (4 bindings with runtime arrays) | `Device::create_bind_group_layout(BindGroupLayoutDescriptor)` with `count` in entries | ✅ |
| VkDescriptorSet + VkUpdateDescriptorSets | `Device::create_bind_group(BindGroupDescriptor)` (immutable, create per-frame) | ⚠️ Design change |
| VkPipelineLayout + push constants | `Device::create_pipeline_layout(PipelineLayoutDescriptor)` with `immediate_size` | ✅ |
| vkCmdPushConstants | `CommandEncoder::set_immediates(layout, offset_bytes, data)` | ✅ |
| VkShaderModule from SPIR-V | `Device::create_shader_module(ShaderInput::SpirV(&[u32]))` | ✅ |
| vkCreateComputePipelines | `Device::create_compute_pipeline(ComputePipelineDescriptor)` | ✅ |
| vkCreateGraphicsPipelines | `Device::create_render_pipeline(RenderPipelineDescriptor)` | ✅ |
| vkCmdBeginRendering (dynamic rendering) | `CommandEncoder::begin_render_pass(RenderPassDescriptor)` | ✅ (render pass model) |
| vkCmdDraw/DrawIndexed/DrawIndirect | `CommandEncoder::draw/draw_indexed/draw_indirect` | ✅ |
| vkCmdDrawMeshTasksEXT | `CommandEncoder::draw_mesh_tasks` | ✅ |
| vkCmdDispatch/DispatchIndirect | `CommandEncoder::dispatch/dispatch_indirect` | ✅ |
| vkCmdPipelineBarrier2 | `transition_buffers + transition_textures` | ✅ |
| VkSemaphore (TIMELINE) | `Fence + FenceValue` | ✅ |
| vkGetSemaphoreCounterValue | `Device::get_fence_value` | ✅ |
| vkWaitSemaphores | `Device::wait` | ✅ |
| vkCmdCopyBuffer | `CommandEncoder::copy_buffer_to_buffer` | ✅ |
| vkCmdCopyBufferToImage | `CommandEncoder::copy_buffer_to_texture` | ✅ |
| vkMapMemory (persistent) | `Device::map_buffer` (persistent mapping) | ✅ |
| vkCmdBindDescriptorSets | `CommandEncoder::set_bind_group` | ✅ |
| VkAccelerationStructure | `Device::create_acceleration_structure` | ✅ |
| vkCmdSetViewport/Scissor | `CommandEncoder::set_viewport/set_scissor_rect` | ✅ |

## Implementation Plan

### Phase 1: Core Platform Scaffolding
1. Create `crates/core/src/platform/wgpu_hal/` module directory
2. Add `mod.rs` with conditional compilation (`#[cfg(feature = "wgpu-hal")]`)
3. Define wrapper types for wgpu-hal Api type (Vulkan-focused initially)
4. Implement `PendingExecution` using `Fence + FenceValue + Arc`

### Phase 2: Descriptor Management (Key Design Change)
5. Implement bind group layout creation with 4 entries (buffer, storage_image, sampled_image, sampler) each with large `count`
6. Implement per-frame bind group recreation strategy:
   - Keep a `Vec` of current descriptor bindings (textures, buffers, samplers)
   - On each frame commit, create a new `BindGroup` with all current bindings
   - Track bind group lifetime with fences; destroy after fence completion
7. Implement `BindlessPlatform` trait (buffer/image/sampler alloc, descriptor updates)

### Phase 3: Pipeline & Command Recording
8. Implement `BindlessPipelinePlatform` — shader module from SPIR-V, compute/render/mesh pipelines
9. Implement `RecordingContext` — barrier insertion, buffer/image copies, dispatch
10. Implement `RecordingResourceContext` — transition_buffers/textures as barrier commands
11. Implement `RenderingContext` — begin_render_pass, draw commands, set_immediates for push constants

### Phase 4: Integration & Testing
12. Adapt `crates/winit/` or add a new window integration for wgpu-hal surface management
13. Run existing integration tests against the new backend
14. Port `swapchain_triangle` example as validation

## Key Design Decisions

1. **Bind group recreation vs update-after-bind**: Accept per-frame bind group creation. The Vulkan backend's `create_bind_group` internally calls `vkAllocateDescriptorSets + vkUpdateDescriptorSets`, which is comparable cost to the current `vkUpdateDescriptorSets` in the ash impl.

2. **Render pass model**: wgpu-hal uses `begin_render_pass/end_render_pass` instead of Vulkan's dynamic rendering. The `RenderingContext` trait method `begin_rendering` will map to `begin_render_pass`.

3. **Backend selection**: Initially use `wgpu_hal::vulkan::Api`. The wgpu-hal `Api` trait is generic, allowing future backends (DX12, Metal) if shader cross-compilation is added.

4. **Shader format**: Use `ShaderInput::SpirV(&[u32])` directly — no Naga conversion needed. This limits the initial backend to Vulkan, but is the simplest path.

## Risks & Limitations

- **Vulkan-only initially**: SpirV shaders only work with the Vulkan backend. Cross-platform requires adding Naga/HLSL/MSL shader compilation.
- **Bind group creation overhead**: Creating a full bind group per frame with thousands of entries may have performance implications. Needs benchmarking.
- **wgpu-hal is `unsafe`**: The entire API is unsafe, offering no safety advantages over ash. The benefit is purely abstraction/portability.
- **No `begin_rendering` (dynamic rendering)**: wgpu-hal uses traditional render passes, not VK_KHR_dynamic_rendering. Minor compatibility concern.
- **GLES backend unlikely to work**: GLES doesn't support binding arrays or many required features.

