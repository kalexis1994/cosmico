# Cosmico

A real-time cosmological simulation engine with an interactive node-based pipeline editor. Built in C++20 / CUDA on top of Vulkan and ImGui.

Cosmico ships seven compute backends — direct N-body, Barnes-Hut, Particle-Mesh (FFT), Inflation, 2D and 3D Causal Dynamical Triangulations, and a visual node-graph builder — each wrapped in a gallery launcher with deep-dive science papers, recording/playback, and a dockable 3D viewport.

## Features

- **7 compute backends**
  - **Brute-force N-body** — Vulkan compute shader, O(N²)
  - **Barnes-Hut** — CUDA octree, O(N log N)
  - **Particle-Mesh** — CUDA + cuFFT, O(N + M log M)
  - **Inflation** — CUDA scalar field on a 3D grid
  - **CDT 2D** — CPU Monte Carlo with Pachner moves on a 1+1D triangulation
  - **CDT 3D** — CUDA-accelerated causal dynamical triangulations
  - **Node Graph** — visual pipeline builder; decompose the PM solver into composable nodes
- **Vulkan ↔ CUDA interop** via exported memory handles (Win32) — particle buffers shared zero-copy
- **Dockable ImGui workspace** — viewport, debug panel, science papers, node editor, timeline
- **Recording & playback** — binary `.cosmsnap` format with O(1) random-access seek
- **Per-simulation science papers** — LaTeX rendered to Unicode in-app
- **Gallery launcher** — card-based browser with previews, descriptions, parameter overrides

## Building

### Requirements

- **Windows 10/11** (Vulkan-CUDA interop currently uses Win32 handles)
- **MSVC 2022** (Build Tools or Community)
- **CUDA Toolkit 12.x** — tested with 12.1
- **Vulkan SDK 1.3+** — auto-detected from `C:/VulkanSDK/*` or `$VULKAN_SDK`
- **CMake 3.25+**
- A GPU with **compute capability 7.5 / 8.6 / 8.9** (Turing, Ampere, or Ada)

All other dependencies (GLFW, Dear ImGui docking branch, imnodes, nlohmann/json, stb) are fetched automatically by CMake.

### Build

```powershell
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

The first configure pulls and compiles dependencies and may take several minutes. Subsequent builds are incremental.

CMake post-build steps copy `shaders/`, `simulations/`, `papers/`, `resources/`, and the required CUDA runtime DLLs alongside the executable, so `build/Release/cosmico.exe` is portable.

> **Note:** if you add new `.cpp` or `.cu` files, re-run `cmake -B build` — sources are collected via `GLOB_RECURSE`, which doesn't auto-detect new files in incremental builds.

### Run

```powershell
./build/Release/cosmico.exe
```

The Gallery opens. Pick a simulation card, click **Launch** to enter the dockable workspace, **Back to Gallery** (or `ESC`) to return.

## Project layout

```
cosmico/
├── src/
│   ├── core/         Application loop, window, input, timer
│   ├── vulkan/       Vulkan bootstrap, swapchain, pipelines, ImGui backend
│   ├── cuda/         Vulkan-CUDA interop helpers
│   ├── simulation/   7 compute backends + orchestrator
│   ├── nodes/        Node graph: SimNode, compiler, executor, registry, UI
│   ├── renderer/     Vulkan renderers (particles, volume, CMB, CDT, offscreen)
│   ├── recording/    Snapshot recorder, player, .cosmsnap I/O
│   └── ui/           Gallery, debug panel, science panel, timeline
├── cuda/kernels/     CUDA kernels (.cu/.cuh)
├── shaders/          GLSL sources (compiled to SPIR-V at build)
├── simulations/      Per-simulation configs, previews, papers
├── papers/           Legacy LaTeX sources
└── resources/        Planet textures and other static assets
```

## Adding a simulation to the gallery

1. Create `simulations/<your-sim>/config.json`:
   ```json
   {
     "title": "My Simulation",
     "description": "What it computes and why.",
     "backend": "PM",
     "params": { "particleCount": 131072, "dt": 0.005 },
     "camera": { "distance": 150.0, "pitch": 0.4, "yaw": 0.3 }
   }
   ```
2. Add `simulations/<your-sim>/preview.png` (card thumbnail).
3. *(Optional)* `simulations/<your-sim>/papers/index.tex` for the science panel.
4. *(Node Graph backend only)* `simulations/<your-sim>/graph.json`.

The catalog auto-discovers folders at startup.

## License

MIT — see [LICENSE](LICENSE).
