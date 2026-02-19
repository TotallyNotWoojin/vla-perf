# VLA-Perf: Performance Modeling for Vision-Language-Action Models

This directory contains performance modeling and analysis tools for **Vision-Language-Action (VLA)** models used in robotics. It evaluates inference latency, throughput, and hardware utilization across different hardware platforms (e.g., NVIDIA A100, H100, B100, Jetson AGX Thor) and network configurations.

## Directory Structure

```
vla-perf/
├── README.md                     # This file
├── LICENSE                       # MIT License
├── perf_utils.py                 # Shared utility functions
├── pi0_perf.py                   # Pi0 family performance evaluation (main script)
├── openvla_perf.py               # OpenVLA performance evaluation
├── network_latency.py            # Network latency estimation
├── perf_results/                 # Output CSV results
├── paper_figures/                # Generated plots (PDF + PNG)
├── paper_tables/                 # Generated LaTeX tables
├── plot_scripts/                 # Plotting and table generation scripts
│   ├── plot_util.py              # Shared plotting utilities
│   ├── print_0_configs_hw_models.py
│   ├── print_1_base_pi0.py
│   ├── print_2_scale_model.py
│   ├── print_3_long_context.py
│   ├── print_4_denoise_steps_action_lengths.py
│   ├── print_5_autoregressive_vs_diffusion.py
│   ├── print_6_device_vs_server.py
│   ├── print_7_async_inference.py
│   └── paper_tables/             # Additional LaTeX table templates
└── test_scripts/                 # Test and debugging scripts
    ├── test_bound.py
    ├── test_llm_perf.py
    ├── vision_encoder_perf.py
    ├── print_vla_param_counts.py
    └── perf_results/             # Test-specific output results
```

## Core Script

### `pi0_perf.py` — Pi0 Family Performance Evaluation

The main evaluation script used in the paper. Models the **Pi0** VLA architecture from Physical Intelligence, which uses:

- **Vision encoder**: SigLIP SoViT-400m (encoder-only, prefill)
- **VLM backbone**: Gemma 2B (prefill for visual + text tokens)
- **Action Expert**: Diffusion Transformer (DiT) running N denoising iterations via Flow Matching

Contains the following experiments (each generates CSV results in `perf_results/`):

| # | Experiment | Function | Description |
|---|---|---|---|
| 1 | Base E2E Performance | `get_all_pi0_perf()` | End-to-end latency breakdown (vision, VLM, action) across hardware |
| 2 | Model Size Scaling | `get_model_size_scaling_perf()` | How latency scales as each component (vision/VLM/action expert) grows |
| 3 | Long Context | `run_long_context_experiment()` | Effect of increasing observation history (number of camera frames over time) |
| 4 | Denoising Steps x Action Lengths | `compare_denoising_steps_action_lengths()` | Joint sweep over diffusion denoising steps and action chunk sizes |
| 5 | Autoregressive vs Diffusion | `compare_autoregressive_vs_diffusion()` | Compares diffusion-based vs autoregressive action generation |
| 6 | Device vs Server | `run_device_vs_server_comparison()` | Compares on-device (Jetson Thor) vs edge/cloud server inference with network latency |
| 7 | Device-Server Collaboration | `run_device_server_collaboration_comparison()` | Split inference: vision on-device + VLM/action on server (Helix-style) |

**How to run:**

```bash
python pi0_perf.py
```

By default, all experiments run (`runall = True`). To selectively run experiments, set `runall = False` and toggle individual flags near the bottom of the script:

```python
# Set runall to True to run all experiments, or False to selectively run
runall = True

# Individual experiment flags (only used if runall=False)
run_exp_1_base_pi0 = True
run_exp_2_model_size_scaling = True
run_exp_3_long_context = True
run_exp_4_denoise_steps_action_lengths = True
run_exp_5_autoregressive_vs_diffusion = True
run_exp_6_device_vs_server = True
run_exp_7_device_server_collaboration = True
```

Results are saved to `perf_results/` as CSV files, with logs written to `perf_results/pi0_perf.log`.

## Other Scripts

### `perf_utils.py` — Shared Utility Functions

Provides common helper functions used across all evaluation scripts:

- **`setup_logging()`** — Configures logging to both console and file.
- **`evaluate_boundness()`** — Determines whether a workload is compute-bound, memory-bound, or communication-bound, and computes weighted-average operational intensity.
- **`get_best_precision_for_system()`** — Resolves the best available precision (e.g., bf16, fp16) for a given hardware system.
- **`calculate_kv_cache_size_mb()`** — Calculates KV-cache memory footprint for a given model and sequence length.
- **`get_parallelism()`** — Enumerates valid (tensor parallel, pipeline parallel) configurations for a given device count.
- **`get_pareto_df()` / `get_optimal_df()`** — Filters performance results to keep only Pareto-optimal batch size configurations.
- **`collect_prefill_perf()` / `collect_decode_perf()` / `collect_parallel_decode_perf()`** — Core data collection functions that sweep over batch sizes and parallelism strategies, calling the GenZ performance modeling backend.
- **`calculate_transformer_params()` / `format_param_count()`** — Compute and format total parameter counts for transformer model configs.

### `openvla_perf.py` — OpenVLA Performance Evaluation

Models the **OpenVLA** (openvla-7b) architecture:

- **Vision**: Dual encoder (DINOv2 ViT-L/14 + SigLIP SoViT-400m/14), each producing 256 tokens
- **Projector**: 2-layer MLP fusing vision features into LLM space
- **LLM**: Llama 2 7B backbone (32 layers, 4096 hidden)
- **Action**: 7-DoF actions discretized into 256 bins, decoded autoregressively (7 tokens)

Evaluates each component independently (vision, LLM prefill, LLM decode) and computes end-to-end latency. Results are saved to `perf_results/openvla_*.csv`.

### `network_latency.py` — Network Latency Estimation

Estimates network transmission latency for robot-server communication under various network conditions. Does **not** require GenZ hardware modeling — it is a pure analytical calculation.

Supports the following network types:
- **Cellular**: 4G LTE, 5G NR (sub-6 GHz)
- **WiFi**: WiFi 5/6/6E/7 (802.11ac/ax/be)
- **Ethernet**: 1G / 10G / 25G / 100G / 400G
- **Cloud**: Fast (10 Gbps, 10 ms) and Slow (1 Gbps, 100 ms)

Evaluates five categories of data transfer:
1. **Image transmission** (Robot -> Server) — raw and JPEG-compressed at various resolutions
2. **Action transmission** (Server -> Robot) — various DoF and action chunk sizes
3. **KV-cache transmission** (Server -> Robot) — for distributed VLM inference
4. **Bidirectional: Image + Action** — full robot control loop round-trip
5. **Bidirectional: Image + KV-cache** — distributed inference round-trip

## Plot Scripts (`plot_scripts/`)

Each script reads CSV results from `perf_results/` and generates figures and/or LaTeX tables:

| Script | Input CSV(s) | Output |
|---|---|---|
| `plot_util.py` | — | Shared utility (hardware name formatting) |
| `print_0_configs_hw_models.py` | — | LaTeX tables for hardware and model config parameters |
| `print_1_base_pi0.py` | `pi0_family_e2e_perf.csv` | LaTeX tables for base Pi0 performance and workload characteristics |
| `print_2_scale_model.py` | `pi0_model_size_scaling.csv`, `pi0_model_*_params.csv` | Plots and tables for model size scaling |
| `print_3_long_context.py` | `pi0_long_context.csv` | Log-log plot of latency vs context length |
| `print_4_denoise_steps_action_lengths.py` | `pi0_denoising_steps_action_lengths.csv` | Heatmaps for denoising steps vs action chunk size |
| `print_5_autoregressive_vs_diffusion.py` | `pi0_autoregressive_vs_diffusion.csv` | Comparison plots: diffusion vs autoregressive action heads |
| `print_6_device_vs_server.py` | `pi0_device_vs_server.csv` | Latency comparison across device/server/network configs |
| `print_7_async_inference.py` | `pi0_device_vs_server.csv` | LaTeX table comparing sync vs async inference throughput |

## Test Scripts (`test_scripts/`)

Scripts for development, debugging, and validation:

| Script | Purpose |
|---|---|
| `test_bound.py` | Validates GenZ backend by inspecting layer-level performance output (compute/memory/communication times, op intensity, boundness) for a single model |
| `test_llm_perf.py` | Evaluates raw LLM performance (prefill, sequential decode, parallel decode) independently of VLA pipelines. Useful for benchmarking the LLM backbone in isolation |
| `vision_encoder_perf.py` | Benchmarks standalone SigLIP2 vision encoder variants across hardware platforms |
| `print_vla_param_counts.py` | Prints parameter counts for all VLA models defined in `vla_models.py` |
