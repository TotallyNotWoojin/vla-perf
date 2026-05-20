# SPDX-FileCopyrightText: Copyright (c) 2024 Multifidelity Roofline Analysis
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND MIT. Portions are Apache-2.0 while others are MIT.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by NVIDIA CORPORATION & AFFILIATES.

import copy
from ..default_models import ModelConfig, get_all_model_configs

##### SigLIP2 Models #####
# Note: We assume the projection from patches to hidden dimensions to be negligible.
# GPT: for SigLIP2-scale models the patch-embed + NaFlex cost is tiny compared to the 
# transformer itself – on the order of <1% of total FLOPs.

########################################
# SigLIP2 ViT-B (base, 86M vision)
# google/siglip2-base-patch16-224
########################################

siglip2_vitb_vision_config = ModelConfig(
    model="vla/siglip2-base-patch16-224-vision",
    # vision tower is ViT-B/16
    vocab_size=0,  # unused for vision
    # NaFlex = “Neural Aspect Ratio Flexible” positional embedding.
    max_model_len=256,  # 16x16 patches at 224px -> 14x14=196, SigLIP2 uses 256 “naflex” patches
    hidden_size=768,
    intermediate_size=3072,
    num_ffi=1,
    num_encoder_layers=12,
    num_decoder_layers=0,
    num_attention_heads=12,
    num_key_value_heads=12,  # MHA
    hidden_act="gelu_pytorch_tanh",
)

########################################
# SigLIP2 ViT-L (large, ~303M vision)
# google/siglip2-large-patch16-384
########################################

siglip2_vitl_vision_config = ModelConfig(
    model="vla/siglip2-large-patch16-384-vision",
    # ViT-L/16
    vocab_size=0,
    max_model_len=576,  # 384/16 = 24 -> 24x24 = 576 patches
    hidden_size=1024,
    intermediate_size=4096,
    num_ffi=1,
    num_encoder_layers=24,
    num_decoder_layers=0,
    num_attention_heads=16,
    num_key_value_heads=16,
    hidden_act="gelu_pytorch_tanh",
)

########################################
# SigLIP2 So400m (shape-optimized, “400M” vision)
# google/siglip2-so400m-patch14-384
########################################

siglip2_so400m_vision_config = ModelConfig(
    model="vla/siglip2-so400m-patch14-384-vision",
    # SoViT-400m-style backbone (shape-optimized)
    vocab_size=0,
    max_model_len=256,  # SigLIP2 uses 256 patches with naflex; seq len ~256
    hidden_size=1152,
    intermediate_size=4304,
    num_ffi=1,
    num_encoder_layers=27,
    num_decoder_layers=0,
    num_attention_heads=16,
    num_key_value_heads=16,
    hidden_act="gelu_pytorch_tanh",
)

########################################
# SigLIP2 g (giant, “1B” vision)
# google/siglip2-giant-opt-patch16-384
########################################

siglip2_g_vision_config = ModelConfig(
    model="vla/siglip2-giant-opt-patch16-384-vision",
    # ViT-g-style backbone
    vocab_size=0,
    max_model_len=576,  # 384/16 = 24 -> 24x24 = 576
    hidden_size=1536,
    intermediate_size=6144,
    num_ffi=1,
    num_encoder_layers=40,
    num_decoder_layers=0,
    num_attention_heads=16,  # HF config uses 16 heads
    num_key_value_heads=16,
    hidden_act="gelu_pytorch_tanh",
)


########################################
# DINOv2 ViT-L/14 (for OpenVLA Prismatic)
# facebook/dinov2-large
########################################

dinov2_vitl_config = ModelConfig(
    model="vla/dinov2-large-patch14-vision",
    # DINOv2 ViT-L/14 backbone
    vocab_size=0,
    max_model_len=256,  # 224/14 = 16 -> 16x16 = 256 patches (+ CLS token typically ignored)
    hidden_size=1024,
    intermediate_size=4096,
    num_ffi=1,
    num_encoder_layers=24,
    num_decoder_layers=0,
    num_attention_heads=16,
    num_key_value_heads=16,
    hidden_act="gelu",
)


########################################
# OpenVLA (Prismatic-7B based VLA)
# openvla/openvla-7b
#
# Architecture:
#   - Vision: DINOv2 ViT-L/14 (256 tokens) + SigLIP SoViT-400m/14 (256 tokens)
#   - Projector: 2-layer MLP (2176 -> 4096)
#   - LLM: Llama 2 7B (32 layers, 4096 hidden)
#   - Action: 7 DoF discretized to 256 bins
#
# Note: For performance modeling, we separately model:
#   1. Vision encoders (can run in parallel)
#   2. LLM prefill (for projected visual tokens)
#   3. LLM decode (for action token generation)
########################################

openvla_7b_llm_config = ModelConfig(
    model="vla/openvla-7b-llm",
    # Llama 2 7B backbone for OpenVLA
	# if we set large vocab size, the latency of the final prediction layer is also calculated
    vocab_size=256, # number of discrete tokens 
    max_model_len=2048,  # Reduced context for VLA use case
    hidden_size=4096,
    intermediate_size=11008,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=32,
    num_attention_heads=32,
    num_key_value_heads=32,  # MHA in Llama 2
    hidden_act="silu",
)


########################################
# π0 (pi-zero) from Physical Intelligence
# π0.5 has exact the same architecture as π0
# https://www.physicalintelligence.company/blog/pi0
#
# Architecture:
#   - Vision: SigLIP SoViT-400m/14 (reuse siglip2_so400m_vision_config)
#   - VLM backbone: Gemma 2B (18 layers, 2048 hidden)
#   - Action Expert: ~300M DiT for flow matching (~12 layers, 1024 hidden)
#   - Action: Continuous actions via flow matching (multiple denoising steps)
#
# Note: For performance modeling:
#   1. Vision encoder (SigLIP) - prefill
#   2. VLM prefill (Gemma 2B processes visual + text tokens)
#   3. Action Expert DiT (runs N denoising iterations)
########################################

pi0_vision_config = copy.deepcopy(siglip2_so400m_vision_config)
pi0_vision_config.model = "vla/pi0-vision"

# pi0 VLM backbone (Gemma 2B based, from PaliGemma)
# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py
pi0_vlm_config = ModelConfig(
    model="vla/pi0-vlm",
    # Gemma 2B backbone for pi0
	# if we set large 256K vocab size, the latency of the final prediction layer is also calculated
    vocab_size=0, # action chunk size 
    max_model_len=1024**3,  # Context for VLA use case
    hidden_size=2048,
    intermediate_size=16384,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=18,
	# Gemma-2B actually uses 8 heads, 1 KV head, head_dim=256; the “18 heads” in the paper 
	# is almost certainly a typo and does not match the released PaliGemma weights.
    num_attention_heads=8,
    num_key_value_heads=1,  # MQA in Gemma 2B
    head_dim=256,
    hidden_act="gelu",
)

# pi0 Action Expert (DiT for flow matching, ~300M params)
# Estimated architecture: ~12 layers, 1024 hidden, 4096 intermediate
# https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/pi0_config.py
pi0_action_expert_config = ModelConfig(
    model="vla/pi0-action-expert",
    vocab_size=0,  # Not used for continuous actions
    max_model_len=1024**3, # Action sequence length (action chunks)
    hidden_size=1024,
    intermediate_size=4096,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=18, 
    num_attention_heads=8,
    num_key_value_heads=1, 
    head_dim=256,
    hidden_act="gelu",
)


########################################
# π0.5 (pi-zero-point-five) from Physical Intelligence
# https://arxiv.org/abs/2504.16054
# https://www.physicalintelligence.company/blog/pi05
#
# Architecture:
#   - Vision: SigLIP SoViT-400m/14 (same as pi0)
#   - VLM backbone: Gemma 2B (same as pi0: 18 layers, 2048 hidden)
#   - Action Expert: gemma_300m (same transformer dims as pi0: 18 layers, 1024 hidden)
#     BUT with adaRMSNorm for flow matching timestep injection
#
# Key differences from π0 (for performance modeling):
#   1. Discrete state input: robot state is tokenized as ~32 language tokens
#      in the VLM prefix (instead of continuous embedding in action suffix).
#      This increases VLM prefill length by ~32 tokens.
#   2. Longer max_token_len: 200 vs 48 (for subtask prediction and state tokens)
#   3. adaRMSNorm in action expert: each transformer block has an additional
#      Dense(1024, 3072) per norm layer (2 per block) for timestep-conditioned
#      modulation (scale, shift, gate). This is a per-batch operation (not per-token),
#      adding ~0.5% FLOPs overhead — negligible for latency modeling.
#   4. Two-stage inference: VLM first generates subtask text tokens autoregressively,
#      then action expert runs flow matching. Subtask generation adds VLM decode steps.
#
# Source: github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/gemma.py
#   gemma_2b: width=2048, depth=18, mlp_dim=16384, num_heads=8, num_kv_heads=1, head_dim=256
#   gemma_300m: width=1024, depth=18, mlp_dim=4096, num_heads=8, num_kv_heads=1, head_dim=256
########################################


########################################
# π0.6 (pi-zero-point-six) from Physical Intelligence
# https://www.pi.website/research/pi0-6
#
# Architecture (scaled up):
#   - Vision: SigLIP SoViT-400m/14 (same as pi0)
#   - VLM backbone: Gemma 3 4B (34 layers, 2560 hidden)
#   - Action Expert: ~860M DiT (larger than pi0)
#   - Inference: ~63ms on H100
########################################

pi0_6_vision_config = copy.deepcopy(siglip2_so400m_vision_config)
pi0_6_vision_config.model = "vla/pi0.6-vision"

# pi0.6 VLM backbone (Gemma 3 4B based)
pi0_6_vlm_config = ModelConfig(
    model="vla/pi0.6-vlm",
    # Gemma 3 4B backbone for pi0.6
	# if we set large 256K vocab size, the latency of the final prediction layer is also calculated
    vocab_size=0, 
    max_model_len=1024**3,
    hidden_size=2560,
    intermediate_size=10240,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=34,
    num_attention_heads=16,
    num_key_value_heads=4,  # GQA
	head_dim=160,
    hidden_act="gelu_pytorch_tanh",
    sliding_window=1024,
)

# Note: Estimated! We dont' know the exact parameters of the action expert
# Actual size = 890M versus 860M reported
pi0_6_action_expert_config = ModelConfig(
    model="vla/pi0.6-action-expert",
    vocab_size=0,              
    max_model_len=1024**3,         
    hidden_size=1280,          
    intermediate_size=5120,    
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=34,     
    num_attention_heads=16,    
    num_key_value_heads=4,     
    head_dim=160,              
    hidden_act="gelu",
)

########################################
# SmolVLA (~450M total)
# https://arxiv.org/abs/2506.01844
#
# Architecture:
#   - Vision: SigLIP-SO400M/14 (400M, same as pi0)
#   - VLM backbone: SmolLM2-1.7B (24 layers, 2048 hidden)
#   - Action Expert: Flow Matching Transformer (~100M, cross-attends to VLM)
#
# Inference pipeline:
#   1. Vision encoding (SigLIP prefill)
#   2. VLM prefill (SmolLM2 processes visual + text tokens)
#   3. Action Expert flow matching (N denoising steps via parallel decode)
########################################

smolvla_vision_config = copy.deepcopy(siglip2_so400m_vision_config)
smolvla_vision_config.model = "vla/smolvla-vision"

smollm2_1p7b_config = ModelConfig(
    model="vla/smollm2-1.7b",
    vocab_size=49152,
    max_model_len=8192,
    hidden_size=2048,
    intermediate_size=8192,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=24,
    num_attention_heads=32,
    num_key_value_heads=32,
    head_dim=64,
    hidden_act="silu",
)

# ~100M flow matching transformer; cross-attends to VLM hidden states
smolvla_action_expert_config = ModelConfig(
    model="vla/smolvla-action-expert",
    vocab_size=0,
    max_model_len=1024**3,
    hidden_size=768,
    intermediate_size=3072,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=10,
    num_attention_heads=12,
    num_key_value_heads=12,
    head_dim=64,
    hidden_act="gelu",
)


########################################
# Qwen2-VL-7B Vision Encoder
# (shared backbone for Qwen2-VL-based VLAs such as CogACT)
# https://arxiv.org/abs/2409.12191
#
# Qwen2-VL uses a native-resolution ViT with 2×2 spatial token merging.
# At 224px input with patch_size=14: 16×16=256 raw patches → 64 merged tokens.
########################################

qwen2_vl_7b_vision_config = ModelConfig(
    model="vla/qwen2-vl-7b-vision",
    vocab_size=0,
    max_model_len=256,    # 256 raw patches (224px / 14px patch); merged to 64 by projector
    hidden_size=1152,
    intermediate_size=4608,
    num_ffi=1,
    num_encoder_layers=32,
    num_decoder_layers=0,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=72,           # 1152 / 16 = 72
    hidden_act="gelu",
)

# Qwen2-VL-7B language model backbone
qwen2_vl_7b_llm_config = ModelConfig(
    model="vla/qwen2-vl-7b-llm",
    vocab_size=152064,
    max_model_len=32768,
    hidden_size=3584,
    intermediate_size=18944,
    num_ffi=2,              # SwiGLU gate + up projections
    num_encoder_layers=0,
    num_decoder_layers=28,
    num_attention_heads=28,
    num_key_value_heads=4,  # GQA
    head_dim=128,           # 3584 / 28 = 128
    hidden_act="silu",
)


########################################
# X-VLA (~0.9B total)
# https://arxiv.org/abs/2510.10274
#
# Architecture:
#   - VLM: Florence-2-Large (DaViT vision encoder + BART-style text encoder, 0.77B)
#     Modeled as a single encoder-only prefill producing fused vision+language tokens.
#   - Policy: SoftPromptedTransformer (24 layers, 1024 hidden)
#     Processes Florence-2 outputs with learnable embodiment-specific soft prompts.
#   - Action: Flow Matching denoiser (folded into policy transformer; ~30 steps)
#
# Inference pipeline:
#   1. Florence-2 encoding (vision + text prefill)
#   2. Policy transformer prefill (with soft prompts + Florence-2 tokens)
#   3. Flow matching parallel decode (N denoising steps)
########################################

# Florence-2-Large approximated as a single encoder-only transformer
# (DaViT hierarchical ViT compressed to equivalent flat-transformer FLOPs)
florence2_large_encoder_config = ModelConfig(
    model="vla/florence2-large-encoder",
    vocab_size=51289,       # Florence-2 vocabulary
    max_model_len=576,      # fused vision + text tokens at 768px input
    hidden_size=1024,
    intermediate_size=4096,
    num_ffi=1,
    num_encoder_layers=24,
    num_decoder_layers=0,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=64,
    hidden_act="gelu",
)

# X-VLA SoftPromptedTransformer policy backbone
xvla_policy_config = ModelConfig(
    model="vla/xvla-policy",
    vocab_size=0,
    max_model_len=1024**3,
    hidden_size=1024,
    intermediate_size=4096,
    num_ffi=2,
    num_encoder_layers=0,
    num_decoder_layers=24,
    num_attention_heads=16,
    num_key_value_heads=16,
    head_dim=64,
    hidden_act="silu",
)


vla_models = get_all_model_configs(__name__)

vla_models.update(
    {
        # SigLIP2 vision encoders
        "siglip2-base-patch16-224-vision": siglip2_vitb_vision_config,
        "siglip2-large-patch16-384-vision": siglip2_vitl_vision_config,
        "siglip2-so400m-patch14-384-vision": siglip2_so400m_vision_config,
        "siglip2-giant-opt-patch16-384-vision": siglip2_g_vision_config,
        # DINOv2 vision encoder (for OpenVLA)
        "dinov2-large-patch14-vision": dinov2_vitl_config,
        # OpenVLA components
        "openvla-7b-llm": openvla_7b_llm_config,
        # pi0 components
		"pi0-vision": pi0_vision_config,
        "pi0-vlm": pi0_vlm_config,
        "pi0-action-expert": pi0_action_expert_config,
        # pi0.6 components (scaled up version)
		"pi0.6-vision": pi0_6_vision_config,
        "pi0.6-vlm": pi0_6_vlm_config,
        "pi0.6-action-expert": pi0_6_action_expert_config,
        # SmolVLA components
        "smolvla-vision": smolvla_vision_config,
        "smollm2-1.7b": smollm2_1p7b_config,
        "smolvla-action-expert": smolvla_action_expert_config,
        # Qwen2-VL-7B components (backbone for Qwen2-VL-based VLAs)
        "qwen2-vl-7b-vision": qwen2_vl_7b_vision_config,
        "qwen2-vl-7b-llm": qwen2_vl_7b_llm_config,
        # X-VLA components
        "florence2-large-encoder": florence2_large_encoder_config,
        "xvla-policy": xvla_policy_config,
    }
)
