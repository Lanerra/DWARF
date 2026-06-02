"""
model_registry.py — Single source of truth for DWARF model configurations.

All eval/viz tools import from here. Adding a new arch requires editing
only this file.
"""

import importlib.util
import os

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SE015_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]

SE096_OFFSETS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 15, 16, 21, 23, 28, 32, 48, 64, 69, 96, 107,
    128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608,
    640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 1024, 1056, 1088,
    1120, 1152, 1184, 1216, 1248, 1280, 1312, 1344, 1376, 1408, 1440, 1472, 1504,
    1536, 1568, 1600, 1632, 1664, 1696, 1728, 1760, 1792, 1824, 1856, 1888, 1920,
    1952, 1984, 2016, 2048, 2080, 2112, 2144, 2176, 2199,
]

J44_OFFSETS = list(range(33)) + [48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]

J20D_V10_OFFSETS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 16, 23, 32, 64, 128, 256, 512, 1024]

J13D_OFFSETS = [1, 2, 4, 5, 8, 16, 32, 64, 128, 256, 512, 768, 1024]

J12_OFFSETS = [1, 2, 4, 8, 16, 64, 96, 192, 384, 512, 768, 1024]

D41S3_OFFSETS = list(range(42)) + [48, 128, 384]

D41S5_OFFSETS = list(range(42)) + [48, 128, 384, 768, 1536]

D41_35M_OFFSETS = list(range(49)) + [96, 128, 384]

TOKENIZER_PATHS = {
    'fineweb_32k': os.path.join(REPO, 'results', 'fineweb_tokenizer_32k.json'),
    'cond_i': os.path.join(REPO, 'results', '2048_condI_tokenizer.json'),
}

# ─────────────────────────────────────────────────────────────────────────────
# Registry schema per entry:
#   train_script           str       relative to REPO
#   model_class            str       class name in train script
#   D, H, FFN, L           int|None  None → read from train script globals
#   full_attn_layer        int|None  None → read from train script or absent
#   offsets                list|'per_layer'   default SE015_OFFSETS
#   tokenizer              str       key into TOKENIZER_PATHS or raw path
#   scale_embed_init       float|None  None → don't pass to __init__
#   is_standard            bool      standard transformer (no DSQG)
#   no_num_layers          bool      skip num_layers kwarg (TriadicJ96)
#   no_full_attn           bool      no full attention layer (pure DSQG)
#   no_scale_embed         bool      arch lacks scale_embed param
#   needs_interference_interval  bool  read INTERFERENCE from module & pass
#   init_kwargs            dict      extra kwargs for cls.__init__
#   label                  str|None
#   checkpoint             str|None  relative to REPO
#   params_ref             int|None
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    # ── Moonshot lineage (AutoresearchTransformerPhysics, preIF, fineweb 32K) ──
    'moonshot_58m': {
        'train_script': 'train/train_moonshot_58m_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 512, 'H': 8, 'FFN': 2048, 'L': 8, 'full_attn_layer': 2,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
        'label': 'Moonshot-58M (J24 se015, D=512, FA@L2)',
        'checkpoint': 'autoresearch/checkpoints/moonshot_58m_best.pt',
    },
    'moonshot_58m_sft': {
        'train_script': 'train/train_moonshot_58m_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 512, 'H': 8, 'FFN': 2048, 'L': 8, 'full_attn_layer': 2,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },

    # ── d768 family ──
    'd768_l16': {
        'train_script': 'train/train_d768_l16_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 768, 'H': 12, 'FFN': 1536, 'L': 16, 'full_attn_layer': 4,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },
    'd768_l24': {
        'train_script': 'train/train_d768_l24_h100_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 768, 'H': 12, 'FFN': 1536, 'L': 24, 'full_attn_layer': 6,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },
    'd768_l32': {
        'train_script': 'train/train_d768_l32_h100_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 768, 'H': 12, 'FFN': 1536, 'L': 32, 'full_attn_layer': 8,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },

    # ── d1024 ──
    'd1024_267m': {
        'train_script': 'train/train_d1024_267m_h100_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 1024, 'H': 16, 'FFN': 2048, 'L': 24, 'full_attn_layer': 6,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },

    # ── FFN ablation ──
    'ffn_ablation': {
        'train_script': 'train/train_ffn_ablation_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },

    # ── Triadic family ──
    'triadic_l13_so2_mixed': {
        'train_script': 'train/train_d512_l13_triadic_aabbc_mixed_scratch_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'Triadic-J96 L13 D512 SO(2) Mixed-Domain (48M, PPL=33.64, passkey=98.3%)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_triadic_aabbc_mixed_scratch_best.pt',
        'params_ref': 48_359_073,
    },
    'triadic_l13': {
        'train_script': 'train/train_d512_l13_triadic_j96_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'Triadic-J96 L13 D512 (J=32/layer, 4 triads, FA@L3)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_triadic_j96_best.pt',
        'params_ref': 48_359_073,
    },
    'triadic_l25_d768': {
        'train_script': 'train/train_d768_l25_triadic_j96_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 768, 'H': 12, 'FFN': 1536, 'L': 25, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'Triadic-J96 L25 D768 (J=32/layer, 8 triads, FA@L6, phase-informed)',
        'checkpoint': 'autoresearch/checkpoints/d768_l25_triadic_j96_best.pt',
        'params_ref': 159_977_185,
    },
    'triadic_l25_mixed': {
        'train_script': 'train/train_d768_l25_triadic_mixed_scratch_3090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 768, 'H': 12, 'FFN': 1536, 'L': 25, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'Triadic-J96 L25 D768 Mixed-Domain from Scratch (160M, PPL=22.88, passkey=97.5%)',
        'checkpoint': 'autoresearch/checkpoints/d768_l25_triadic_mixed_scratch_best.pt',
        'params_ref': 159_977_185,
    },
    'd640_l20_movt_r4': {
        'train_script': 'train/train_d640_l20_movt_r4_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 640, 'H': 10, 'FFN': 1280, 'L': 20, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'D640 L20 Triadic J=96 MOVT R_PLANES=4 (140M, ep2 PPL=36.23, passkey=99.2%)',
        'checkpoint': 'autoresearch/checkpoints/d640_l20_movt_r4_ep2_full_attn.pt',
        'params_ref': 140_000_000,
    },
    'triadic_l25_d1024_mixed': {
        'train_script': 'train/train_d1024_l25_triadic_aabbc_h200_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 1024, 'H': 16, 'FFN': 2048, 'L': 25, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'Triadic-J96 L25 D1024 Mixed-Domain from Scratch (273M, PPL=20.92, passkey=96.7%)',
        'checkpoint': 'autoresearch/checkpoints/d1024_l25_triadic_aabbc_mixed_scratch_best.pt',
        'params_ref': 273_100_000,
    },
    'topk_l13_d512': {
        'train_script': 'train/train_d512_l13_topk_distill.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'is_topk': True,
        'topk_k': 64,
        'label': 'TopK-L13 D512 (k=64, distillation, 100% passkey)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_topk_distill_best.pt',
        'params_ref': 48_100_241,
    },
    'topk_l13_d512_nodistill': {
        'train_script': 'train/train_d512_l13_topk_nodistill.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'is_topk': True,
        'topk_k': 64,
        'label': 'TopK-L13 D512 (k=64, no distillation, ablation)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_topk_nodistill_best.pt',
    },
    'topk_l25_d768': {
        'train_script': 'train/train_d768_l25_topk_distill.py',
        'model_class': 'TriadicJ96',
        'D': 768, 'H': 12, 'FFN': 1536, 'L': 25, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'is_topk': True,
        'topk_k': 64,
        'label': 'TopK-L25 D768 (k=64, distillation)',
        'checkpoint': 'autoresearch/checkpoints/d768_l25_topk_distill_best.pt',
    },
    'topk_l25_d1024': {
        'train_script': 'train/train_d1024_l25_topk_distill.py',
        'model_class': 'TriadicJ96',
        'D': 1024, 'H': 16, 'FFN': 2048, 'L': 25, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'is_topk': True,
        'topk_k': 64,
        'label': 'TopK-L25 D1024 (k=64, distillation)',
        'checkpoint': 'autoresearch/checkpoints/d1024_l25_topk_distill_best.pt',
    },
    'dsr_nodistill_l13_d512': {
        'train_script': 'train/train_d512_l13_dsr_nodistill.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'is_dsr': True,
        'dsr_num_chunks': 32,
        'dsr_top_k': 4,
        'label': 'DSR-L13 D512 (C=32, top_k=4, no distillation, ep10 93.3% passkey)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_dsr_nodistill_best.pt',
        'params_ref': 48_100_241,
    },
    'd512_l10_hisa_h16_v2': {
        'train_script': 'train/train_d512_l10_hisa_h16_v2.py',
        'model_class': 'TriadicJ96Dsr',
        'D': 512, 'H': 8, 'FFN': 1536, 'L': 10, 'full_attn_layer': 6,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'no_full_attn': True,
        'tokenizer': 'tokenizers/mixed_tokenizer_32k.json',
        'scale_embed_init': 0.15,
        'init_kwargs': {'dsr_layer': 6, 'num_chunks': 32, 'top_k_chunks': 4},
        'label': 'D512-L10 HISA v2 (ep1 PPL=39.01, passkey=85.0%)',
        'checkpoint': 'autoresearch/checkpoints/d512_l10_hisa_h16_v2_best.pt',
        'params_ref': 46_115_334,
    },
    'd512_l10_hisa_h16_v2_l3': {
        'train_script': 'train/train_d512_l10_hisa_h16_v2_l3.py',
        'model_class': 'TriadicJ96Dsr',
        'D': 512, 'H': 8, 'FFN': 1536, 'L': 10, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'no_full_attn': True,
        'tokenizer': 'tokenizers/mixed_tokenizer_32k.json',
        'scale_embed_init': 0.15,
        'init_kwargs': {'dsr_layer': 3, 'num_chunks': 32, 'top_k_chunks': 4},
        'label': 'D512-L10 HISA v2 relocated to L3',
        'params_ref': 46_115_334,
    },
    'd512_l10_fa_h16_v2_lq': {
        'train_script': 'train/train_d512_l10_fa_h16_v2_lq.py',
        'model_class': 'TriadicJ96FA',
        'D': 512, 'H': 8, 'FFN': 1536, 'L': 10, 'full_attn_layer': 2,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'tokenizers/mixed_tokenizer_32k.json',
        'scale_embed_init': 0.15,
        'label': 'D512-L10 FA/DSQG v2, single Triton FA at L//4',
        'checkpoint': None,
        'params_ref': None,
    },
    'pure_dsqg_l13_so2': {
        'train_script': 'train/train_d512_l13_pure_dsqg_so2_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 13,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'no_full_attn': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
        'label': 'Pure DSQG L13 D512 SO(2) — NO FA (48M, PPL=61.38, passkey=13.3% FIRST EVER)',
        'checkpoint': 'autoresearch/checkpoints/d512_l13_pure_dsqg_so2_best.pt',
        'params_ref': 48_000_000,
    },
    'triadic_l9': {
        'train_script': 'train/train_d512_l9_triadic_j96_4090_bf16.py',
        'model_class': 'TriadicJ96',
        'D': 512, 'H': 8, 'FFN': 1024, 'L': 9, 'full_attn_layer': 3,
        'offsets': 'per_layer',
        'no_num_layers': True,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.15,
    },

    # ── Borg family (AutoresearchTransformerPhysics) ──
    'borg_gen5_L8_preIF': {
        'train_script': 'train/train_borg_gen5_L8_preIF_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 512, 'H': 8, 'FFN': 2048, 'L': 8, 'full_attn_layer': 2,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'scale_embed_init': 0.1,
        'needs_interference_interval': True,
    },
    'borg_gen5_L11_preIF': {
        'train_script': 'train/train_borg_gen5_L11_preIF_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'scale_embed_init': 0.1,
        'needs_interference_interval': True,
    },
    'borg_gen4_L11': {
        'train_script': 'train/train_borg_gen4_L11_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
    },
    'borg_gen3_L8': {
        'train_script': 'train/train_borg_gen3_L8_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
    },
    'borg_j12_30m': {
        'train_script': 'train/train_borg_j12_30m_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': J12_OFFSETS,
        'tokenizer': 'cond_i',
    },
    'borg_adapt_warmstart': {
        'train_script': 'train/train_borg_adapt_13m_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'borg_midattn': {
        'train_script': 'train/train_borg_midattn_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'borg_lastattn': {
        'train_script': 'train/train_borg_lastattn_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'borg_midattn_gen2': {
        'train_script': 'train/train_borg_midattn_unfreeze_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'borg_midfa_L0': {
        'train_script': 'train/train_borg_midfa_L0_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'borg2_dual_fa': {
        'train_script': 'train/train_borg2_dual_fa_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'borg_L11': {
        'train_script': 'train/train_borg_L11_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },

    # ── J-series autoresearch ──
    'j24d_int2_physics': {
        'train_script': 'train/train_j24d_int2_physics_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'j26d_int2_physics': {
        'train_script': 'train/train_j26d_int2_physics_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'j20d_v10_L8': {
        'train_script': 'train/train_j20d_v10_L8_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'D': 512, 'H': 8, 'FFN': 2048, 'L': 8, 'full_attn_layer': 7,
        'offsets': J20D_V10_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'j20d_v10_L10': {
        'train_script': 'train/train_j20d_v10_L10_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': J20D_V10_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'j20d_v10_L12': {
        'train_script': 'train/train_j20d_v10_L12_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': J20D_V10_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'j20d_v10_L32': {
        'train_script': 'train/train_j20d_v10_L32_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': J20D_V10_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },

    # ── J13D / Moonshot lineage (fineweb 32K) ──
    'j13d_v2_30m': {
        'train_script': 'train/train_borg_j13d_v2_30m_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': J13D_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },
    'j13d_v3_30m': {
        'train_script': 'train/train_borg_j13d_v3_l8_4090_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': J13D_OFFSETS,
        'tokenizer': 'fineweb_32k',
        'scale_embed_init': 0.1,
    },

    # ── Curve ──
    'curve_27m': {
        'train_script': 'train/train_curve_27m_bf16.py',
        'model_class': 'CurveTransformer',
        'D': 512, 'H': 8, 'FFN': 768, 'L': 6, 'full_attn_layer': 5,
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },

    # ── Old cond family ──
    'condu': {
        'train_script': 'train/train_2048_condU.py',
        'model_class': 'CondMTransformer',
        'offsets': J44_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'd41s3': {
        'train_script': 'train/train_2048_14m_d41s3.py',
        'model_class': 'CondMTransformer',
        'offsets': D41S3_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'd41s5': {
        'train_script': 'train/train_2048_14m_d41s5.py',
        'model_class': 'CondMTransformer',
        'offsets': D41S5_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'd41_35m': {
        'train_script': 'train/train_2048_35m_d41.py',
        'model_class': 'CondMTransformer',
        'offsets': D41_35M_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'condx_v2': {
        'train_script': 'train/train_2048_35m_condX_v2_bf16.py',
        'model_class': 'CondXTransformer',
        'offsets': J44_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'condm_85m': {
        'train_script': 'train/train_2048_85m_condM.py',
        'model_class': 'CondMTransformer',
        'offsets': J44_OFFSETS,
        'tokenizer': 'cond_i',
        'no_scale_embed': True,
        'needs_interference_interval': True,
    },
    'condu_v5': {
        'train_script': 'train/train_2048_condU_v5.py',
        'model_class': 'CondUV5Transformer',
        'offsets': J44_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'condv': {
        'train_script': 'train/train_2048_condV.py',
        'model_class': 'CondMTransformer',
        'offsets': J44_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
    'condw': {
        'train_script': 'train/train_2048_condW.py',
        'model_class': 'CondWTransformer',
        'offsets': J44_OFFSETS,
        'tokenizer': 'cond_i',
        'no_full_attn': True,
        'needs_interference_interval': True,
    },

    # ── Standard baselines ──
    'std_85m': {
        'train_script': 'train/train_2048_85m_standard_baseline.py',
        'model_class': 'StandardTransformer85M',
        'offsets': [],
        'tokenizer': 'cond_i',
        'is_standard': True,
    },
    'std_13m': {
        'train_script': 'train/train_2048_85m_standard_baseline.py',
        'model_class': 'StandardTransformer85M',
        'offsets': [],
        'tokenizer': 'cond_i',
        'is_standard': True,
    },

    # ── DWARF large models ──
    'dwarf_104m_d1024': {
        'train_script': 'train/train_dwarf_104m_d1024_h100_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'scale_embed_init': 0.1,
        'needs_interference_interval': True,
    },
    'dwarf_1b_d4096': {
        'train_script': 'train/train_dwarf_1b_d4096_bf16.py',
        'model_class': 'AutoresearchTransformerPhysics',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },

    # ── CondDelta ──
    'cond_delta': {
        'train_script': 'train/train_cond_delta_bf16.py',
        'model_class': 'AutoresearchTransformerCondDelta',
        'offsets': SE015_OFFSETS,
        'tokenizer': 'cond_i',
        'needs_interference_interval': True,
    },
}


def _import_train_script(relative_path):
    """Import a train script as a module."""
    script_path = os.path.join(REPO, relative_path)
    spec = importlib.util.spec_from_file_location(
        f'train_script_{os.path.basename(relative_path)}',
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    original_set_threads = getattr(torch, 'set_num_threads', None)
    original_set_interop = getattr(torch, 'set_num_interop_threads', None)

    def _safe_thread_setter(fn):
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except RuntimeError as exc:
                if 'cannot set number of interop threads' in str(exc) or 'cannot set number of threads' in str(exc):
                    return None
                raise
        return wrapper

    if original_set_threads is not None:
        torch.set_num_threads = _safe_thread_setter(original_set_threads)
    if original_set_interop is not None:
        torch.set_num_interop_threads = _safe_thread_setter(original_set_interop)
    try:
        spec.loader.exec_module(module)
    finally:
        if original_set_threads is not None:
            torch.set_num_threads = original_set_threads
        if original_set_interop is not None:
            torch.set_num_interop_threads = original_set_interop
    return module


def _dims_from_checkpoint(checkpoint_path):
    """Infer StandardTransformer dimensions from checkpoint state dict."""
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ck.get('model_state_dict', ck)
    number_of_layers = sum(1 for k in state_dict if k.startswith('blocks.') and k.endswith('.ln1.weight'))
    embedding_dim = state_dict['token_emb.weight'].shape[1]
    ffn_dim = state_dict['blocks.0.ffn.fc1.weight'].shape[0]
    return {
        'embedding_dim': embedding_dim,
        'num_layers': number_of_layers,
        'ffn_dim': ffn_dim,
        'num_heads': 8,
    }


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint with standard key stripping. Returns (missing, unexpected, raw_checkpoint)."""
    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck.get('model', ck))
    if any('_orig_mod' in k for k in state):
        state = {k.replace('._orig_mod', '').replace('_orig_mod.', ''): v
                 for k, v in state.items()}
    state = {k: v for k, v in state.items() if 'causal_mask' not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected, ck


def build_model(arch_name, device='cpu', checkpoint_path=None):
    """
    Instantiate and optionally load a model from the registry.

    Returns (model, cfg_copy, train_module).
    cfg_copy is a shallow copy of the registry entry with runtime fields:
      _val_ppl  — validation perplexity from checkpoint (if loaded)
    """
    assert arch_name in MODEL_REGISTRY, f'Unknown arch: {arch_name}'
    cfg = dict(MODEL_REGISTRY[arch_name])
    module = _import_train_script(cfg['train_script'])

    cls = getattr(module, cfg['model_class'])

    embedding_dim = cfg.get('D') or getattr(module, 'EMBEDDING_DIM', 512)
    number_of_layers = cfg.get('L') or getattr(module, 'NUM_LAYERS', 8)
    number_of_heads = cfg.get('H') or getattr(module, 'NUM_HEADS', 8)
    ffn_dim = cfg.get('FFN') or getattr(module, 'FFN_DIM', 2048)
    full_attn_layer = cfg.get('full_attn_layer')
    if full_attn_layer is None and not cfg.get('is_standard', False) and not cfg.get('no_full_attn', False):
        full_attn_layer = getattr(module, 'FULL_ATTN_LAYER', number_of_layers - 1)
    vocab_size = cfg.get('vocab_size') or getattr(module, 'VOCAB_SIZE', 32000)
    sequence_length = cfg.get('seq_len', 2048)

    kwargs = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'num_heads': number_of_heads,
        'ffn_dim': ffn_dim,
        'seq_len': sequence_length,
    }

    if not cfg.get('no_num_layers', False):
        kwargs['num_layers'] = number_of_layers

    if not cfg.get('is_standard', False) and not cfg.get('no_full_attn', False):
        kwargs['full_attn_layer'] = full_attn_layer

    scale_embed = cfg.get('scale_embed_init')
    if scale_embed is not None:
        kwargs['scale_embed_init_val'] = scale_embed

    if cfg.get('needs_interference_interval', False):
        kwargs['interference_interval'] = getattr(module, 'INTERFERENCE', 2)

    kwargs.update(cfg.get('init_kwargs', {}))

    ckpt = checkpoint_path or cfg.get('checkpoint')
    absolute_checkpoint = None
    if ckpt:
        absolute_checkpoint = os.path.join(REPO, ckpt) if not os.path.isabs(ckpt) else ckpt

    if cfg.get('is_standard', False) and absolute_checkpoint:
        dims = _dims_from_checkpoint(absolute_checkpoint)
        kwargs['embedding_dim'] = dims['embedding_dim']
        kwargs['num_layers'] = dims['num_layers']
        kwargs['ffn_dim'] = dims['ffn_dim']
        kwargs['num_heads'] = dims['num_heads']

    model = cls(**kwargs)

    if cfg.get('is_topk', False) and full_attn_layer is not None:
        from kernels.topk_sparse_attn import TopKSparseAttention
        head_dim = embedding_dim // number_of_heads
        topk_k = cfg.get('topk_k', 64)
        model.blocks[full_attn_layer].attn = TopKSparseAttention(
            D=embedding_dim, H=number_of_heads, hd=head_dim, k=topk_k,
        )
        print(f'  TopK: replaced blocks[{full_attn_layer}].attn → '
              f'TopKSparseAttention(k={topk_k})')

    if cfg.get('is_dsr', False) and full_attn_layer is not None:
        from kernels.hierarchical_sparse_attn import HierarchicalSparseAttention
        head_dim = embedding_dim // number_of_heads
        num_chunks = cfg.get('dsr_num_chunks', 32)
        top_k_chunks = cfg.get('dsr_top_k', 4)
        model.blocks[full_attn_layer].attn = HierarchicalSparseAttention(
            D=embedding_dim, H=number_of_heads, hd=head_dim,
            num_chunks=num_chunks, top_k_chunks=top_k_chunks,
        )
        print(f'  DSR: replaced blocks[{full_attn_layer}].attn → '
              f'HierarchicalSparseAttention(C={num_chunks}, top_k={top_k_chunks})')

    if absolute_checkpoint:
        _missing, _unexpected, ck = load_checkpoint(model, absolute_checkpoint)
        cfg['_val_ppl'] = ck.get('val_ppl', '?')
        print(f'  Loaded: {ckpt}  (val_ppl={cfg["_val_ppl"]})')

    model.to(device)
    model.eval()

    cfg['_resolved_D'] = embedding_dim
    cfg['_resolved_L'] = number_of_layers
    cfg['_resolved_H'] = number_of_heads
    cfg['_resolved_FFN'] = ffn_dim
    cfg['_resolved_full_attn_layer'] = full_attn_layer

    return model, cfg, module


def get_tokenizer(arch_name):
    """Return raw Tokenizer for the given arch."""
    cfg = MODEL_REGISTRY[arch_name]
    tok_key = cfg.get('tokenizer', 'fineweb_32k')
    path = TOKENIZER_PATHS.get(tok_key, tok_key)
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def get_offsets(arch_name):
    """Return the offset list for an arch; SE096_OFFSETS for 'per_layer' archs."""
    cfg = MODEL_REGISTRY[arch_name]
    offsets = cfg.get('offsets', SE015_OFFSETS)
    if offsets == 'per_layer':
        return SE096_OFFSETS
    return offsets
