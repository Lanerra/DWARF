"""
compute_coupling.py — Cross-component coupling metric for DWARF models.

Measures how much each component's gradient depends on the other being active.
Coupling metric:
  coupling_DSQG  = ||grad_DSQG_joint - grad_DSQG_frozen_full|| / ||grad_DSQG_joint||
  coupling_full  = ||grad_full_joint  - grad_full_frozen_DSQG|| / ||grad_full_joint||
  coupling_mean  = (coupling_DSQG + coupling_full) / 2

Run on existing checkpoints (3090, no new training):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/compute_coupling.py \
      --model condU_13M --device cuda

Run all epochs for a model:
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 tools/compute_coupling.py \
      --model condU_13M --all_epochs --device cuda

Available models: condU_13M, condU_35M, condM_I2G0, condM_85M
"""

import sys, os, json, argparse, importlib.util
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Known model configs  (D, H, FFN, L, full_layer, interference_interval, script)
# ---------------------------------------------------------------------------
MODEL_CONFIGS = {
    'condU_13M': {
        'script':     'train/train_2048_condU.py',
        'class':      'CondMTransformer',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6,
        'full_layer': 5, 'interference': 3, 'vocab': 32000,
        'checkpoint_dir': 'checkpoints/condU',
        'compiled':   False,
    },
    'condU_35M': {
        'script':     'train/train_2048_27m_condU.py',
        'class':      'CondUTransformer',
        'D':          512, 'H': 8, 'FFN': 2048, 'L': 6,
        'full_layer': 5, 'interference': 3, 'vocab': 32000,
        'checkpoint_dir': 'checkpoints/2048_condU_35m_checkpoints',
        'compiled':   True,   # saved with torch.compile (_orig_mod. prefix)
    },
    'condM_I2G0': {
        'script':     'train/train_2048_condM_I2G0.py',
        'class':      'CondMTransformer',
        'D':          256, 'H': 8, 'FFN': 1024, 'L': 6,
        'full_layer': 5, 'interference': 2, 'vocab': 32000,
        'checkpoint_dir': 'checkpoints/condM_I2G0',
        'compiled':   False,
    },
    'condM_85M': {
        'script':     'train/train_2048_85m_condM.py',
        'class':      'CondMTransformer',
        'D':          640, 'H': 8, 'FFN': 2560, 'L': 12,
        'full_layer': 11, 'interference': 3, 'vocab': 32000,
        'checkpoint_dir': 'checkpoints/2048_condM_85m_checkpoints',
        'compiled':   False,
    },
}

# Theoretical coupling coefficient for Rust calibration
def theoretical_coupling(cfg, injection_type='kv'):
    """
    Theoretical coupling bound:
      K/V injection:  n_heads * D * L  (nonlinear softmax pathway)
      Residual:       D                 (linear residual addition)
    """
    if injection_type == 'kv':
        return cfg['H'] * cfg['D'] * cfg['L']
    else:
        return cfg['D']


def load_model(model_name, checkpoint_path, device):
    cfg = MODEL_CONFIGS[model_name]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root, cfg['script'])

    spec = importlib.util.spec_from_file_location('train_script', script_path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    ModelClass = getattr(m, cfg['class'])
    model = ModelClass(
        vocab_size    = cfg['vocab'],
        embedding_dim = cfg['D'],
        num_layers    = cfg['L'],
        num_heads     = cfg['H'],
        ffn_dim       = cfg['FFN'],
        seq_len       = 2048,
        full_attn_layer      = cfg['full_layer'],
        interference_interval= cfg['interference'],
    )

    ck = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state = ck.get('model_state_dict', ck)

    # Strip torch.compile prefix if present
    if cfg['compiled']:
        state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}

    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, ck.get('val_ppl', None), ck.get('epoch', None)


def get_param_groups(model, full_layer_idx):
    """Split parameters into DSQG group and full-attention group."""
    dsqg_params, full_params = [], []
    for name, param in model.named_parameters():
        parts = name.split('.')
        if parts[0] == 'blocks' and int(parts[1]) == full_layer_idx:
            full_params.append((name, param))
        else:
            dsqg_params.append((name, param))
    return dsqg_params, full_params


def compute_coupling_metric(model, batch_ids, full_layer_idx, device):
    """
    Three forward+backward passes:
      1. Joint: both components active
      2. Frozen full-attn: measure DSQG gradient change
      3. Frozen DSQG: measure full-attn gradient change

    Returns dict with coupling metrics.
    """
    model.train()
    dsqg_params, full_params = get_param_groups(model, full_layer_idx)
    x = batch_ids[:, :-1].to(device)
    y = batch_ids[:, 1:].to(device)

    def fwd_bwd():
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        return loss.item()

    # --- Pass 1: Joint gradients ---
    loss_joint = fwd_bwd()
    grad_dsqg_joint = torch.cat([p.grad.detach().clone().flatten()
                                  for _, p in dsqg_params if p.grad is not None])
    grad_full_joint = torch.cat([p.grad.detach().clone().flatten()
                                  for _, p in full_params if p.grad is not None])

    # --- Pass 2: Freeze full attention, get DSQG gradient ---
    for _, p in full_params:
        p.requires_grad_(False)
    loss_frozen_full = fwd_bwd()
    grad_dsqg_frozen = torch.cat([p.grad.detach().clone().flatten()
                                   for _, p in dsqg_params if p.grad is not None])
    for _, p in full_params:
        p.requires_grad_(True)

    # --- Pass 3: Freeze DSQG, get full-attn gradient ---
    for _, p in dsqg_params:
        p.requires_grad_(False)
    loss_frozen_dsqg = fwd_bwd()
    grad_full_frozen = torch.cat([p.grad.detach().clone().flatten()
                                   for _, p in full_params if p.grad is not None])
    for _, p in dsqg_params:
        p.requires_grad_(True)

    model.zero_grad()

    # --- Coupling metrics ---
    eps = 1e-8
    coupling_dsqg = ((grad_dsqg_joint - grad_dsqg_frozen).norm() /
                     (grad_dsqg_joint.norm() + eps)).item()
    coupling_full = ((grad_full_joint - grad_full_frozen).norm() /
                     (grad_full_joint.norm() + eps)).item()
    coupling_mean = (coupling_dsqg + coupling_full) / 2

    return {
        'coupling_dsqg':      coupling_dsqg,
        'coupling_full':      coupling_full,
        'coupling_mean':      coupling_mean,
        'loss_joint':         loss_joint,
        'loss_frozen_full':   loss_frozen_full,
        'loss_frozen_dsqg':   loss_frozen_dsqg,
        'grad_norm_dsqg':     grad_dsqg_joint.norm().item(),
        'grad_norm_full':     grad_full_joint.norm().item(),
    }


def load_data_batch(batch_size=8, seq_len=2049, seed=42):
    """Load a fixed batch from the cached FineWeb-Edu dataset."""
    cache = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         'logs/fineweb_encoded_2048.pt')
    if os.path.exists(cache):
        print(f'Loading data from cache: {cache}')
        data = torch.load(cache, weights_only=True)
        # data is typically a dict with 'train', 'val', 'test'
        if isinstance(data, dict):
            val = data.get('val', data.get('train'))
        else:
            val = data
        torch.manual_seed(seed)
        idx = torch.randperm(len(val))[:batch_size]
        batch = torch.stack([val[i][:seq_len] for i in idx])
        return batch
    else:
        print('Warning: no cached data found, using random token IDs')
        torch.manual_seed(seed)
        return torch.randint(0, 32000, (batch_size, seq_len))


def run_single(model_name, checkpoint_path, device, batch):
    cfg = MODEL_CONFIGS[model_name]
    print(f'  Loading {model_name} from {checkpoint_path} ...')
    model, val_ppl, epoch = load_model(model_name, checkpoint_path, device)
    n_params = sum(p.numel() for p in model.parameters())
    injection = 'kv' if 'condU' in model_name else 'residual'
    theo = theoretical_coupling(cfg, injection)

    print(f'  Computing coupling metric ...')
    metrics = compute_coupling_metric(model, batch, cfg['full_layer'], device)

    result = {
        'model':              model_name,
        'checkpoint':         checkpoint_path,
        'epoch':              epoch,
        'val_ppl':            val_ppl,
        'n_params':           n_params,
        'injection_type':     injection,
        'theoretical_coupling': theo,
        **metrics,
    }
    ppl_str = f'{val_ppl:.3f}' if val_ppl is not None else 'N/A'
    print(f'  epoch={epoch} val_ppl={ppl_str}  '
          f'coupling_mean={metrics["coupling_mean"]:.4f}  '
          f'(DSQG={metrics["coupling_dsqg"]:.4f}, full={metrics["coupling_full"]:.4f})')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--checkpoint', default=None, help='Single checkpoint path')
    parser.add_argument('--all_epochs', action='store_true',
                        help='Run all epoch_*.pt checkpoints in checkpoint_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--out', default=None, help='Output JSON path')
    args = parser.parse_args()

    device = torch.device(args.device)
    batch = load_data_batch(batch_size=args.batch_size)

    cfg = MODEL_CONFIGS[args.model]
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    results = []

    if args.all_epochs:
        ckpt_dir = os.path.join(root, cfg['checkpoint_dir'])
        checkpoints = sorted(
            [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
             if f.startswith('epoch_') and f.endswith('.pt')]
        )
        if not checkpoints:
            print(f'No epoch_*.pt found in {ckpt_dir}')
            sys.exit(1)
        print(f'Found {len(checkpoints)} epoch checkpoints for {args.model}')
        for ckpt in checkpoints:
            r = run_single(args.model, ckpt, device, batch)
            results.append(r)
    else:
        ckpt = args.checkpoint or os.path.join(root, cfg['checkpoint_dir'], 'best.pt')
        results.append(run_single(args.model, ckpt, device, batch))

    out_path = args.out or os.path.join(
        root, 'logs', f'coupling_{args.model}.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out_path}')

    # Print summary table
    print(f'\n{"Epoch":>6} {"val_ppl":>9} {"coupling_mean":>14} {"DSQG":>8} {"full_attn":>10}')
    print('-' * 55)
    for r in results:
        ppl = r['val_ppl']
        ppl_s = f'{ppl:>9.3f}' if ppl is not None else f'{"N/A":>9}'
        print(f'{str(r.get("epoch","??")):>6} '
              f'{ppl_s} '
              f'{r["coupling_mean"]:>14.4f} '
              f'{r["coupling_dsqg"]:>8.4f} '
              f'{r["coupling_full"]:>10.4f}')


if __name__ == '__main__':
    main()
