"""
🔬 DWARF Autoresearch — J16D Depth-Aware Head Regime Specialization (14M)

Hypothesis (autoresearcher depth_aware_head_regime_specialization, score=0.9021):
  Different DSQG layer depths should have different head regime assignments.
  L0 is the "skeleton" layer (uniquely critical — skip = 0% passkey).
  L3 is the "refinement" layer (IF block present).

  Currently: all DSQG layers initialized with identical alphas linspace(0.2, 2.0).
  This means all heads start with the same local/distal bias profile.

  Test: initialize L0 pos_bias with SHALLOW alphas (0.05, 0.8) — more distal-receptive
  to bootstrap long-range relay signal at the earliest possible layer.
  All other DSQG layers: standard alphas (0.2, 2.0).

Architecture: D=256, H=8, FFN=1024, L=6 (14M), J=16, V=32K
Compared to: autoresearcher capped best (ppl=61.75, passkey=18.3%)

Run (from /tmp/dwarf-j17d):
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python3 autoresearch/train_j16d_depth_aware.py \
    > autoresearch/run_j16d_depth_aware.log 2>&1
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Base on train_ar.py but with the depth-aware pos_bias init
# We import everything from train_ar and override just the model init
EMBEDDING_DIM = 256
NUM_HEADS     = 8
FFN_DIM       = 1024
NUM_LAYERS    = 6
INTERFERENCE  = 3
FULL_ATTN_LAYER = 5
SCALE_EMBED_INIT_VAL = 0.1
SCALE_EMBED_LR_MULT  = 15.0
LR            = 3e-4
SCREEN_EPOCHS = 3

# KEY CHANGE: L0 gets shallow alphas (distal-biased), rest get standard
L0_ALPHA_MIN  = 0.05   # much shallower than standard 0.2 — L0 heads more distal-receptive
L0_ALPHA_MAX  = 0.8    # vs standard 2.0
STD_ALPHA_MIN = 0.2    # standard for all other DSQG layers
STD_ALPHA_MAX = 2.0

import json, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F

OFFSETS = [1, 3, 4, 13, 15, 21, 23, 28, 48, 64, 96, 192, 384, 512, 768, 1024]
VOCAB_SIZE     = 32000
BATCH_SIZE     = 8
GRAD_ACCUM     = 4
MAX_SEQ_LEN    = 2048
MAX_TRAIN_SEQS = 52_716
MAX_VAL_SEQS   = 5_582
PASSKEY_TRIALS = 20
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 1536]
FW_CACHE_FILE  = 'benchmarks/logs/condm_fineweb_edu_doc_cache.json'
TOKENIZER_CANDIDATES = ['benchmarks/logs/condm_tokenizer.json', 'results/2048_condI_tokenizer.json']
_FILLER_SENTENCE = 'the weather was mild and the air was still . '
_INTRO_TEMPLATE  = 'the secret word is {word} .'
_RETRIEVAL_CUE   = 'the secret word is'
_PASSKEY_WORDS   = ['apple', 'banana', 'orange', 'cherry', 'grape', 'lemon', 'mango', 'peach', 'plum', 'berry']

_kd = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'kernels'))
sys.path.insert(0, _kd)
from dsqg_attention_v6 import DSQGAttentionV6

assert len(OFFSETS) == 16
assert FULL_ATTN_LAYER == NUM_LAYERS - 1

class FFN(nn.Module):
    def __init__(self, d, ffn, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d, ffn); self.fc2 = nn.Linear(ffn, d); self.drop = nn.Dropout(drop)
    def forward(self, x): return self.fc2(self.drop(F.gelu(self.fc1(x))))

class DSQGBlockV6(nn.Module):
    def __init__(self, d, h, ffn, seq_len, drop=0.1, interference=False):
        super().__init__()
        self.interference = interference; self.num_heads = h; self.head_dim = d // h
        self.norm1 = nn.LayerNorm(d); self.norm2 = nn.LayerNorm(d)
        self.attn  = DSQGAttentionV6(d, h, seq_len=seq_len, dropout=drop)
        self.ffn   = FFN(d, ffn, drop)
        if interference:
            self.inter_norm = nn.LayerNorm(d); self.inter_gate = nn.Linear(d, d)
            self.inter_k_proj = nn.Linear(d, d); self.inter_v_proj = nn.Linear(d, d)
    def forward(self, x):
        kv = None
        if self.interference:
            xi = self.inter_norm(x); B, N, D = xi.shape; H, HD = self.num_heads, self.head_dim
            cum = xi.cumsum(1); cnt = torch.arange(1, N+1, device=xi.device, dtype=xi.dtype)[None, :, None]
            pool = cum / cnt; inter = torch.sigmoid(self.inter_gate(xi)) * pool
            kd = self.inter_k_proj(inter).view(B, N, H, HD).permute(0,2,1,3).contiguous()
            vd = self.inter_v_proj(inter).view(B, N, H, HD).permute(0,2,1,3).contiguous()
            kv = (kd, vd)
        x = x + self.attn(self.norm1(x), kv_inject=kv)
        return x + self.ffn(self.norm2(x))

class FullCausalAttn(nn.Module):
    def __init__(self, d, h, drop=0.1):
        super().__init__()
        self.num_heads = h; self.head_dim = d // h
        self.qkv = nn.Linear(d, 3*d); self.out = nn.Linear(d, d)
        self.gate = nn.Linear(d, d); nn.init.constant_(self.gate.bias, 0); self.drop = drop
    def forward(self, x):
        B, N, D = x.shape; H, HD = self.num_heads, self.head_dim
        q, k, v = self.qkv(x).split(D, -1)
        q = q.view(B,N,H,HD).permute(0,2,1,3); k = k.view(B,N,H,HD).permute(0,2,1,3); v = v.view(B,N,H,HD).permute(0,2,1,3)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0)
        o = o.permute(0,2,1,3).reshape(B,N,D)
        return F.dropout(self.out(o * torch.sigmoid(self.gate(x))), p=self.drop, training=self.training)

class FullAttnBlock(nn.Module):
    def __init__(self, d, h, ffn, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.attn = FullCausalAttn(d, h, drop); self.ffn = FFN(d, ffn, drop)
    def forward(self, x): return x + self.ffn(self.n2(x + self.attn(self.n1(x))))

class DepthAwareTransformer(nn.Module):
    def __init__(self, vocab_size, d, L, H, ffn, seq_len, full_layer, interf_interval,
                 se_init=0.1, drop=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Embedding(seq_len+2, d)
        self.drop = nn.Dropout(drop)
        blocks = []
        for i in range(L):
            if i == full_layer:
                blocks.append(FullAttnBlock(d, H, ffn, drop))
            else:
                has_if = interf_interval is not None and i % interf_interval == interf_interval - 1
                blocks.append(DSQGBlockV6(d, H, ffn, seq_len, drop, has_if))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, vocab_size, bias=False)
        self.out.weight = self.embedding.weight
        self._init(se_init, full_layer)

    def _init(self, se_init, full_layer):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.02); m.bias is not None and nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0, 0.02)
        for m in self.modules():
            if hasattr(m, 'gate') and isinstance(getattr(m,'gate',None), nn.Linear): nn.init.constant_(m.gate.bias, 0)
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6):
                nn.init.normal_(m.phase_base, 0, 0.01); nn.init.normal_(m.query_probes, 0, 0.01)
                nn.init.normal_(m.key_probes, 0, 0.01); nn.init.normal_(m.phase_gain, 0, 0.001)
                if se_init != 0: nn.init.constant_(m.scale_embed, se_init)
        # DEPTH-AWARE: re-init L0 pos_bias with shallow alphas
        ALL_OFFSETS = OFFSETS
        delta_vals = torch.tensor([math.log(1.0 + d) for d in ALL_OFFSETS], dtype=torch.float32)
        std_alphas = torch.linspace(STD_ALPHA_MIN, STD_ALPHA_MAX, NUM_HEADS)
        l0_alphas  = torch.linspace(L0_ALPHA_MIN,  L0_ALPHA_MAX,  NUM_HEADS)
        for i, block in enumerate(self.blocks):
            if i == full_layer: continue
            if hasattr(block, 'attn') and isinstance(block.attn, DSQGAttentionV6):
                alphas = l0_alphas if i == 0 else std_alphas
                with torch.no_grad():
                    block.attn.pos_bias.copy_(-delta_vals.unsqueeze(1) * alphas.unsqueeze(0))

    def forward(self, idx):
        B, N = idx.shape; pos = torch.arange(N, device=idx.device).unsqueeze(0)
        x = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for b in self.blocks: x = b(x)
        return self.out(self.norm(x))
    def param_count(self): return sum(p.numel() for p in self.parameters())
    def scale_embed_params(self):
        for m in self.modules():
            if isinstance(m, DSQGAttentionV6): yield m.scale_embed
    def other_params(self):
        se_ids = {id(p) for p in self.scale_embed_params()}
        for p in self.parameters():
            if id(p) not in se_ids: yield p

class BPEWrap:
    def __init__(self, tok): self.t = tok
    def encode(self, text): return self.t.encode(text).ids
    def decode(self, ids): return self.t.decode(ids)
    def vocab_size(self): return self.t.get_vocab_size()

def load_tok():
    from tokenizers import Tokenizer
    p = next((p for p in TOKENIZER_CANDIDATES if os.path.exists(p)), None)
    if not p: raise FileNotFoundError(f'Tokenizer not found: {TOKENIZER_CANDIDATES}')
    return BPEWrap(Tokenizer.from_file(p))

def load_data(tok):
    if os.path.exists(FW_CACHE_FILE):
        with open(FW_CACHE_FILE) as f: docs = json.load(f)
        texts = docs if isinstance(docs, list) else list(docs.values())[0] if isinstance(list(docs.values())[0], list) else docs.get('documents', [])
    else: raise FileNotFoundError(FW_CACHE_FILE)
    cache = 'logs/fineweb_encoded_2048.pt'
    if os.path.exists(cache):
        c = torch.load(cache, weights_only=True); return c['train'], c['val']
    toks = []
    for t in texts: toks.extend(tok.encode(t if isinstance(t,str) else t.get('text','')))
    arr = torch.tensor(toks, dtype=torch.long); n = len(arr)//MAX_SEQ_LEN
    seqs = arr[:n*MAX_SEQ_LEN].view(n, MAX_SEQ_LEN)
    n_val = int(n*0.05); return seqs[n_val:], seqs[:n_val]

def passkey_accuracy(model, tok, device):
    model.eval(); res = {}
    fill_ids = tok.encode(_FILLER_SENTENCE); cue_ids = tok.encode(_RETRIEVAL_CUE)
    for d in PASSKEY_DISTANCES:
        ok = 0
        for trial in range(PASSKEY_TRIALS):
            word = _PASSKEY_WORDS[trial % len(_PASSKEY_WORDS)]
            intro = tok.encode(_INTRO_TEMPLATE.format(word=word))
            cands = [(tok.encode(' '+w) or tok.encode(w))[0] for w in _PASSKEY_WORDS]
            nb = max(0, d - len(intro)); na = max(0, MAX_SEQ_LEN - d - len(cue_ids) - 10)
            seq = (fill_ids*(nb//len(fill_ids)+1))[:nb] + intro + (fill_ids*(na//len(fill_ids)+1))[:na] + cue_ids
            seq = seq[:MAX_SEQ_LEN-1]
            ids = torch.tensor([seq], device=device)
            with torch.no_grad(): logits = model(ids)
            pred = logits[0,-1,torch.tensor(cands,device=device)].argmax().item()
            if _PASSKEY_WORDS[pred] == word: ok += 1
        res[d] = ok/PASSKEY_TRIALS
    return res

def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    print(f'DEPTH-AWARE: L0 alphas ({L0_ALPHA_MIN},{L0_ALPHA_MAX}) vs std ({STD_ALPHA_MIN},{STD_ALPHA_MAX})')

    tok = load_tok(); print(f'Tokenizer: {tok.vocab_size()} vocab')
    train_data, val_data = load_data(tok)
    if len(train_data) > MAX_TRAIN_SEQS:
        train_data = train_data[torch.randperm(len(train_data))[:MAX_TRAIN_SEQS]]
    if len(val_data) > MAX_VAL_SEQS: val_data = val_data[:MAX_VAL_SEQS]
    print(f'  train: {len(train_data):,}  val: {len(val_data):,}')

    model = DepthAwareTransformer(
        VOCAB_SIZE, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, FFN_DIM, MAX_SEQ_LEN,
        FULL_ATTN_LAYER, INTERFERENCE, SCALE_EMBED_INIT_VAL
    ).to(dev)
    print(f'Parameters: {model.param_count():,} ({model.param_count()/1e6:.1f}M)')

    opt = torch.optim.AdamW([
        {'params': list(model.other_params()), 'lr': LR},
        {'params': list(model.scale_embed_params()), 'lr': LR * SCALE_EMBED_LR_MULT},
    ], weight_decay=0.1)
    loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
    steps_ep = len(loader) // GRAD_ACCUM
    total    = steps_ep * SCREEN_EPOCHS
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)

    results = {}; t0 = time.time(); gstep = 0
    for ep in range(1, SCREEN_EPOCHS+1):
        model.train(); opt.zero_grad(); aloss = acnt = 0
        for bi, batch in enumerate(loader):
            x = batch[:,:-1].to(dev); y = batch[:,1:].to(dev)
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x).reshape(-1, VOCAB_SIZE), y.reshape(-1))
            (loss/GRAD_ACCUM).backward(); aloss += loss.item(); acnt += 1
            if (bi+1)%GRAD_ACCUM==0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); gstep += 1
                if gstep%200==0: print(f'  Step {gstep}/{total} | Loss {aloss/acnt:.4f}'); aloss=acnt=0

        model.eval(); vl = vc = 0
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(val_data[:2000], batch_size=16):
                x=batch[:,:-1].to(dev); y=batch[:,1:].to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
                    l = F.cross_entropy(model(x).reshape(-1,VOCAB_SIZE), y.reshape(-1))
                vl += l.item()*len(batch); vc += len(batch)
        ppl = math.exp(vl/vc); pk = passkey_accuracy(model, tok, dev)
        pk_mean = sum(pk.values())/len(pk)*100
        se_vals = torch.cat([m.scale_embed.detach().abs().flatten() for m in model.modules() if isinstance(m, DSQGAttentionV6)])
        print(f'\nEp {ep}/{SCREEN_EPOCHS} | Val PPL {ppl:.2f} *')
        print(f' scale_embed |mean|={se_vals.mean():.4f} |max|={se_vals.max():.4f}')
        print(f' Passkey mean={pk_mean:.1f}%')
        print(' ' + '  '.join(f'd={d}:{v*100:.0f}%' for d,v in pk.items()))
        results[f'ppl_ep{ep}'] = round(ppl,2); results[f'passkey_ep{ep}'] = round(pk_mean,1)

    ppl3=results['ppl_ep3']; pk3=results['passkey_ep3']
    ar = (pk3-18.3) - max(0, ppl3-61.75)*0.5
    results.update({'ar_score': round(ar,2), 'elapsed_s': round(time.time()-t0,1),
                    'num_params_M': round(model.param_count()/1e6,1),
                    'l0_alpha': f'({L0_ALPHA_MIN},{L0_ALPHA_MAX})',
                    'description': f'J16D depth-aware 14M: L0 alphas=({L0_ALPHA_MIN},{L0_ALPHA_MAX})'})
    print('\n---')
    for k,v in results.items(): print(f'{k}: {v}')
    out = os.path.join(os.path.dirname(__file__), 'results_j16d_depth_aware.json')
    json.dump(results, open(out,'w'), indent=2); print(f'\nSaved to {out}')

if __name__ == '__main__': main()
