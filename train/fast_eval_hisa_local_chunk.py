#!/usr/bin/env python3
"""
Fast evaluation: HISA local-chunk guarantee.
Trains for 1 epoch, evaluates passkey at all distances.
Compares with/without the local-chunk guarantee.

Uses a simplified but correct HISA implementation (pure PyTorch, no Triton)
to isolate the chunk selection behavior.

Usage:
  python train/fast_eval_hisa_local_chunk.py
"""

import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# Config
# ============================================================
SEED = 42
D = 512
L = 8
H = 8
hd = D // H     # 64
VOCAB = 512
SEQ_LEN = 1024
NUM_CHUNKS = 16
TOP_K = 4
HISA_M = 32
BATCH_SIZE = 4
EPOCHS = 1
LR = 5e-4
CE_CHUNK = 512
PASSKEY_DISTANCES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
PASSKEY_TRIALS = 15
_PASSKEY_WORDS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def seed_all(s):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ============================================================
# Tokenizer: single-char tokens for passkey words, filler tokens for context
# ============================================================
class PasskeyTokenizer:
    def __init__(self, vocab_size=512):
        self.vocab_size = vocab_size
        # Special tokens
        self.word2idx = {'<pad>': 0, '<unk>': 1, '<bos>': 2, '<eos>': 3}
        self.idx2word = {0: '<pad>', 1: '<unk>', 2: '<bos>', 3: '<eos>'}
        # Passkey words get indices 4..11
        for i, w in enumerate(_PASSKEY_WORDS):
            idx = 4 + i
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        # Filler tokens
        idx = 4 + len(_PASSKEY_WORDS)
        for i in range(vocab_size - idx):
            token = f'<f{i}>'
            self.word2idx[token] = idx
            self.idx2word[idx] = token
            idx += 1

    def encode_passkey(self, distance, target_idx, rng):
        """Generate a passkey sequence: [BOS] filler... [TARGET] filler... [EOS] what is the word? [TARGET]"""
        ids = [self.word2idx['<bos>']]
        # Filler before target
        pre_len = distance // 2
        for _ in range(pre_len):
            ids.append(rng.randint(12, self.vocab_size - 1))
        # Target
        ids.append(4 + target_idx)
        # Filler after target
        post_len = distance - pre_len
        for _ in range(post_len):
            ids.append(rng.randint(12, self.vocab_size - 1))
        # Question
        ids.append(self.word2idx['<eos>'])
        # Answer position (model should predict the target word here)
        ids.append(self.word2idx['<unk>'])  # placeholder
        return ids, 4 + target_idx  # return ids and target token id

    def encode_random(self, length, rng):
        """Generate random text for training."""
        return [rng.randint(4, self.vocab_size - 1) for _ in range(length)]

# ============================================================
# Simplified HISA (pure PyTorch, no Triton)
# ============================================================
class SimpleHISA(nn.Module):
    """Simplified HISA for fast eval. Matches the chunk selection logic
    from hierarchical_sparse_attn_v15_hisa.py but uses pure PyTorch."""

    def __init__(self, D, H, hd, num_chunks=16, top_k=4, hisa_m=32,
                 local_chunk_guarantee=True):
        super().__init__()
        self.H = H
        self.hd = hd
        self.num_chunks = num_chunks
        self.top_k = top_k
        self.hisa_m = hisa_m
        self.local_chunk_guarantee = local_chunk_guarantee
        self.W_q = nn.Linear(D, H * hd, bias=False)
        self.W_k = nn.Linear(D, H * hd, bias=False)
        self.W_v = nn.Linear(D, H * hd, bias=False)
        self.W_o = nn.Linear(H * hd, D, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        H, hd = self.H, self.hd
        C = self.num_chunks
        k = self.top_k
        m = self.hisa_m
        chunk_size = math.ceil(N / C)
        device = x.device

        def to_heads(t):
            return t.reshape(B, N, H, hd).transpose(1, 2)

        Q = to_heads(self.W_q(x))
        K = to_heads(self.W_k(x))
        V = to_heads(self.W_v(x))

        pad_len = chunk_size * C - N
        K_pad = F.pad(K, (0, 0, 0, pad_len)) if pad_len > 0 else K
        V_pad = F.pad(V, (0, 0, 0, pad_len)) if pad_len > 0 else V
        Q_pad = F.pad(Q, (0, 0, 0, pad_len)) if pad_len > 0 else Q

        # Chunk representatives (mean K per chunk)
        chunk_reps = K_pad.reshape(B, H, C, chunk_size, hd).mean(dim=3)

        # Routing logits
        routing_logits = torch.matmul(Q, chunk_reps.transpose(-2, -1)) / math.sqrt(hd)

        # Causal mask at chunk level
        positions = torch.arange(N, device=device)
        chunk_starts = torch.arange(C, device=device) * chunk_size
        causal_ok = chunk_starts.unsqueeze(0) < positions.unsqueeze(1)
        routing_logits = routing_logits.masked_fill(~causal_ok[None, None], float('-inf'))

        routing_weights = F.softmax(routing_logits, dim=-1)
        routing_weights = torch.nan_to_num(routing_weights, nan=0.0)

        # Stage 1: Chunk selection (THIS IS THE KEY DIFFERENCE)
        selected_chunks = []  # list of (B, H, k) tensors
        for c_q in range(C):
            q_start = c_q * chunk_size
            q_end = min(q_start + chunk_size, N)
            if q_start >= N:
                selected_chunks.append(torch.full((B, H, k), -1, dtype=torch.long, device=device))
                continue

            if self.local_chunk_guarantee:
                # Include self-chunk in valid set
                n_valid_with_self = min(c_q + 1, C)
                w_c_full = routing_weights[:, :, q_start:q_end, :n_valid_with_self]
                w_mean_full = w_c_full.mean(dim=2)  # (B, H, n_valid_with_self)

                n_others = min(c_q, C - 1)
                if n_others > 0:
                    w_others = w_mean_full.clone()
                    w_others[:, :, c_q] = float('-inf')
                    topk_others = min(k - 1, n_others)
                    _, idx_others = w_others.topk(topk_others, dim=-1)
                    idx = torch.cat([
                        idx_others,
                        torch.full((B, H, 1), c_q, dtype=torch.long, device=device)
                    ], dim=-1)
                    if idx.shape[-1] < k:
                        pad = torch.full((B, H, k - idx.shape[-1]), -1, dtype=torch.long, device=device)
                        idx = torch.cat([idx, pad], dim=-1)
                else:
                    idx = torch.full((B, H, k), c_q, dtype=torch.long, device=device)
                    if k > 1:
                        idx[:, :, 1:] = -1
            else:
                # Original: exclude self-chunk
                n_valid = min(c_q, C)
                if n_valid > 0:
                    w_c = routing_weights[:, :, q_start:q_end, :n_valid]
                    w_mean = w_c.mean(dim=2)
                    n_k = min(k, n_valid)
                    _, idx = w_mean.topk(n_k, dim=-1)
                    if n_k < k:
                        pad = torch.full((B, H, k - n_k), -1, dtype=torch.long, device=device)
                        idx = torch.cat([idx, pad], dim=-1)
                else:
                    idx = torch.full((B, H, k), -1, dtype=torch.long, device=device)

            selected_chunks.append(idx)

        # Stage 2: Attention within selected chunks
        N_padded = chunk_size * C
        K_reshaped = K_pad.view(B, H, C, chunk_size, hd)
        V_reshaped = V_pad.view(B, H, C, chunk_size, hd)

        out = torch.zeros(B, H, N_padded, hd, device=device)

        for c_q in range(C):
            q_start = c_q * chunk_size
            q_end_padded = min(q_start + chunk_size, N_padded)
            if q_start >= N:
                break
            q_actual = q_end_padded - q_start

            Q_c = Q_pad[:, :, q_start:q_end_padded, :]  # (B, H, q_actual, hd)
            sel = selected_chunks[c_q]  # (B, H, k)

            # Gather K, V for selected chunks
            b_idx = torch.arange(B, device=device).view(B, 1, 1)
            h_idx = torch.arange(H, device=device).view(1, H, 1)
            ci = sel.clamp(0, C - 1)  # (B, H, k)

            K_sel = K_reshaped[b_idx, h_idx, ci]  # (B, H, k, chunk_size, hd)
            V_sel = V_reshaped[b_idx, h_idx, ci]

            # Attention: Q_c @ K_sel^T
            # (B, H, q_actual, hd) @ (B, H, k, chunk_size, hd)^T
            scores = torch.matmul(
                Q_c.unsqueeze(2),  # (B, H, 1, q_actual, hd)
                K_sel.transpose(-2, -1)  # (B, H, k, hd, chunk_size)
            ).transpose(2, 3) / math.sqrt(hd)  # (B, H, q_actual, k, chunk_size)

            # Causal mask: k_pos (B,H,k,chunk_size) vs q_pos (q_actual,)
            q_pos = torch.arange(q_start, q_start + q_actual, device=device)
            k_pos_list = []
            for ki in range(k):
                c_idx = sel[:, :, ki]
                starts = c_idx.clamp(0, C - 1) * chunk_size
                ks = starts.unsqueeze(-1) + torch.arange(chunk_size, device=device)
                k_pos_list.append(ks)
            k_pos = torch.stack(k_pos_list, dim=2)  # (B, H, k, chunk_size)
            # Expand: k_pos (B,H,1,k,chunk_size) vs q_pos (1,1,q_actual,1,1)
            causal = (k_pos.unsqueeze(2) < q_pos.view(1, 1, q_actual, 1, 1))
            scores = scores.masked_fill(~causal, float('-inf'))

            # HISA: top-m token selection
            m_actual = min(m, k * chunk_size)
            scores_flat = scores.reshape(B, H, q_actual, -1)  # (B, H, q_actual, k*chunk_size)
            _, top_m = scores_flat.topk(m_actual, dim=-1)
            mask = torch.zeros_like(scores_flat, dtype=torch.bool)
            mask.scatter_(-1, top_m, True)
            scores_flat = scores_flat.masked_fill(~mask, float('-inf'))
            scores = scores_flat.view(B, H, q_actual, k, chunk_size)

            # Softmax and value weighted sum
            attn = F.softmax(scores, dim=-1)  # softmax over tokens within each chunk
            # Reshape for efficient computation
            attn_flat = attn.reshape(B, H, q_actual, k * chunk_size)
            V_flat = V_sel.reshape(B, H, k * chunk_size, hd)
            out_c = torch.matmul(attn_flat, V_flat)  # (B, H, q_actual, hd)

            out[:, :, q_start:q_end_padded, :] = out_c

        out_flat = out[:, :, :N, :].transpose(1, 2).reshape(B, N, H * hd)
        return self.W_o(out_flat)

# ============================================================
# Transformer block + model
# ============================================================
class Block(nn.Module):
    def __init__(self, D, H, hd, local_chunk_guarantee=True):
        super().__init__()
        self.attn = SimpleHISA(D, H, hd,
                               num_chunks=NUM_CHUNKS, top_k=TOP_K,
                               hisa_m=HISA_M,
                               local_chunk_guarantee=local_chunk_guarantee)
        self.ffn = nn.Sequential(
            nn.Linear(D, D * 4),
            nn.GELU(),
            nn.Linear(D * 4, D)
        )
        self.norm1 = nn.LayerNorm(D, eps=1e-5)
        self.norm2 = nn.LayerNorm(D, eps=1e-5)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class Model(nn.Module):
    def __init__(self, vocab, D, L, H, local_chunk_guarantee=True):
        super().__init__()
        self.embed = nn.Embedding(vocab, D, padding_idx=0)
        self.pos_embed = nn.Embedding(SEQ_LEN, D)
        self.layers = nn.ModuleList([
            Block(D, H, D // H, local_chunk_guarantee=local_chunk_guarantee)
            for _ in range(L)
        ])
        self.norm = nn.LayerNorm(D, eps=1e-5)
        self.lm_head = nn.Linear(D, vocab, bias=False)
        # Tie weights
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids, positions=None):
        B, N = input_ids.shape
        if positions is None:
            positions = torch.arange(N, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(positions)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

# ============================================================
# Passkey evaluation
# ============================================================
def eval_passkey(model, tokenizer, device, distances=None, trials=15):
    if distances is None:
        distances = PASSKEY_DISTANCES

    model.eval()
    results = {}

    for d in distances:
        correct = 0
        total = 0
        for t in range(trials):
            target_idx = t % len(_PASSKEY_WORDS)
            rng = random.Random(SEED + d * 100 + t)
            ids, target_id = tokenizer.encode_passkey(d, target_idx, rng)

            if len(ids) > SEQ_LEN:
                ids = ids[:SEQ_LEN]

            # Input: all except last token. Target: last token (the answer).
            input_ids = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
            positions = torch.arange(len(ids) - 1, device=device).unsqueeze(0)

            with torch.no_grad():
                logits = model(input_ids, positions)
                last_logits = logits[0, -1, :]  # (vocab,)
                pred_id = last_logits.argmax().item()

            if pred_id == target_id:
                correct += 1
            total += 1

        results[d] = (correct / total) * 100 if total > 0 else 0

    return results

# ============================================================
# Training
# ============================================================
def train(model, tokenizer, epochs=EPOCHS, lr=LR):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    total_samples = 200
    step = 0
    total_steps = epochs * total_samples // BATCH_SIZE
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    for epoch in range(epochs):
        losses = []
        for _ in range(total_samples // BATCH_SIZE):
            rng = random.Random(SEED + epoch * 10000 + _)
            batch_ids = []
            for _ in range(BATCH_SIZE):
                ids = tokenizer.encode_random(SEQ_LEN, rng)
                batch_ids.append(ids)

            batch = torch.tensor(batch_ids, dtype=torch.long, device=device)
            positions = torch.arange(SEQ_LEN, device=device).unsqueeze(0).expand(BATCH_SIZE, -1)

            logits = model(batch, positions)
            logits = logits[:, :-1, :].reshape(-1, VOCAB)
            targets = batch[:, 1:].reshape(-1)

            loss = F.cross_entropy(logits, targets, ignore_index=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            losses.append(loss.item())
            step += 1

        avg_loss = sum(losses) / len(losses)
        print(f'  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f} ppl={math.exp(avg_loss):.2f} steps={step}')

    return model

# ============================================================
# Main
# ============================================================
def main():
    print('HISA Local-Chunk Guarantee: Fast Evaluation')
    print(f'  D={D} L={L} H={H} hd={hd}')
    print(f'  seq_len={SEQ_LEN} chunks={NUM_CHUNKS} chunk_size={math.ceil(SEQ_LEN/NUM_CHUNKS)}')
    print(f'  top_k={TOP_K} hisa_m={HISA_M} device={device}')
    print()

    tokenizer = PasskeyTokenizer(VOCAB)

    # ---- RUN 1: WITHOUT guarantee ----
    print('=' * 60)
    print('RUN 1: WITHOUT local-chunk guarantee (original)')
    print('=' * 60)
    seed_all(SEED)
    model_no = Model(VOCAB, D, L, H, local_chunk_guarantee=False).to(device)
    train(model_no, tokenizer)

    pk_no = eval_passkey(model_no, tokenizer, device)
    print('\n  Passkey (no guarantee):')
    for d in PASSKEY_DISTANCES:
        print(f'    d={d:>4}: {pk_no[d]:.0f}%')
    mean_no = sum(pk_no.values()) / len(pk_no)
    print(f'  Mean: {mean_no:.1f}%  |  d=32: {pk_no.get(32, "N/A")}%')

    # ---- RUN 2: WITH guarantee ----
    print()
    print('=' * 60)
    print('RUN 2: WITH local-chunk guarantee (fix)')
    print('=' * 60)
    seed_all(SEED)
    model_yes = Model(VOCAB, D, L, H, local_chunk_guarantee=True).to(device)
    train(model_yes, tokenizer)

    pk_yes = eval_passkey(model_yes, tokenizer, device)
    print('\n  Passkey (with guarantee):')
    for d in PASSKEY_DISTANCES:
        print(f'    d={d:>4}: {pk_yes[d]:.0f}%')
    mean_yes = sum(pk_yes.values()) / len(pk_yes)
    print(f'  Mean: {mean_yes:.1f}%  |  d=32: {pk_yes.get(32, "N/A")}%')

    # ---- Comparison ----
    print()
    print('=' * 60)
    print('COMPARISON (with - without)')
    print('=' * 60)
    print(f'  {"Distance":>8} {"No Guarantee":>14} {"With Guarantee":>14} {"Delta":>8}')
    for d in PASSKEY_DISTANCES:
        delta = pk_yes.get(d, 0) - pk_no.get(d, 0)
        print(f'  d={d:>4} {pk_no.get(d, 0):>13.0f}% {pk_yes.get(d, 0):>13.0f}% {delta:>+7.0f}%')
    delta_mean = mean_yes - mean_no
    print(f'  {"MEAN":>8} {mean_no:>13.1f}% {mean_yes:>13.1f}% {delta_mean:>+7.1f}%')
    print()
    print(f'  d=32 improvement: {pk_yes.get(32, 0) - pk_no.get(32, 0):+.0f}%')

if __name__ == '__main__':
    main()
