"""
Distance-Conditioned Loss Evaluation — condP vs Standard Baseline

PURPOSE
-------
Find WHERE condP still loses to baseline by bucketing per-token cross-entropy
loss by absolute position within the sequence.

This tells us if losses are concentrated at positions requiring specific
lookback distances (e.g., condP may be better at long-range but worse at
short-range, or vice versa).

DESIGN
------
- Load both models
- No external test data found → use 10 hardcoded diverse paragraphs
- For each sequence: compute per-token cross-entropy loss
- Bucket by absolute position: [0-16], [17-64], [65-256], [257-512],
  [513-1024], [1025-2048]
- Compute delta (condP_loss - baseline_loss) per position
  (positive = condP worse, negative = condP better)

Usage:
  cd /home/dlewis3/Desktop/AI/DWARF
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 benchmarks/eval_distance_loss.py
"""

import json, math, os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# ─── Constants ────────────────────────────────────────────────────────────────
VOCAB_SIZE    = 32000
MAX_SEQ_LEN   = 2048
EMBEDDING_DIM = 256
NUM_LAYERS    = 6
NUM_HEADS     = 8
FFN_DIM       = 1024
INTERFERENCE  = 3

_DENSE_LOCAL_W_P     = 64
_DYADIC_LONG_RANGE_P = [96, 128, 192, 256, 384, 512, 768, 1024, 1536]
_COND_P_OFFSETS      = sorted(set(range(0, _DENSE_LOCAL_W_P + 1)) |
                               set(_DYADIC_LONG_RANGE_P))
assert len(_COND_P_OFFSETS) == 74


# ─── condP Architecture ───────────────────────────────────────────────────────

class DSQGAttentionN(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=2048,
                 offsets=None, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads     = num_heads
        self.head_dim      = embedding_dim // num_heads
        if offsets is None:
            offsets = _COND_P_OFFSETS
        self.register_buffer('offsets', torch.tensor(offsets, dtype=torch.long))
        self.n_offsets = len(offsets)
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj   = nn.Linear(embedding_dim, embedding_dim, bias=True)
        self.gate_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        alphas     = torch.linspace(0.2, 2.0, num_heads)
        delta_vals = torch.tensor(
            [math.log(1.0 + d) for d in offsets], dtype=torch.float32)
        self.pos_bias = nn.Parameter(
            -delta_vals.unsqueeze(1) * alphas.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).permute(0, 2, 1, 3)
        k = k.view(B, N, H, HD).permute(0, 2, 1, 3)
        v = v.view(B, N, H, HD).permute(0, 2, 1, 3)
        scale = HD ** -0.5
        K_list, V_list = [], []
        for delta in self.offsets.tolist():
            if delta == 0:
                K_list.append(k); V_list.append(v)
            elif delta >= N:
                K_list.append(torch.zeros_like(k))
                V_list.append(torch.zeros_like(v))
            else:
                pad = k.new_zeros(B, H, delta, HD)
                K_list.append(torch.cat([pad, k[:, :, :N - delta, :]], dim=2))
                V_list.append(torch.cat([pad, v[:, :, :N - delta, :]], dim=2))
        K_all  = torch.stack(K_list, dim=3)
        V_all  = torch.stack(V_list, dim=3)
        scores = (q.unsqueeze(3) * K_all).sum(-1) * scale
        scores = scores + self.pos_bias.T.unsqueeze(0).unsqueeze(2)
        n_idx  = torch.arange(N, device=x.device).unsqueeze(1)
        d_idx  = self.offsets.unsqueeze(0)
        scores = scores.masked_fill(
            (n_idx < d_idx).unsqueeze(0).unsqueeze(0), float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        out   = (alpha.unsqueeze(-1) * V_all).sum(dim=3)
        flat  = out.permute(0, 2, 1, 3).reshape(B, N, D)
        gate  = torch.sigmoid(self.gate_proj(x))
        return self.dropout(self.out_proj(flat * gate))


class FFN_condP(nn.Module):
    def __init__(self, d, ffn, dropout=0.1):
        super().__init__()
        self.fc1  = nn.Linear(d, ffn)
        self.fc2  = nn.Linear(ffn, d)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class DSQGBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len,
                 dropout=0.1, interference=False, offsets=None):
        super().__init__()
        self.interference = interference
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.attn  = DSQGAttentionN(embedding_dim, num_heads,
                                     seq_len=seq_len, dropout=dropout,
                                     offsets=offsets)
        self.ffn   = FFN_condP(embedding_dim, ffn_dim, dropout)
        if interference:
            self.inter_norm = nn.LayerNorm(embedding_dim)
            self.inter_gate = nn.Linear(embedding_dim, embedding_dim)
            self.inter_pool = nn.Linear(embedding_dim, embedding_dim)

    def _attn_fn(self, x):
        return self.attn(self.norm1(x))

    def forward(self, x):
        x = x + torch.utils.checkpoint.checkpoint(
            self._attn_fn, x, use_reentrant=False)
        if self.interference:
            xi = self.inter_norm(x)
            B, N, D = xi.shape
            counts = torch.arange(1, N + 1, device=xi.device,
                                  dtype=xi.dtype).view(1, N, 1)
            pool = xi.cumsum(dim=1) / counts
            x = x + torch.sigmoid(self.inter_gate(xi)) * self.inter_pool(pool)
        x = x + self.ffn(self.norm2(x))
        return x


class CondNTransformer(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 seq_len=MAX_SEQ_LEN, interference_interval=INTERFERENCE,
                 dropout=0.1, offsets=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(seq_len + 2, embedding_dim)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList([
            DSQGBlock(embedding_dim, num_heads, ffn_dim, seq_len,
                      dropout=dropout,
                      interference=(i % interference_interval == interference_interval - 1),
                      offsets=offsets)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embedding_dim)
        self.out  = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.out.weight = self.embedding.weight

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.drop(self.embedding(idx) + self.pos_embed(pos))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


# ─── Baseline Architecture ────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embedding_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv_proj  = nn.Linear(embedding_dim, 3 * embedding_dim, bias=True)
        self.out_proj  = nn.Linear(embedding_dim, embedding_dim, bias=True)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        H, HD   = self.num_heads, self.head_dim
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(D, dim=-1)
        q = q.view(B, N, H, HD).transpose(1, 2)
        k = k.view(B, N, H, HD).transpose(1, 2)
        v = v.view(B, N, H, HD).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(
            self.causal_mask[:N, :N].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


class FFN_base(nn.Module):
    def __init__(self, embedding_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1     = nn.Linear(embedding_dim, ffn_dim)
        self.fc2     = nn.Linear(ffn_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, ffn_dim, seq_len, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(embedding_dim)
        self.attn = CausalSelfAttention(embedding_dim, num_heads, seq_len, dropout)
        self.ln2  = nn.LayerNorm(embedding_dim)
        self.ffn  = FFN_base(embedding_dim, ffn_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class StandardTransformerBaseline(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM,
                 num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM,
                 seq_len=MAX_SEQ_LEN, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(seq_len, embedding_dim)
        self.blocks    = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, ffn_dim, seq_len, dropout)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(embedding_dim)
        self.out_proj  = nn.Linear(embedding_dim, vocab_size, bias=False)

    def forward(self, idx):
        B, N = idx.shape
        pos  = torch.arange(N, device=idx.device).unsqueeze(0)
        x    = self.token_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.out_proj(self.ln_final(x))


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class BPETokenizerWrapper:
    def __init__(self, tok): self.tokenizer = tok
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids):  return self.tokenizer.decode(ids)


# ─── Diverse text paragraphs ──────────────────────────────────────────────────
# 10 paragraphs, diverse topics. Each ~700-900 words.
# Concatenated and chunked into 2048-token windows.

PARAGRAPHS = [
    # 1. Science — Quantum Mechanics
    """Quantum mechanics is the branch of physics that describes the behavior of particles at the atomic and subatomic level. Unlike classical mechanics, which predicts deterministic outcomes, quantum mechanics operates on probabilities. The central equation of quantum mechanics is the Schrödinger equation, which describes how the quantum state of a physical system evolves over time. A quantum state is represented by a wave function, a mathematical object that encodes all possible outcomes of measurements on the system. When a measurement is made, the wave function collapses to a specific eigenstate corresponding to the measured value. This probabilistic nature led Einstein to famously object that "God does not play dice," though subsequent experiments confirmed quantum predictions with extraordinary precision. The double-slit experiment is perhaps the most striking demonstration of quantum weirdness: when individual particles pass through two slits, they create an interference pattern on a detector screen, as if each particle passes through both slits simultaneously. This wave-particle duality is fundamental to quantum theory. Heisenberg's uncertainty principle states that the position and momentum of a particle cannot both be known to arbitrary precision simultaneously. The more precisely position is determined, the less precisely momentum can be known, and vice versa. This is not a limitation of measurement technology but a fundamental property of nature. Quantum entanglement is another profound phenomenon in which two particles become correlated such that the quantum state of each particle cannot be described independently. Measuring one particle instantaneously affects the state of the other, regardless of the distance between them. Einstein called this "spooky action at a distance" and believed it indicated that quantum mechanics was incomplete, but Bell's theorem and subsequent experiments confirmed that entanglement is real and cannot be explained by local hidden variables. Quantum computing exploits these properties to perform calculations that would be infeasible for classical computers. A quantum bit, or qubit, can exist in a superposition of zero and one, allowing quantum computers to explore many solutions simultaneously. Algorithms like Shor's algorithm for factoring large numbers and Grover's algorithm for searching unsorted databases demonstrate quantum speedups that could revolutionize cryptography and data analysis. Current quantum computers are still limited by noise and decoherence, but rapid progress is being made in error correction and qubit stability. The applications of quantum mechanics extend far beyond computing. Lasers, semiconductors, magnetic resonance imaging, and atomic clocks all rely on quantum principles. Understanding quantum mechanics has been essential for the development of modern technology and continues to drive fundamental research in physics, chemistry, and materials science.""",

    # 2. Narrative — A Day in the Mountains
    """The morning began with a thin mist clinging to the valley below as Marcus laced his boots on the wooden porch of the alpine cabin. He had been planning this hike for months, studying the topographic maps late into winter evenings, tracing the ridge line with his finger and imagining the view from the summit. The trail began easily enough, winding through old-growth pine forest where the light filtered down in pale columns and the only sounds were the creak of branches and the distant percussion of a woodpecker. Marcus walked at an easy pace, letting his legs warm to the rhythm of the ascent. He passed a creek still swollen with snowmelt, crossed it on a log bridge worn smooth by decades of boots, and began climbing in earnest as the trees thinned and the sky opened above him. By mid-morning he had gained significant elevation. The air was cooler here, carrying a sharpness that made every breath feel clarifying. He stopped at a boulder to eat an apple and survey the country below: the valley floor with its patchwork of meadows and dark forest, a glint of river snaking through, and beyond, more ridges layering blue and purple into the distance. A marmot watched him from thirty yards away with apparent indifference, chewing something methodically before vanishing into a crevice. The upper section of the trail became steeper and rockier, requiring Marcus to use his hands in places. He found a rhythm: three steps, pause, three steps. The summit block was a loose scree of granite that shifted underfoot, demanding concentration and care. And then, abruptly, he was on top. The view was everything he had imagined and more. In every direction the land unrolled beneath him, peaks and valleys and the thin silver threads of rivers. He could see a distant lake burning like a mirror in the afternoon light. The wind was steady and cold. He sat on a flat rock, drank water, and felt what he always felt at high places: a combination of smallness and expansion, the paradox of feeling simultaneously insignificant and fully alive. He lingered for nearly an hour before the clouds began building to the west. He knew that look. He started down, moving efficiently, the descent harder on his knees than the climb had been on his lungs. By the time he reached the treeline the clouds had piled into towers and the first distant thunder reached him. He made it to the cabin with twenty minutes to spare before the storm broke, lightning splitting the ridge where he had stood, rain hammering the tin roof in sheets.""",

    # 3. Code — Python Programming Concepts
    """Python is a high-level, interpreted programming language known for its readable syntax and broad applicability. Created by Guido van Rossum and first released in 1991, Python has grown into one of the most widely used programming languages in the world. Its design philosophy emphasizes code readability and simplicity, following principles sometimes called "Pythonic." Functions in Python are first-class objects, meaning they can be passed as arguments, returned from other functions, and assigned to variables. This enables powerful patterns like higher-order functions and decorators. A decorator is a function that wraps another function to extend or modify its behavior without permanently modifying it. The syntax uses the at-sign symbol above a function definition. List comprehensions provide a concise way to create lists from iterables with optional filtering: the expression inside square brackets specifies the output element, the for clause iterates over an iterable, and an optional if clause filters elements. Generator expressions are similar but produce values lazily, one at a time, making them memory-efficient for large datasets. Python's object-oriented system supports classes with inheritance, multiple inheritance, and mixins. The special methods, sometimes called dunder methods because they begin and end with double underscores, allow classes to define behavior for built-in operations. For example, implementing the length dunder method allows an object to respond to the built-in len function, and implementing the iterator dunder methods allows objects to be used in for loops. Context managers, implemented using the with keyword, ensure that resources like file handles and database connections are properly acquired and released. The contextlib module provides utilities for creating context managers using generators. Python's standard library is vast, covering areas from file input and output to networking, cryptography, data compression, and regular expressions. The json module serializes and deserializes JSON data. The pathlib module provides an object-oriented interface to filesystem paths. The collections module provides specialized container data types like namedtuples, defaultdicts, and deques. The typing module adds support for type annotations, which improve code readability and enable static analysis tools like mypy. Python's package ecosystem is one of its greatest strengths: the Python Package Index hosts hundreds of thousands of third-party packages covering machine learning, web development, scientific computing, data visualization, and virtually every other domain.""",

    # 4. Dialogue — A Philosophical Conversation
    """The two friends had been arguing for what felt like hours, though the coffee cups had only just gone cold. Elena leaned back in her chair and watched the rain trace lines down the window. "You keep assuming that consciousness is something that emerges from complexity," she said. "But emergence doesn't explain the hard problem. Explaining how neural correlates map to behavior is not the same as explaining why there is something it is like to be you." Dimitri turned his cup slowly in his hands. "And you keep retreating to mystery. Every time we get close to an explanation, you move the goalpost. The phenomenal experience, you say, is not captured by any functional description. But how would you know? How could you recognize an explanation if you saw one?" "I'd know because it would feel like enough. Right now, nothing feels like enough because the explanatory gap is real." "The gap is an artifact of how we frame the question," Dimitri said. "If we ask why water is wet, we can answer that in terms of molecular interactions. Nobody feels there is still something left unexplained after that answer. The persistence of the hard problem might just be a failure of intuition." Elena shook her head slowly. "Water and wetness are both third-person phenomena. Consciousness is first-person. That asymmetry is not a framing artifact; it is the entire phenomenon we are trying to explain." "But the first-person perspective is itself a functional property. There is a subject who represents itself as having experiences. That is implementable in principle." "Implementable is doing a lot of work there. A philosophical zombie is functionally identical to a conscious being by definition. Yet something would be missing." Dimitri sighed and looked out at the rain with her. After a long silence he said, "Maybe the question is not whether it is explainable but whether the explanation would satisfy anyone. Philosophy has a way of raising standards as the science advances." Elena smiled despite herself. "That is the most honest thing you have said all afternoon." They sat together in the particular silence that follows the exhaustion of argument, not resolved but at rest for the moment, the rain steady against the glass.""",

    # 5. Historical — The Roman Republic
    """The Roman Republic, which lasted from approximately 509 BCE to 27 BCE, was one of the most consequential political experiments in human history. Emerging from the overthrow of the Tarquin kings, the Republic established a system of government designed to prevent any single individual from accumulating too much power. The two consuls who served as chief executives were elected annually and each held the power of veto over the other. The Senate, originally an advisory body composed of former magistrates and aristocrats, gradually accumulated enormous influence over foreign policy, finances, and the appointment of provincial governors. The tension between patricians, who held traditional political privileges, and plebeians, the broader citizenry, drove a series of constitutional reforms over two centuries. The Conflict of the Orders resulted in plebeians gaining the right to hold major offices and to pass binding laws through their own assembly. The twelve tables, Rome's earliest written legal code, established that the law applied to all citizens regardless of class. The Middle Republic saw Rome expand aggressively beyond Italy. The three Punic Wars with Carthage defined this era. The first established Roman dominance in Sicily. The second, in which Hannibal famously crossed the Alps with war elephants and invaded Italy, brought Rome to the brink of defeat but ultimately ended in Roman victory after Scipio Africanus defeated Hannibal at the Battle of Zama in 202 BCE. The third resulted in the complete destruction of Carthage. Roman provincial governance became increasingly complex as the state acquired territories across the Mediterranean. Generals commanding provincial armies accumulated personal wealth and loyal troops, creating a structural vulnerability that would eventually undermine republican institutions. The Gracchi brothers, tribunes who attempted agrarian reforms to address inequality, were both killed, setting precedents for political violence. Marius reformed the army by opening recruitment to the landless poor, creating soldiers who owed their livelihood to their commanders rather than the state. Sulla marched his legions on Rome twice. Caesar crossed the Rubicon. The Republic, having generated the very forces that destroyed it, gave way finally to the Augustan Principate.""",

    # 6. Mathematical — Number Theory
    """Number theory is the branch of pure mathematics devoted to the study of integers and integer-valued functions. Ancient civilizations recognized interesting properties of numbers, but systematic study began with the Greeks. Euclid's proof that there are infinitely many prime numbers remains one of the most elegant arguments in mathematics: assume finitely many primes, multiply them together and add one, and the result must be divisible by a prime not in the original list, yielding a contradiction. The Euclidean algorithm for computing the greatest common divisor of two integers is one of the oldest algorithms still in widespread use. Given two positive integers, the algorithm repeatedly replaces the larger with the remainder when divided by the smaller, until the remainder is zero. The last nonzero remainder is the greatest common divisor. The fundamental theorem of arithmetic states that every integer greater than one can be uniquely factored into a product of primes, up to the order of factors. This unique factorization property underpins much of number theory. Modular arithmetic, developed systematically by Gauss in his Disquisitiones Arithmeticae, studies the remainder when integers are divided by a fixed modulus. Fermat's little theorem states that if p is prime and a is not divisible by p, then a raised to the power p minus one is congruent to one modulo p. This result has profound applications in cryptography. The RSA cryptosystem relies on the difficulty of factoring large numbers that are products of two large primes. Given only the product, it is computationally infeasible to recover the factors with current algorithms for sufficiently large inputs, yet given the factors, encryption and decryption are straightforward. Diophantine equations are polynomial equations for which integer solutions are sought. Fermat's last theorem, that no positive integers satisfy the equation with integer exponents greater than two for the sum of two nth powers equaling a third, was conjectured in 1637 and proved by Andrew Wiles only in 1995 after more than three hundred years of effort. The proof required deep machinery from algebraic geometry and the theory of elliptic curves, illustrating how distant branches of mathematics can illuminate simple-sounding questions about integers. The Riemann hypothesis, concerning the distribution of zeros of the Riemann zeta function, remains unproven and is considered by many mathematicians the most important unsolved problem in mathematics.""",

    # 7. Medical — The Immune System
    """The immune system is the body's defense network against pathogens, damaged cells, and foreign substances. It comprises an intricate array of cells, tissues, and molecular signals that work together to identify and eliminate threats while tolerating the body's own tissues. The system is conventionally divided into innate immunity and adaptive immunity. Innate immunity is the first line of defense. It responds rapidly and non-specifically to pathogens through pattern recognition receptors that detect conserved molecular signatures found on bacteria, viruses, and fungi but not on host cells. Toll-like receptors are among the best-characterized pattern recognition receptors. When activated, they trigger inflammatory responses that recruit immune cells to the site of infection and release signaling molecules called cytokines. Natural killer cells patrol the body and eliminate cells that lack proper surface markers indicating healthy host identity. Macrophages engulf and destroy pathogens and cellular debris through phagocytosis, presenting molecular fragments on their surface to initiate adaptive immune responses. The adaptive immune system is slower but highly specific and capable of immunological memory. It centers on lymphocytes: B cells and T cells. B cells produce antibodies, protein molecules that recognize and bind to specific antigens on pathogens with extraordinary precision. Each B cell expresses a unique receptor, and when it encounters its matching antigen and receives appropriate co-stimulatory signals, it proliferates and differentiates into plasma cells that secrete antibodies in large quantities. Helper T cells coordinate immune responses by secreting cytokines that stimulate B cells, activate cytotoxic T cells, and regulate macrophage activity. Cytotoxic T cells directly kill infected cells by recognizing pathogen-derived peptides presented on cell surfaces. After an infection clears, a subset of B cells and T cells persist as memory cells. Upon re-exposure to the same pathogen, these memory cells mount a faster and stronger response, which is the immunological basis of vaccination. Autoimmune diseases arise when the immune system mistakenly attacks the body's own tissues. Multiple sclerosis, rheumatoid arthritis, and type one diabetes are autoimmune conditions involving failures of self-tolerance. Treatment often involves immunosuppressive drugs that dampen immune activity, though this comes with increased susceptibility to infection.""",

    # 8. Philosophical — Free Will and Determinism
    """The debate over free will and determinism is one of the oldest in philosophy, touching questions of moral responsibility, personal identity, and the nature of causation. Determinism is the view that every event, including every human action and decision, is the inevitable consequence of prior causes operating according to natural laws. If determinism is true, then given a complete description of the physical state of the universe at any moment and the complete set of natural laws, every future state is in principle predictable. Hard determinism holds that this is incompatible with genuine free will. If my action tomorrow is already determined by the state of particles in my brain today, which is in turn determined by events before my birth, in what meaningful sense did I choose it? The compatibilist tradition, represented by Hume, Mill, and more recently Frankfurt and Dennett, argues that free will and determinism are not in conflict because free will is correctly defined not as the ability to have acted otherwise given identical prior conditions but as the ability to act in accordance with one's own reasons and desires without external compulsion. On this view, an action is free when it flows from the agent's own deliberative processes, even if those processes are deterministic. Libertarian free will, distinct from political libertarianism, holds that determinism is false and that humans possess genuine causal power not reducible to prior physical causes. Some libertarians appeal to quantum indeterminacy at the neurological level, though critics argue that random quantum events do not constitute freedom of any interesting kind. Frankfurt's cases introduced a different challenge: imagine a neuroscientist who can monitor and intervene in your brain, prepared to ensure you make a certain decision if you show signs of deciding otherwise, but who never actually needs to intervene because you make that decision freely. In this scenario, it seems you could not have decided otherwise but still acted freely, suggesting that alternative possibilities are not required for moral responsibility. Recent experimental philosophy has examined lay intuitions about free will and moral responsibility, finding that people's attributions of responsibility are sensitive to context in ways that resist simple theoretical systematization.""",

    # 9. Environmental — Climate Change
    """Climate change refers to long-term shifts in global temperatures and weather patterns, driven primarily since the mid-twentieth century by human activities. The burning of fossil fuels releases carbon dioxide and other greenhouse gases into the atmosphere. Greenhouse gases trap outgoing infrared radiation from Earth's surface, causing the planet to warm. The concentration of atmospheric carbon dioxide has increased from approximately 280 parts per million before the industrial revolution to more than 420 parts per million today, the highest level in at least 800,000 years as measured from ice core records. The consequences of warming are already manifesting across multiple Earth systems. Global average surface temperature has risen by approximately 1.1 degrees Celsius above pre-industrial levels. Arctic warming is occurring roughly four times faster than the global average, causing accelerated melting of sea ice and the Greenland ice sheet. Thermal expansion of ocean water and melting ice contribute to sea level rise, which threatens coastal communities and low-lying island nations. Extreme weather events including heatwaves, heavy precipitation, droughts, and intense tropical cyclones are becoming more frequent and severe in many regions. Ocean acidification, caused by the absorption of carbon dioxide into seawater forming carbonic acid, threatens marine ecosystems, particularly coral reefs and shell-forming organisms. The Paris Agreement, adopted in 2015, aimed to limit warming to well below two degrees Celsius above pre-industrial levels and to pursue efforts to limit warming to 1.5 degrees. Achieving these targets requires rapid, far-reaching transitions in energy, land use, transport, and industry. The deployment of renewable energy technologies, particularly solar and wind power, has accelerated dramatically due to declining costs. Electric vehicles are gaining market share in major economies. Carbon capture and storage technologies aim to remove carbon dioxide from point sources or directly from the atmosphere, though these remain expensive and unproven at scale. Adaptation measures, including sea walls, drought-resistant crops, and heat action plans, are increasingly necessary alongside mitigation efforts. The equity dimensions of climate policy are significant: historically large emitters bear primary responsibility, while low-income and developing nations often face the greatest impacts with the least resources to adapt.""",

    # 10. Technology — Neural Networks and Deep Learning
    """Neural networks are computational systems loosely inspired by the structure of biological neural networks in animal brains. A neural network consists of layers of interconnected nodes, each of which performs a simple computation on its inputs and passes the result to subsequent layers. The power of neural networks arises from the composition of many such simple computations and from the adjustment of connection weights through training. Deep learning refers to neural networks with many hidden layers between the input and output. The depth allows the network to learn hierarchical representations of data, with early layers detecting simple features and later layers combining these into increasingly abstract representations. In image recognition, early layers detect edges and colors, intermediate layers detect textures and shapes, and later layers detect high-level object features. The training process uses gradient descent, an optimization algorithm that iteratively adjusts the network weights to minimize a loss function measuring the discrepancy between network outputs and desired outputs. The gradient of the loss with respect to each weight is computed using backpropagation, which applies the chain rule of calculus to propagate error signals backward through the network. Convolutional neural networks are specialized architectures for processing grid-structured data like images. Convolutional layers apply learned filters across the input, detecting local patterns regardless of position. Pooling layers reduce spatial resolution, creating translation invariance. These properties make convolutional networks highly effective for image classification, object detection, and segmentation. Recurrent neural networks process sequential data by maintaining a hidden state that is updated at each step. Long short-term memory networks address the vanishing gradient problem in plain recurrent networks by using gating mechanisms that control what information is retained or discarded. Transformer architectures, introduced in 2017, use attention mechanisms to relate every position in a sequence to every other position, enabling highly parallel computation and effective modeling of long-range dependencies. Transformers have become dominant in natural language processing, powering language models of increasing scale and capability. Generative adversarial networks consist of two networks, a generator and a discriminator, trained in opposition. The generator produces synthetic data samples and the discriminator attempts to distinguish them from real data. This adversarial training produces remarkably realistic synthetic images and other media.""",
]


# ─── Position buckets ─────────────────────────────────────────────────────────

BUCKETS = [
    ('0-16',     0,    16),
    ('17-64',    17,   64),
    ('65-256',   65,  256),
    ('257-512',  257, 512),
    ('513-1024', 513, 1024),
    ('1025-2047', 1025, 2047),
]


def bucket_index(pos):
    for name, lo, hi in BUCKETS:
        if lo <= pos <= hi:
            return name
    return None


# ─── Per-sequence loss ────────────────────────────────────────────────────────

def compute_per_token_loss(model, token_ids, device):
    """
    Given a 1D tensor of token_ids (length L),
    compute per-token cross-entropy loss at each position i in [0, L-2]:
      loss[i] = -log P(token[i+1] | token[0..i])
    Returns a list of floats of length L-1.
    """
    model.eval()
    ids = token_ids.unsqueeze(0).to(device)  # (1, L)
    with torch.no_grad():
        logits = model(ids)                   # (1, L, V)
    # input positions 0..L-2, predicting tokens 1..L-1
    logits_in = logits[0, :-1, :]            # (L-1, V)
    targets   = ids[0, 1:]                    # (L-1,)
    losses    = F.cross_entropy(logits_in, targets, reduction='none')  # (L-1,)
    return losses.cpu().tolist()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--condp_checkpoint',
                        default='checkpoints/2048_condP_checkpoints/best.pt')
    parser.add_argument('--baseline_checkpoint',
                        default='checkpoints/2048_standard_baseline_checkpoints/best.pt')
    parser.add_argument('--tokenizer',
                        default='benchmarks/results/2048_condI_tokenizer.json')
    parser.add_argument('--output',
                        default='benchmarks/results/distance_loss_results.json')
    parser.add_argument('--chunk_size', type=int, default=2048,
                        help='Sequence chunk size (default 2048 = MAX_SEQ_LEN)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    from tokenizers import Tokenizer
    if not os.path.exists(args.tokenizer):
        print(f'ERROR: tokenizer not found: {args.tokenizer}'); sys.exit(1)
    tokenizer = BPETokenizerWrapper(Tokenizer.from_file(args.tokenizer))
    print(f'Loaded tokenizer: {args.tokenizer}')

    # Load condP
    print(f'\nLoading condP from: {args.condp_checkpoint}')
    condp_model = CondNTransformer(offsets=_COND_P_OFFSETS).to(device)
    state = torch.load(args.condp_checkpoint, map_location=device, weights_only=True)
    condp_model.load_state_dict(state)
    condp_model.eval()
    condp_params = sum(p.numel() for p in condp_model.parameters())
    print(f'  condP: {condp_params:,} params')

    # Load baseline
    print(f'Loading baseline from: {args.baseline_checkpoint}')
    base_model = StandardTransformerBaseline().to(device)
    state = torch.load(args.baseline_checkpoint, map_location=device, weights_only=True)
    base_model.load_state_dict(state)
    base_model.eval()
    base_params = sum(p.numel() for p in base_model.parameters())
    print(f'  Baseline: {base_params:,} params')

    # Tokenize all paragraphs and concatenate
    print(f'\nTokenizing {len(PARAGRAPHS)} diverse paragraphs...')
    all_ids = []
    for i, para in enumerate(PARAGRAPHS):
        ids = tokenizer.encode(para)
        all_ids.extend(ids)
        print(f'  Paragraph {i+1}: {len(ids)} tokens')
    print(f'  Total: {len(all_ids)} tokens')

    # Split into chunks of chunk_size
    chunk_size = args.chunk_size
    chunks = []
    for start in range(0, len(all_ids) - chunk_size, chunk_size):
        chunk = all_ids[start: start + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(torch.tensor(chunk, dtype=torch.long))
    print(f'  Chunks of {chunk_size} tokens: {len(chunks)}')

    if not chunks:
        # Fallback: use whatever we have
        chunks = [torch.tensor(all_ids[:min(len(all_ids), chunk_size)], dtype=torch.long)]
        print(f'  Warning: using single truncated chunk of {len(chunks[0])} tokens')

    # Accumulate per-position losses
    # bucket_losses[bucket_name][model] = list of losses
    bucket_losses = {name: {'condP': [], 'baseline': []} for name, _, _ in BUCKETS}

    print('\n' + '=' * 80)
    print('  DISTANCE-CONDITIONED LOSS EVALUATION')
    print(f'  {len(chunks)} sequences of {chunk_size} tokens')
    print('=' * 80)

    for seq_i, chunk in enumerate(chunks):
        t0 = time.time()
        condp_losses = compute_per_token_loss(condp_model, chunk, device)
        base_losses  = compute_per_token_loss(base_model,  chunk, device)
        elapsed = time.time() - t0

        # condp_losses[i] = loss at position i (predicting token i+1)
        # Bucket by position i
        for i, (cl, bl) in enumerate(zip(condp_losses, base_losses)):
            bname = bucket_index(i)
            if bname is not None:
                bucket_losses[bname]['condP'].append(cl)
                bucket_losses[bname]['baseline'].append(bl)

        condp_mean = sum(condp_losses) / len(condp_losses)
        base_mean  = sum(base_losses)  / len(base_losses)
        print(f'  Seq {seq_i+1:2d}: condP_loss={condp_mean:.4f}  '
              f'base_loss={base_mean:.4f}  delta={condp_mean-base_mean:+.4f}  '
              f'({elapsed:.1f}s)')

    # Aggregate per bucket
    print('\n' + '=' * 80)
    print('  BUCKETED LOSS BY SEQUENCE POSITION')
    print('=' * 80)
    print(f'  {"Bucket":<14} {"condP_loss":>12} {"base_loss":>12} '
          f'{"delta":>10} {"n_tokens":>10} {"condP_better":>13}')
    print(f'  {"-"*14} {"-"*12} {"-"*12} {"-"*10} {"-"*10} {"-"*13}')

    bucket_results = {}
    total_condp_loss = []
    total_base_loss  = []

    for bname, blo, bhi in BUCKETS:
        bl  = bucket_losses[bname]
        cp_list = bl['condP']
        bs_list = bl['baseline']
        n = len(cp_list)
        if n == 0:
            print(f'  {bname:<14} {"n/a":>12} {"n/a":>12} {"n/a":>10} {0:>10}')
            bucket_results[bname] = {'condP': None, 'baseline': None,
                                      'delta': None, 'n': 0}
            continue

        cp_mean = sum(cp_list) / n
        bs_mean = sum(bs_list) / n
        delta   = cp_mean - bs_mean
        n_better = sum(1 for c, b in zip(cp_list, bs_list) if c < b)
        pct_better = 100.0 * n_better / n

        total_condp_loss.extend(cp_list)
        total_base_loss.extend(bs_list)

        print(f'  {bname:<14} {cp_mean:>12.4f} {bs_mean:>12.4f} '
              f'{delta:>+10.4f} {n:>10,} {pct_better:>12.1f}%')

        bucket_results[bname] = {
            'condP_mean_loss':    cp_mean,
            'baseline_mean_loss': bs_mean,
            'delta':              delta,
            'n_tokens':           n,
            'condP_better_pct':   pct_better,
        }

    # Overall totals
    if total_condp_loss:
        overall_cp = sum(total_condp_loss) / len(total_condp_loss)
        overall_bs = sum(total_base_loss)  / len(total_base_loss)
        overall_delta = overall_cp - overall_bs
        print(f'\n  {"OVERALL":<14} {overall_cp:>12.4f} {overall_bs:>12.4f} '
              f'{overall_delta:>+10.4f} {len(total_condp_loss):>10,}')

    print('\n  Interpretation:')
    print('  - delta > 0 → condP has higher (worse) loss at that position range')
    print('  - delta < 0 → condP has lower (better) loss at that position range')
    print('  - condP_better% → fraction of individual positions where condP wins')
    print('  - If condP improves at large positions (>512) → long-range helps')
    print('  - If condP is worse at small positions (<64) → local attention gap')

    # Save
    output_data = {
        'condP_checkpoint':    args.condp_checkpoint,
        'baseline_checkpoint': args.baseline_checkpoint,
        'condP_params':        condp_params,
        'baseline_params':     base_params,
        'chunk_size':          chunk_size,
        'n_chunks':            len(chunks),
        'n_paragraphs':        len(PARAGRAPHS),
        'data_source':         'hardcoded_diverse_paragraphs',
        'bucket_results':      bucket_results,
        'overall': {
            'condP_mean_loss':    overall_cp if total_condp_loss else None,
            'baseline_mean_loss': overall_bs if total_condp_loss else None,
            'delta':              overall_delta if total_condp_loss else None,
            'n_tokens':           len(total_condp_loss),
        }
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f'\n  Results → {args.output}')


if __name__ == '__main__':
    main()
