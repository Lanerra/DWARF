"""
Dispersion & Phase Alignment Analysis for Triadic DSQG Relay
=============================================================
Uses existing relay_analysis JSON files (no GPU needed).
Computes:
  1. Per-group (A/B/C) signal contribution (delta cosine sim per layer)
  2. DFT of signal flow curve — dominant spatial frequencies
  3. Phase alignment between groups — constructive vs destructive
  4. Dispersion relation — group velocity per triad group
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(ROOT, 'tools', 'viz_output', 'triadic_l25')
os.makedirs(OUT, exist_ok=True)

RELAY_D512  = os.path.join(OUT, 'relay_d512',  'relay_triadic_l25_d768',
                            'relay_analysis_triadic_l25_d768_d512.json')
RELAY_D1536 = os.path.join(OUT, 'relay_d1536', 'relay_triadic_l25_d768',
                            'relay_analysis_triadic_l25_d768_d1536.json')

# ── load ───────────────────────────────────────────────────────────────────
with open(RELAY_D512)  as f: r512  = json.load(f)
with open(RELAY_D1536) as f: r1536 = json.load(f)

def extract_sf(relay):
    sf = relay['signal_flow']['layers']
    layers  = [e['layer_idx']   for e in sf]
    cos     = [e['cos_sim']     for e in sf]
    is_fa   = [e['is_full_attn'] for e in sf]
    return layers, np.array(cos), is_fa

layers, cos512,  fa_flags = extract_sf(r512)
_,      cos1536, _        = extract_sf(r1536)

FA_LAYER = next(layers[i] for i, f in enumerate(fa_flags) if f)   # 6
print(f'FA layer: {FA_LAYER}')

# ── group assignment ────────────────────────────────────────────────────────
GROUP_COLORS = {'embed': '#888', 'FA': '#00ffff', 'A': '#ff6b35', 'B': '#4ecdc4', 'C': '#ffe66d'}
GROUP_LABELS = {'A': 'GROUP_A (local δ≤28)', 'B': 'GROUP_B (mid δ≤480)', 'C': 'GROUP_C (long δ≥480)'}

def layer_group(idx):
    if idx == -1:              return 'embed'
    if idx == FA_LAYER:        return 'FA'
    if idx < FA_LAYER:         return ['A','B','C'][idx % 3]
    return ['A','B','C'][(idx - FA_LAYER - 1) % 3]

groups = [layer_group(l) for l in layers]

# ── 1. per-group delta contributions ───────────────────────────────────────
# delta[i] = signal arriving AT layer i (change from previous layer)
deltas512  = np.diff(cos512)
deltas1536 = np.diff(cos1536)
delta_groups = groups[1:]   # delta[i] corresponds to groups[i+1]
delta_layers = layers[1:]

gd = {g: {'layers':[], 'd512':[], 'd1536':[]} for g in ['A','B','C','FA']}
for i, g in enumerate(delta_groups):
    if g in gd:
        gd[g]['layers'].append(delta_layers[i])
        gd[g]['d512'].append(deltas512[i])
        gd[g]['d1536'].append(deltas1536[i])

# ── 2. DFT of signal flow ──────────────────────────────────────────────────
# Use layers 0..24 only (skip embed at -1)
body_idx = [i for i, l in enumerate(layers) if l >= 0]
body_layers = np.array([layers[i] for i in body_idx])   # 0..24
c512_body   = np.array([cos512[i]  for i in body_idx])
c1536_body  = np.array([cos1536[i] for i in body_idx])

n = len(body_layers)
freqs = np.fft.rfftfreq(n)   # cycles per layer-step
period_labels = {f: f'{1/f:.1f}' if f > 0 else '∞' for f in freqs}

fft512  = np.abs(np.fft.rfft(c512_body))
fft1536 = np.abs(np.fft.rfft(c1536_body))

# Detrend first (remove linear trend so DFT reflects oscillation not ramp)
trend512  = np.polyval(np.polyfit(body_layers, c512_body,  1), body_layers)
trend1536 = np.polyval(np.polyfit(body_layers, c1536_body, 1), body_layers)
osc512    = c512_body  - trend512
osc1536   = c1536_body - trend1536
fft512_dt  = np.abs(np.fft.rfft(osc512))
fft1536_dt = np.abs(np.fft.rfft(osc1536))

top_freqs = np.argsort(fft512_dt)[::-1][:5]
print('\nTop 5 spatial frequencies (detrended, d=512):')
for k in top_freqs:
    period = 1/freqs[k] if freqs[k] > 0 else float('inf')
    print(f'  freq={freqs[k]:.4f} (period={period:.1f} layers), amp={fft512_dt[k]:.6f}')

# ── 3. phase alignment ─────────────────────────────────────────────────────
# For each group, compute: mean layer of positive contribution (weighted by magnitude)
# and measure cross-correlation between group A and B, B and C contributions
# to assess constructive vs destructive interference.

def group_signal_array(g, use_1536=False):
    """Full 25-element array with group contribution only (0 elsewhere)."""
    out = np.zeros(n)
    key = 'd1536' if use_1536 else 'd512'
    for i, (lyr, delta) in enumerate(zip(gd[g]['layers'], gd[g][key])):
        if 0 <= lyr < n:
            out[lyr] = delta
    return out

ga512  = group_signal_array('A', False)
gb512  = group_signal_array('B', False)
gc512  = group_signal_array('C', False)
ga1536 = group_signal_array('A', True)
gb1536 = group_signal_array('B', True)
gc1536 = group_signal_array('C', True)

def xcorr(a, b):
    """Normalized cross-correlation, return peak lag and value."""
    cor = np.correlate(a - a.mean(), b - b.mean(), mode='full')
    if cor.std() < 1e-10: return 0, 0.0
    cor_norm = cor / (len(a) * a.std() * b.std() + 1e-10)
    lags = np.arange(-(len(a)-1), len(a))
    peak_lag = lags[np.argmax(np.abs(cor_norm))]
    peak_val = cor_norm[np.argmax(np.abs(cor_norm))]
    return peak_lag, peak_val

print('\nPhase alignment (d=512):')
for g1, g2, a1, a2 in [('A','B', ga512, gb512), ('B','C', gb512, gc512), ('A','C', ga512, gc512)]:
    lag, val = xcorr(a1, a2)
    print(f'  {g1}↔{g2}: peak_lag={lag:+d} layers, corr={val:+.3f} '
          f'({"constructive" if val > 0 else "destructive"})')

print('\nPhase alignment (d=1536):')
for g1, g2, a1, a2 in [('A','B', ga1536, gb1536), ('B','C', gb1536, gc1536), ('A','C', ga1536, gc1536)]:
    lag, val = xcorr(a1, a2)
    print(f'  {g1}↔{g2}: peak_lag={lag:+d} layers, corr={val:+.3f} '
          f'({"constructive" if val > 0 else "destructive"})')

# Group velocity: rate of cumulative signal accumulation per group
def cumulative_group(g, use_1536=False):
    key = 'd1536' if use_1536 else 'd512'
    pos_deltas  = [(l, d) for l, d in zip(gd[g]['layers'], gd[g][key]) if d > 0]
    if not pos_deltas: return None, None
    lyrs, vals = zip(*pos_deltas)
    weighted_layer = np.average(lyrs, weights=vals)
    total_gain = sum(vals)
    return weighted_layer, total_gain

print('\nGroup velocities (weighted mean layer of positive contribution):')
for d_label, use1536 in [('d=512', False), ('d=1536', True)]:
    print(f'  {d_label}:')
    for g in ['A', 'B', 'C']:
        wl, gain = cumulative_group(g, use1536)
        if wl is not None:
            print(f'    GROUP_{g}: mean_layer={wl:.1f}, total_gain={gain:.4f}')

# ── 4. PLOT ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 14))
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# -- row 0: raw signal flow + per-group deltas
ax_flow   = fig.add_subplot(gs[0, :2])
ax_deltas = fig.add_subplot(gs[0, 2])

for cos, lbl, ls in [(cos512,  'd=512',  '-'), (cos1536, 'd=1536', '--')]:
    ax_flow.plot(layers, cos, ls, lw=2, label=lbl, color='white' if lbl=='d=512' else '#aaa')

# shade groups
for i, (l, g) in enumerate(zip(layers, groups)):
    if g in GROUP_COLORS and l >= 0:
        ax_flow.axvspan(l - 0.5, l + 0.5, alpha=0.15, color=GROUP_COLORS[g], zorder=0)

ax_flow.axvline(FA_LAYER, color=GROUP_COLORS['FA'], lw=1.5, ls=':', label=f'FA@L{FA_LAYER}')
ax_flow.set_xlabel('Layer', fontsize=10); ax_flow.set_ylabel('Cosine similarity (passkey↔cue)', fontsize=9)
ax_flow.set_title('Relay Signal Flow — shading by group', fontsize=11)
ax_flow.legend(fontsize=8); ax_flow.set_facecolor('#1a1a2e')

# stacked bar of per-layer deltas
bar_layers = delta_layers
b_colors   = [GROUP_COLORS.get(g, '#555') for g in delta_groups]
bars = ax_deltas.bar(bar_layers, deltas512, color=b_colors, edgecolor='none', alpha=0.85)
ax_deltas.axhline(0, color='white', lw=0.5, ls='--')
ax_deltas.axvline(FA_LAYER, color=GROUP_COLORS['FA'], lw=1.2, ls=':')
ax_deltas.set_xlabel('Layer', fontsize=9); ax_deltas.set_ylabel('Δ cosine sim', fontsize=9)
ax_deltas.set_title('Per-layer signal contribution (d=512)', fontsize=10)
ax_deltas.set_facecolor('#1a1a2e')
# legend patches
from matplotlib.patches import Patch
ax_deltas.legend(handles=[Patch(color=GROUP_COLORS[g], label=f'Grp {g}') for g in ['A','B','C','FA']],
                 fontsize=7, loc='upper left')

# -- row 1: DFT amplitude spectra (detrended)
ax_fft_all  = fig.add_subplot(gs[1, :2])
ax_fft_zoom = fig.add_subplot(gs[1, 2])

periods = np.where(freqs > 0, 1/freqs, np.inf)
ax_fft_all.plot(freqs, fft512_dt,  lw=2, color='#ff6b35', label='d=512')
ax_fft_all.plot(freqs, fft1536_dt, lw=2, color='#4ecdc4', label='d=1536', ls='--')
ax_fft_all.axvline(1/3, color='#ffe66d', lw=1.5, ls='--', alpha=0.8, label='f=1/3 (period=3 layers)')
ax_fft_all.axvline(1/8, color='#888',    lw=1,   ls=':',  alpha=0.6, label='f=1/8 (Phase2 ~8L ramp)')
ax_fft_all.set_xlabel('Spatial frequency (cycles / layer)', fontsize=10)
ax_fft_all.set_ylabel('DFT amplitude', fontsize=10)
ax_fft_all.set_title('DFT of signal flow (detrended) — spatial frequency content', fontsize=11)
ax_fft_all.legend(fontsize=8); ax_fft_all.set_facecolor('#1a1a2e')

# zoom: just the low-freq end
mask = freqs < 0.2
ax_fft_zoom.plot(freqs[mask], fft512_dt[mask],  lw=2, color='#ff6b35', label='d=512')
ax_fft_zoom.plot(freqs[mask], fft1536_dt[mask], lw=2, color='#4ecdc4', label='d=1536', ls='--')
ax_fft_zoom.axvline(1/3, color='#ffe66d', lw=1.5, ls='--', alpha=0.8, label='1/3')
ax_fft_zoom.set_xlabel('Frequency (low end)', fontsize=9)
ax_fft_zoom.set_title('Zoom: low-freq (long-period) content', fontsize=10)
ax_fft_zoom.legend(fontsize=7); ax_fft_zoom.set_facecolor('#1a1a2e')
# annotate peak
pk = np.argmax(fft512_dt[mask])
ax_fft_zoom.annotate(f'peak\nf={freqs[mask][pk]:.3f}\n(T={1/freqs[mask][pk]:.0f}L)' if freqs[mask][pk]>0 else '',
                     xy=(freqs[mask][pk], fft512_dt[mask][pk]),
                     xytext=(freqs[mask][pk]+0.02, fft512_dt[mask][pk]*0.8),
                     fontsize=7, color='white', arrowprops=dict(arrowstyle='->', color='white', lw=0.8))

# -- row 2: per-group cumulative signal + phase cross-correlations
ax_cumA = fig.add_subplot(gs[2, 0])
ax_cumB = fig.add_subplot(gs[2, 1])
ax_xcorr = fig.add_subplot(gs[2, 2])

for g, ax_c, col in [('A', ax_cumA, '#ff6b35'), ('B', ax_cumB, '#4ecdc4')]:
    sig512  = group_signal_array(g, False)
    sig1536 = group_signal_array(g, True)
    ax_c.bar(body_layers, sig512,  color=col, alpha=0.7, label='d=512')
    ax_c.bar(body_layers, sig1536, color=col, alpha=0.35, label='d=1536', hatch='//')
    ax_c.axhline(0, color='white', lw=0.5); ax_c.axvline(FA_LAYER, color='cyan', lw=1, ls=':')
    ax_c.set_title(f'GROUP_{g} signal contribution', fontsize=10)
    ax_c.set_xlabel('Layer', fontsize=9); ax_c.set_facecolor('#1a1a2e')
    ax_c.legend(fontsize=7)

# cross-correlation curves
lags = np.arange(-(n-1), n)
pairs = [('A','B', ga512, gb512, '#ff6b35'), ('B','C', gb512, gc512, '#4ecdc4'), ('A','C', ga512, gc512, '#ffe66d')]
for g1, g2, a1, a2, col in pairs:
    cor = np.correlate(a1 - a1.mean(), a2 - a2.mean(), mode='full')
    norm = len(a1) * (a1.std() * a2.std() + 1e-10)
    ax_xcorr.plot(lags, cor/norm, lw=1.5, color=col, label=f'{g1}↔{g2}')
ax_xcorr.axvline(0, color='white', lw=0.5, ls='--')
ax_xcorr.axvline(1, color='#888',  lw=0.5, ls=':')
ax_xcorr.axvline(-1, color='#888', lw=0.5, ls=':')
ax_xcorr.set_xlim(-8, 8); ax_xcorr.set_xlabel('Lag (layers)', fontsize=9)
ax_xcorr.set_ylabel('Normalized cross-correlation', fontsize=9)
ax_xcorr.set_title('Phase alignment: group cross-correlations', fontsize=10)
ax_xcorr.legend(fontsize=7); ax_xcorr.set_facecolor('#1a1a2e')

fig.patch.set_facecolor('#0d1117')
for ax in fig.get_axes():
    ax.tick_params(colors='#ccc', labelsize=8)
    ax.xaxis.label.set_color('#ccc'); ax.yaxis.label.set_color('#ccc')
    ax.title.set_color('white')
    for spine in ax.spines.values(): spine.set_color('#444')

out_path = os.path.join(OUT, 'dispersion_phase_analysis.jpg')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f'\nSaved: {out_path}')
