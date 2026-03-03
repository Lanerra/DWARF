#!/usr/bin/env python3
"""
Download and cache NLP benchmark datasets for DWARF external evaluation.

Downloads HellaSwag, PIQA, ARC-Easy, ARC-Challenge, and Winogrande from
HuggingFace and saves them as local JSON cache files. Run this once before
running eval_external.py.

Usage:
  .venv/bin/python3 benchmarks/download_benchmarks.py

Cache location: benchmarks/logs/benchmark_cache/
"""

import json, os, sys

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'logs', 'benchmark_cache')

BENCHMARKS = {
    'hellaswag': {
        'dataset': 'Rowan/hellaswag',
        'split': 'validation',
        'config': None,
        'n_choices': 4,
        'desc': 'HellaSwag — commonsense NLI completion (4-way MC)',
    },
    'piqa': {
        'dataset': 'piqa',
        'split': 'validation',
        'config': None,
        'n_choices': 2,
        'desc': 'PIQA — physical intuition (binary MC)',
    },
    'arc_easy': {
        'dataset': 'allenai/ai2_arc',
        'split': 'validation',
        'config': 'ARC-Easy',
        'n_choices': 4,
        'desc': 'ARC-Easy — elementary science QA (4-way MC)',
    },
    'arc_challenge': {
        'dataset': 'allenai/ai2_arc',
        'split': 'validation',
        'config': 'ARC-Challenge',
        'n_choices': 4,
        'desc': 'ARC-Challenge — harder science QA (4-way MC)',
    },
    'winogrande': {
        'dataset': 'winogrande',
        'split': 'validation',
        'config': 'winogrande_xl',
        'n_choices': 2,
        'desc': 'Winogrande — coreference resolution (binary MC)',
    },
    'lambada': {
        'dataset': 'EleutherAI/lambada_openai',
        'split': 'test',
        'config': None,
        'n_choices': None,  # generative / last-word prediction
        'desc': 'LAMBADA — last-word prediction, long-range context',
    },
}


def normalize_hellaswag(examples):
    """Convert HellaSwag to standard format."""
    out = []
    for ex in examples:
        ctx = ex['ctx']
        choices = ex['endings']
        label = int(ex['label'])
        out.append({'context': ctx, 'choices': choices, 'label': label})
    return out


def normalize_piqa(examples):
    out = []
    for ex in examples:
        ctx = ex['goal']
        choices = [ex['sol1'], ex['sol2']]
        label = int(ex['label'])
        out.append({'context': ctx, 'choices': choices, 'label': label})
    return out


def normalize_arc(examples):
    out = []
    for ex in examples:
        ctx = ex['question']
        # ARC choices come as dict with 'text' list and 'label' list
        choice_dict = ex['choices']
        labels_list = choice_dict['label']
        text_list   = choice_dict['text']
        # Answer key is like 'A', 'B', 'C', 'D' — map to index
        answer_key  = ex['answerKey']
        try:
            label = labels_list.index(answer_key)
        except ValueError:
            # Sometimes numeric: '1','2','3','4' → 0-indexed
            try:
                label = int(answer_key) - 1
            except ValueError:
                continue  # skip malformed
        out.append({'context': ctx, 'choices': text_list, 'label': label})
    return out


def normalize_winogrande(examples):
    out = []
    for ex in examples:
        ctx     = ex['sentence']
        choices = [ex['option1'], ex['option2']]
        label   = int(ex['answer']) - 1  # Winogrande uses '1'/'2'
        out.append({'context': ctx, 'choices': choices, 'label': label})
    return out


def normalize_lambada(examples):
    out = []
    for ex in examples:
        text = ex['text']
        # Split off last word as the target
        words = text.rsplit(' ', 1)
        if len(words) < 2: continue
        context, target = words[0], words[1]
        out.append({'context': context, 'target': target, 'full_text': text})
    return out


NORMALIZERS = {
    'hellaswag':    normalize_hellaswag,
    'piqa':         normalize_piqa,
    'arc_easy':     normalize_arc,
    'arc_challenge': normalize_arc,
    'winogrande':   normalize_winogrande,
    'lambada':      normalize_lambada,
}


def download_all():
    try:
        from datasets import load_dataset
    except ImportError:
        print('ERROR: datasets library not found. Run: pip install datasets')
        sys.exit(1)

    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f'Cache directory: {CACHE_DIR}\n')

    results = {}
    for name, cfg in BENCHMARKS.items():
        cache_file = os.path.join(CACHE_DIR, f'{name}.json')

        if os.path.exists(cache_file):
            with open(cache_file) as f:
                cached = json.load(f)
            print(f'  [{name}] Already cached — {len(cached):,} examples')
            results[name] = len(cached)
            continue

        print(f'  [{name}] {cfg["desc"]}')
        print(f'           Downloading {cfg["dataset"]} ({cfg["split"]})...')
        try:
            if cfg['config']:
                ds = load_dataset(cfg['dataset'], cfg['config'],
                                  split=cfg['split'])
            else:
                ds = load_dataset(cfg['dataset'], split=cfg['split'])
        except Exception as e:
            print(f'           WARNING: Failed — {e}')
            results[name] = 0
            continue

        examples = list(ds)
        normalizer = NORMALIZERS[name]
        normalized = normalizer(examples)

        with open(cache_file, 'w') as f:
            json.dump(normalized, f)

        print(f'           Saved {len(normalized):,} examples → {cache_file}')
        results[name] = len(normalized)

    print('\n── Download summary ──────────────────────────────────────────')
    for name, n in results.items():
        status = f'{n:,} examples' if n > 0 else 'FAILED'
        print(f'  {name:<20} {status}')

    print('\nDone. Run eval_external.py to evaluate a checkpoint.')
    return results


if __name__ == '__main__':
    download_all()
