"""
Load a trained GPT from disk and generate text. No training, no gradients --
just pure forward-pass inference using the saved weights.

Usage: uv run run.py [--model model.json] [--samples 20] [--temperature 0.6]
"""

import sys
import math
import json
import random

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------
model_path = 'model.json'
num_samples = 20
temperature = 0.6

args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == '--model' and i + 1 < len(args):
        model_path = args[i + 1]; i += 2
    elif args[i] == '--samples' and i + 1 < len(args):
        num_samples = int(args[i + 1]); i += 2
    elif args[i] == '--temperature' and i + 1 < len(args):
        temperature = float(args[i + 1]); i += 2
    else:
        print(f"Unknown arg: {args[i]}"); sys.exit(1)

# ---------------------------------------------------------------------------
# Load model from disk
# ---------------------------------------------------------------------------
print(f"Loading model from {model_path}...")

with open(model_path, 'r') as f:
    model = json.load(f)

config = model['config']
chars = model['chars']
weights = model['weights']

n_embd = config['n_embd']
n_head = config['n_head']
n_layer = config['n_layer']
block_size = config['block_size']
vocab_size = config['vocab_size']
head_dim = n_embd // n_head

# Reconstruct tokenizer
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
BOS = stoi['<BOS>']

# Reconstruct state_dict as plain float lists (no Value wrapper needed -- no gradients during inference)
state_dict = {k: [list(row) for row in mat] for k, mat in weights.items()}

num_params = sum(len(row) for mat in state_dict.values() for row in mat)
print(f"Loaded: {vocab_size} tokens, {num_params} params, context length {block_size}")

# ---------------------------------------------------------------------------
# Model architecture (inference-only, plain floats -- no autograd overhead)
# ---------------------------------------------------------------------------
def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(v - max_val) for v in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)

    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]]
            v_h = [vi[hs:hs+head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [max(0, xi) ** 2 for xi in x]  # ReLU^2 on plain floats
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]

    logits = linear(x, state_dict['lm_head'])
    return logits

# ---------------------------------------------------------------------------
# Generate samples
# ---------------------------------------------------------------------------
print(f"\n--- inference (temperature={temperature}) ---")
for sample_idx in range(num_samples):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    result = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=probs)[0]
        if token_id == BOS:
            break
        result.append(itos[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(result)}")
