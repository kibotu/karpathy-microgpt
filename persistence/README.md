# v2: Persistence

Karpathy's [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) trains and infers in a single run. When the script exits, the learned weights are gone. This version splits that into two reusable steps: **train once, run many times**.

## The problem

In the original `gpt.py`, every weight lives as a `Value` object in memory. There's no serialization. Each run spends minutes training from scratch just to generate 20 names. That's fine for understanding the algorithm, but not for actually *using* the model.

## The solution

Two scripts, one file in between:

```
train.py  ──>  model.json  ──>  run.py
(minutes)      (saved weights)   (instant)
```

### `model.json` -- what gets saved

The output file contains everything needed to reconstruct the model without the training data or code that produced it:

| Section | Contents | Why it's needed |
|---|---|---|
| `config` | `n_embd`, `n_head`, `n_layer`, `block_size`, `vocab_size` | Rebuild the same architecture |
| `chars` | `["<BOS>", "a", "b", ..., "z"]` | Reconstruct the tokenizer (encode/decode text) |
| `weights` | `{"wte": [[0.01, -0.03, ...], ...], ...}` | The actual learned knowledge (~4k floats) |

This is the microgpt equivalent of a `.safetensors` or `.pt` checkpoint file. It's plain JSON -- human-readable, portable, no dependencies to load.

### `run.py` -- what's different at inference time

The inference script skips everything training-related:

- **No `Value` wrapper** -- weights are plain Python floats, not autograd nodes. No computation graph, no gradient tracking. This makes inference noticeably faster.
- **No backward pass** -- the `backward()` method doesn't exist here. We only need the forward pass.
- **No optimizer** -- no Adam, no learning rate, no momentum buffers.
- **No dataset** -- the model already learned from the data. It doesn't need to see it again.

The forward pass (`gpt()`, `linear()`, `softmax()`, `rmsnorm()`) is identical to the training version, just operating on raw floats instead of `Value` objects.

## Quick start

**Train** (takes a few minutes, produces `model.json`):

```bash
./train.sh
```

**Run** (instant, loads `model.json` and generates names):

```bash
./run.sh
```

That's the two-step workflow. Train once, run as many times as you want.

## CLI options

### train.sh

| Flag | Default | Description |
|---|---|---|
| `--steps` | `500` | Number of training steps |
| `--output` | `model.json` | Where to save the trained model |

```bash
./train.sh --steps 1000 --output my_model.json
```

### run.sh

| Flag | Default | Description |
|---|---|---|
| `--model` | `model.json` | Path to the saved model |
| `--samples` | `20` | Number of names to generate |
| `--temperature` | `0.6` | Creativity control (0.1 = conservative, 1.0 = wild) |

```bash
./run.sh --model my_model.json --samples 50 --temperature 0.8
```

## What this teaches you

This is the exact same pattern used by every real ML system, just stripped to the bone:

1. **Training produces an artifact** -- in PyTorch it's a `.pt` file, in llama.cpp it's a `.gguf`, here it's a `model.json`. The format doesn't matter. The idea is the same: serialize the learned floats to disk.

2. **Inference doesn't need autograd** -- during training, every operation builds a graph so gradients can flow backward. During inference, you're just doing matrix math. Dropping the graph overhead is why frameworks have a `torch.no_grad()` / `model.eval()` mode.

3. **The config must travel with the weights** -- if you save weights but forget the hyperparameters, you can't reconstruct the model. You won't know the embedding dimension or number of heads. That's why checkpoint formats always bundle metadata alongside the raw numbers.

4. **The tokenizer must travel with the weights** -- the model learned to associate token ID 5 with the letter "e". If you load the weights with a different tokenizer mapping, everything breaks. Tokenizer and weights are inseparable.
