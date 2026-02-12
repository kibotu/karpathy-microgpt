# microgpt

A GPT trained and run in **243 lines of pure, dependency-free Python**. No PyTorch. No NumPy. No nothing. Just `import math` and sheer will.

Based on [Andrej Karpathy](https://x.com/karpathy)'s [microgpt](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) -- the most distilled version of the GPT algorithm that can exist.

> *"This is the full algorithmic content of what is needed. Everything else is just for efficiency. I cannot simplify this any further."*
> -- Andrej Karpathy

## Who is Karpathy?

Andrej Karpathy is one of the most influential figures in modern AI. He was the founding member of OpenAI, led Tesla's Autopilot vision team, and has an extraordinary talent for making deep learning accessible. His [YouTube lectures](https://www.youtube.com/@AndrejKarpathy), blog posts, and projects like [nanoGPT](https://github.com/karpathy/nanoGPT) and [micrograd](https://github.com/karpathy/micrograd) have taught more people how neural networks actually work than most university courses combined.

## Quick start

```bash
./train.sh
```

That's it. The script installs [uv](https://docs.astral.sh/uv/) if needed and runs `gpt.py`. Takes a few minutes on a modern machine -- no GPU required.

## What does it actually do?

Here's a step-by-step walkthrough of the entire script. Read top to bottom -- that's the order it executes.

### 1. Load the dataset

Downloads ~32k human names from [makemore](https://github.com/karpathy/makemore) (e.g. "emma", "olivia", "ava"). These are the "documents" the model learns from. Think of it as the smallest possible text corpus that still produces recognizable output.

### 2. Build a tokenizer

Creates a character-level tokenizer: each unique character (a-z) gets an integer ID, plus a special `<BOS>` (Beginning of Sequence) token. Vocab size ends up at 27. This is the alphabet the model sees -- no subwords, no BPE, just raw characters.

### 3. Implement autograd from scratch

The `Value` class is a tiny automatic differentiation engine (a descendant of Karpathy's [micrograd](https://github.com/karpathy/micrograd)). Every scalar is wrapped in a `Value` that tracks:

- The **forward pass**: compute the result of each operation (`+`, `*`, `**`, `exp`, `log`, `relu`)
- The **backward pass**: compute gradients via the chain rule by walking the computation graph in reverse topological order

This is the same idea behind PyTorch's `autograd` -- just without the C++ backend, CUDA kernels, and 2M lines of code.

### 4. Initialize model parameters

Sets up a GPT-2-style architecture in miniature:

| Hyperparameter | Value |
|---|---|
| Embedding dim | 16 |
| Attention heads | 4 |
| Layers | 1 |
| Context length | 8 |
| **Total params** | **~4k** |

The weight matrices (token embeddings, position embeddings, attention Q/K/V/O projections, MLP layers, output head) are initialized with small random Gaussian values. Every single weight is a `Value` node in the autograd graph.

### 5. Define the forward pass

The `gpt()` function implements one token step of a GPT-2 variant:

1. **Embed** -- look up token + position embeddings, add them together
2. **RMSNorm** -- normalize the vector (like LayerNorm but simpler, no bias/shift)
3. **Multi-head self-attention** -- each of the 4 heads computes Q/K/V projections, dot-product attention scores, softmax weights, and a weighted sum of values. Uses a KV cache for efficiency across positions
4. **Residual connection** -- add the attention output back to the input
5. **MLP** -- two linear layers with a squared ReLU activation (`ReLU(x)^2`) in between
6. **Another residual connection** -- add MLP output back
7. **Output projection** -- linear layer mapping to vocab-sized logits

Key differences from vanilla GPT-2: RMSNorm instead of LayerNorm, no biases anywhere, ReLU^2 instead of GeLU.

### 6. Train with Adam

For 500 steps, the training loop:

1. **Picks a name** from the dataset
2. **Tokenizes it** with `<BOS>` delimiters on both ends
3. **Runs the forward pass** token by token, building a computation graph all the way to the cross-entropy loss
4. **Runs the backward pass** (`loss.backward()`) to compute gradients for all ~4k parameters via reverse-mode autodiff
5. **Updates weights** using the Adam optimizer with linear learning rate decay

Each step processes a single name. Loss starts around ~3.3 (random guessing over 27 chars) and drops to ~2.0.

### 7. Generate names

After training, the model generates 20 new names by:

1. Starting with the `<BOS>` token
2. Running the forward pass to get a probability distribution over the next character
3. Sampling from that distribution (with `temperature=0.6` for controlled randomness)
4. Repeating until `<BOS>` is predicted again (end of name) or max length is reached

You'll see output like: `lellen`, `aman`, `lela`, `kan` -- not real names, but they *look* like they could be.

## What to try next

- **Crank up `num_steps`** beyond 500 -- watch the loss drop and names get more realistic
- **Adjust `temperature`** at inference -- lower = more conservative, higher = more creative
- **Add layers** (`n_layer = 2`) or embedding size (`n_embd = 32`) -- more params, better model, slower training
- **Swap the dataset** -- replace `input.txt` with any line-delimited text (city names, Pokémon, words from any language)
- **Use PyPy for speed** -- `uv run --python pypy gpt.py` gives a significant speedup thanks to better JIT and garbage collection ([credit](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95?permalink_comment_id=5427633#gistcomment-5427633))
- **Read the code** -- it's 243 lines, fully commented, and genuinely beautiful. There is no better way to understand how GPT works at a fundamental level

## Wall art

Karpathy also published the entire algorithm as a single beautifully typeset page at [karpathy.ai/microgpt.html](https://karpathy.ai/microgpt.html). Print it, frame it, hang it next to your monitor. It might be the most mass-appealing piece of wall art that is also a fully functional neural network.

## Source

[karpathy/microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95) -- Gist by Andrej Karpathy

## License

The original `gpt.py` is by [Andrej Karpathy](https://x.com/karpathy/status/2021694437152157847). The `train.sh` wrapper and this README are provided as-is.
