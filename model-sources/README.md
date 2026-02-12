# Model Sources

Tools for downloading training data.

## Wikipedia Dump

Downloads English Wikipedia for GPT training.

```bash
# Show help
uv run download_wikipedia.py --help

# Download full articles (~25GB compressed)
uv run download_wikipedia.py

# Download titles only (~100MB, good for testing)
uv run download_wikipedia.py --type titles

# Download abstracts (~2GB)
uv run download_wikipedia.py --type abstracts

# Custom output directory
uv run download_wikipedia.py --output ../data/wikipedia

# List available dump dates
uv run download_wikipedia.py --list-dates
```

### Dump Types

| Type | Size | Description |
|------|------|-------------|
| `articles` | ~25GB | Full article text (default) |
| `abstracts` | ~2GB | Page summaries |
| `titles` | ~100MB | Article titles only |
| `meta-current` | ~35GB | All pages including talk |

### Notes

- Downloads resume automatically if interrupted
- MD5 checksums are verified by default
- Extracts to ~100GB for the full articles dump
