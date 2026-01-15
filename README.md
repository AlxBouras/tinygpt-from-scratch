# tinygpt-from-scratch
A minimal, readable implementation of a Transformer-based GPT-style language model.

The goal is to implement the core components of a decoder-only Transformer
and train it end-to-end on a small text corpus, with as little abstraction
as possible.

This project uses a basic character-level tokenizer. This tokenizer is intentionally simple to make the data pipeline and model behavior fully transparent.

The vocabulary is constructed by extracting all unique characters present in the training text and assigning each character a unique integer ID. No subword units, byte-pair encoding, or external tokenization libraries are used.


## Scope

**This project focuses on:**
- A clean implementation of a decoder-only Transformer
- Causal self-attention
- Token and positional embeddings
- Autoregressive language modeling
- Training and text generation loops