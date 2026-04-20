# Tiny Transformer Shakespeare

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/marcos-j-ferreira/tiny-transformer-shakespeare)

A small Transformer model trained on Shakespeare's text using PyTorch. This project demonstrates how to build and train a basic language model for text generation.

## About the Transformer

The Transformer is a neural network architecture based on self-attention mechanisms, introduced in the paper "Attention is All You Need" (2017). It excels at handling sequential data like text by allowing the model to focus on relevant parts of the input.

### How It Works (Simplified)

1. **Tokenization**: Text is split into tokens (words or subwords).
2. **Embeddings**: Tokens are converted to vectors.
3. **Self-Attention**: Each token attends to others to capture context.
4. **Feed-Forward**: Non-linear transformations are applied.
5. **Output**: Probabilities for next tokens are generated.

ASCII Illustration of Self-Attention:

```
Input Sequence: [The, cat, sat, on, the, mat]

Self-Attention Matrix:
  The  cat  sat  on   the  mat
The 1.0  0.8  0.5  0.3  0.9  0.4
cat  0.8  1.0  0.9  0.6  0.7  0.5
sat  0.5  0.9  1.0  0.8  0.6  0.7
on   0.3  0.6  0.8  1.0  0.4  0.9
the  0.9  0.7  0.6  0.4  1.0  0.8
mat  0.4  0.5  0.7  0.9  0.8  1.0

(Values represent attention weights; higher means more focus)
```

This project uses a simplified Transformer encoder for next-word prediction.

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Run inference: `python inference.py`
3. Enter prompts in the terminal to generate text.

## Model Details

- **Architecture**: 3 layers, 4 attention heads, embedding dim 64, FFN dim 128.
- **Dataset**: Tiny Shakespeare (~202k words, vocab size ~12k).
- **Training**: Stopped at Epoch 7, Batch 1000/6331, Loss: 3.6487.
- **Purpose**: Generates Shakespeare-like text continuations.

## Test the Model

Try the model online at [Hugging Face Spaces](https://huggingface.co/spaces/marcos-j-ferreira/tiny-transformer-shakespeare).
