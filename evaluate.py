import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import matplotlib.pyplot as plt

# Carregar o texto do dataset
with open("dataset/input.txt", encoding="utf-8") as f:
    text = f.read()

# Limpeza mínima — mantém a estrutura do texto
text = text.lower()
text = re.sub(r"[^a-z\s]", "", text)   # só letras e espaços
text = re.sub(r"\s+", " ", text).strip()

words = text.split()
print(f"Total de palavras: {len(words):,}")

# Vocab
vocab      = ["[PAD]", "[UNK]"] + sorted(set(words))
word2idx   = {w: i for i, w in enumerate(vocab)}
idx2word   = {i: w for i, w in enumerate(vocab)}
vocab_size = len(vocab)

print(f"Vocab size: {vocab_size:,}")

# Parâmetros do modelo
SEQ_LEN = 32
embedding_dim = 64
num_heads = 4
num_layers = 3
ffn_dim = 128

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads,
                 num_layers, max_seq_len, ffn_dim):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embedding_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, embedding_dim)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=num_heads,
            dim_feedforward=ffn_dim, dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.linear_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, tokens):
        seq_len = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos_emb(
            torch.arange(seq_len, device=tokens.device)
        ).unsqueeze(0)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(tokens.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        return self.linear_head(x)

# Instanciar o modelo
model = MiniGPT(
    vocab_size    = vocab_size,
    embedding_dim = embedding_dim,
    num_heads     = num_heads,
    num_layers    = num_layers,
    max_seq_len   = SEQ_LEN,
    ffn_dim       = ffn_dim,
)

# Carregar o modelo treinado
model.load_state_dict(torch.load("models/mini_shakespeare_06_Epoch_7.pth"))
model.eval()

# Processar tokens
all_tokens = [word2idx.get(w, word2idx["[UNK]"]) for w in words]
all_tokens = torch.tensor(all_tokens, dtype=torch.long)

# Dividir em train e val (últimos 10% para val)
train_size = int(0.9 * len(all_tokens))
train_tokens = all_tokens[:train_size]
val_tokens = all_tokens[train_size:]

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")

# Dataset para val
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self):
        return len(self.tokens) - self.seq_len

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_len]
        y = self.tokens[idx + 1 : idx + self.seq_len + 1]
        return x, y

val_dataset = TextDataset(val_tokens, SEQ_LEN)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)

# Avaliar perplexity
model.eval()
total_loss = 0
total_tokens = 0

with torch.no_grad():
    for x, y in val_loader:
        logits = model(x)
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),
            y.view(-1),
            reduction='sum'
        )
        total_loss += loss.item()
        total_tokens += y.numel()

avg_loss = total_loss / total_tokens
perplexity = torch.exp(torch.tensor(avg_loss))

print(f"Average Loss on Val: {avg_loss:.4f}")
print(f"Perplexity: {perplexity:.2f}")

# Gerar texto
def generate(model, prompt, max_new_tokens=20):
    tokens = [word2idx.get(w, word2idx["[UNK]"]) for w in prompt.lower().split()]
    tokens = torch.tensor([tokens])

    result = prompt.split()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp = tokens[:, -SEQ_LEN:]
            logits = model(inp)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            result.append(idx2word[next_tok.item()])
            tokens = torch.cat([tokens, next_tok], dim=1)

    return " ".join(result)

print("\nGenerated Samples:")
prompts = ["To be or", "All the world's", "Romeo and"]
for p in prompts:
    gen = generate(model, p, max_new_tokens=10)
    print(f"Prompt: '{p}' -> '{gen}'")

# Gerar gráfico: top 10 probabilidades para um prompt
prompt = "To be"
tokens = [word2idx.get(w, word2idx["[UNK]"]) for w in prompt.lower().split()]
tokens = torch.tensor([tokens])

with torch.no_grad():
    logits = model(tokens)
    probs = F.softmax(logits[:, -1, :], dim=-1).squeeze()

# Top 10
top_probs, top_indices = torch.topk(probs, 10)
top_words = [idx2word[i.item()] for i in top_indices]

plt.figure(figsize=(10, 6))
plt.barh(top_words[::-1], top_probs.numpy()[::-1])
plt.xlabel('Probability')
plt.title(f'Top 10 Predicted Words for Prompt: "{prompt}"')
plt.tight_layout()
plt.savefig('evaluation_plot.png')
print("\nSaved plot to evaluation_plot.png")
