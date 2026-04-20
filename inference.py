import torch
import torch.nn as nn
import torch.nn.functional as F
import re

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

# Mini transformer pre-trained (MiniGPT)
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

def generate(model, prompt, max_new_tokens=100):
    tokens = [word2idx.get(w, word2idx["[UNK]"]) for w in prompt.lower().split()]
    tokens = torch.tensor([tokens])

    result = prompt.split()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            inp        = tokens[:, -SEQ_LEN:]
            logits     = model(inp)
            probs      = F.softmax(logits[:, -1, :], dim=-1)
            next_tok   = torch.multinomial(probs, num_samples=1)
            result.append(idx2word[next_tok.item()])
            tokens     = torch.cat([tokens, next_tok], dim=1)

    return " ".join(result)

# Interação via terminal
print("Modelo carregado. Digite um prompt para gerar texto (ou 'sair' para encerrar):")
while True:
    prompt = input("Prompt: ")
    if prompt.lower() == 'sair':
        break
    generated = generate(model, prompt)
    print(f"Gerado: {generated}")