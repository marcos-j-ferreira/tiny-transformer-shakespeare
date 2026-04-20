import torch
import torch.nn as nn
import torch.nn.functional as F
import urllib.request
import re
import gradio as gr


urllib.request.urlretrieve(
 "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
 "shakespeare.txt"
)

with open("shakespeare.txt") as f:
    text = f.read()

text = text.lower()
text = re.sub(r"[^a-z\s]", "", text)
text = re.sub(r"\s+", " ", text).strip()

words = text.split()

# ─────────────────────────────────
# Vocabulary
# ─────────────────────────────────

vocab = ["[PAD]", "[UNK]"] + sorted(set(words))

word2idx = {w:i for i,w in enumerate(vocab)}
idx2word = {i:w for i,w in enumerate(vocab)}

vocab_size = len(vocab)


SEQ_LEN = 32

# Mini transformer pre-trained (MiniGPT)
class MiniGPT(nn.Module):

    def __init__(self,vocab_size,emb,heads,layers,max_len,ffn):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size,emb)
        self.pos_emb   = nn.Embedding(max_len,emb)

        layer = nn.TransformerEncoderLayer(
            d_model=emb,
            nhead=heads,
            dim_feedforward=ffn,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(layer,num_layers=layers)

        self.linear_head = nn.Linear(emb,vocab_size)

    def forward(self,tokens):

        seq_len = tokens.shape[1]

        x = self.token_emb(tokens)

        pos = torch.arange(seq_len,device=tokens.device)
        x = x + self.pos_emb(pos).unsqueeze(0)

        mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

        x = self.transformer(x,mask=mask)

        return self.linear_head(x)


# ─────────────────────────────────
# Load model
# ─────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"

model = MiniGPT(vocab_size,64,4,3,SEQ_LEN,128).to(device)

model.load_state_dict(
    torch.load("mini_shakespeare_06_Epoch_7.pth", map_location=device)
)

import time

# ─────────────────────────────────
# Text generation
# ─────────────────────────────────

def generate(prompt,max_new_tokens=20):

    tokens = [word2idx.get(w,1) for w in prompt.lower().split()]
    tokens = torch.tensor([tokens]).to(device)

    result = prompt.split()

    with torch.no_grad():

        for _ in range(max_new_tokens):

            inp = tokens[:,-SEQ_LEN:]

            logits = model(inp)

            probs = F.softmax(logits[:,-1,:],dim=-1)

            next_tok = torch.multinomial(probs,1)

            result.append(idx2word[next_tok.item()])

            tokens = torch.cat([tokens,next_tok],dim=1)

    return " ".join(result)

# ─────────────────────────────────
# Gradio interface
# ─────────────────────────────────

def run(prompt):
    return generate(prompt)

demo = gr.Interface(
    fn=run,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Textbox(label="Generated text"),
    title="tiny-transformer-shakespeare",
    description="Transformer trained on Shakespeare text"
)

demo.launch()