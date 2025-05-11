import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader, Dataset

# ----------------------------
# Positional Encoding
# ----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

# ----------------------------
# Multi-head Self Attention
# ----------------------------
class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.heads, self.head_dim).transpose(1, 2) for t in qkv]

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)

# ----------------------------
# Transformer Block
# ----------------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.drop(self.attn(self.norm1(x), mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x

# ----------------------------
# Mini Transformer Model
# ----------------------------
class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, dim, heads, ff_dim, depth, max_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.positional = PositionalEncoding(dim, max_len)
        self.encoder = nn.ModuleList([TransformerBlock(dim, heads, ff_dim) for _ in range(depth)])
        self.decoder = nn.ModuleList([TransformerBlock(dim, heads, ff_dim) for _ in range(depth)])
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, src, tgt):
        src = self.positional(self.embedding(src))
        tgt = self.positional(self.embedding(tgt))
        for layer in self.encoder:
            src = layer(src)
        for layer in self.decoder:
            tgt = layer(tgt)
        return self.output(tgt)

# ----------------------------
# Dummy Translation Dataset
# ----------------------------
word_map = {
    'PAD': 0, 'SOS': 1, 'EOS': 2,
    'one': 3, 'two': 4, 'three': 5,
    'un': 6, 'deux': 7, 'trois': 8
}
inv_word_map = {v: k for k, v in word_map.items()}

data = [
    ([3], [6]),  # one → un
    ([4], [7]),  # two → deux
    ([5], [8])   # three → trois
]

class NumberDataset(Dataset):
    def __init__(self, pairs):
        self.data = pairs

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        src, tgt = self.data[i]
        src = [1] + src + [2]  # SOS ... EOS
        tgt = [1] + tgt + [2]
        return torch.tensor(src), torch.tensor(tgt)

dataset = NumberDataset(data)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda batch: tuple(zip(*batch)))

# ----------------------------
# Train the Model
# ----------------------------
vocab_size = len(word_map)
model = MiniTransformer(vocab_size, dim=32, heads=4, ff_dim=64, depth=2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

for epoch in range(100):
    for src_batch, tgt_batch in loader:
        src_batch = nn.utils.rnn.pad_sequence(src_batch, batch_first=True)
        tgt_batch = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True)

        output = model(src_batch, tgt_batch[:, :-1])
        logits = output.reshape(-1, vocab_size)
        labels = tgt_batch[:, 1:].reshape(-1)

        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# ----------------------------
# Translate Function
# ----------------------------
def translate(word):
    model.eval()
    input_ids = torch.tensor([[1, word_map[word], 2]])
    output_ids = [1]  # start with SOS
    for _ in range(5):
        tgt_input = torch.tensor([output_ids])
        logits = model(input_ids, tgt_input)
        next_id = logits[0, -1].argmax().item()
        output_ids.append(next_id)
        if next_id == 2:  # EOS
            break
    return ' '.join(inv_word_map[i] for i in output_ids[1:-1])

print("Translate Results:")
print("one   →", translate("one"))
print("two   →", translate("two"))
print("three →", translate("three"))
