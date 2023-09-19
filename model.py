import torch
import torch.nn as nn

# Hyperparameters
vocab_size = 1000
max_seq_len = 8
embedding_dim = 32
dropout = 0.1

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super(BigramLanguageModel, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding_table = nn.Embedding(max_seq_len, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, X):
        token_embedding = self.token_embedding_table(X)
        positional_embedding = self.positional_embedding_table(torch.arange(max_seq_len))
        X = token_embedding + positional_embedding
        logits = self.linear(X)

        return logits
    
class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)
        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        B, T, C = X.shape
        # X: (B, T, C)
        key = self.key(X)
        query = self.query(X)

        # compute attention scores
        tril = torch.ones((T, T), dtype=torch.long, device=device)
        mask = torch.tril(tril, diagonal=0)
        wei = query @ key.transpose(-1, -2) // (C ** 0.5)
        wei = wei.masked_fill(mask == 0, float('-inf'))
        wei = nn.Softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # compute attention values
        output = wei @ self.value(X)
        return output
    
# let's test this model
model = BigramLanguageModel().to(device)
X = torch.randint(0, vocab_size, (32, max_seq_len), dtype=torch.long, device=device)
print("X.shape: ", X.shape)
# test the model
logits = model(X)
print("returned logits shape: ", logits.shape) # torch.shape(logits) == (32, 8, 1000)