import torch
import torch.nn as nn

# Hyperparameters
vocab_size = 1000
max_seq_len = 8
embedding_dim = 128
dropout = 0.1
num_heads = 4
head_size = 32

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Embedding_model(nn.Module):

    def __init__(self):
        super(Embedding_model, self).__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_dim)
        self.positional_embedding_table = nn.Embedding(max_seq_len, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, X):
        token_embedding = self.token_embedding_table(X)
        positional_embedding = self.positional_embedding_table(torch.arange(max_seq_len))
        embeddings = token_embedding + positional_embedding

        return embeddings

def generate(X, max_len):
      
    B, T = X.shape

    for _ in range(max_len):

        # get the context_window
        context_window = X[:, -max_seq_len:]
        # get the logits
        logits = model(context_window)
        # get the last token
        last_token = logits[:, -1, :]
        # apply softmax
        last_token = nn.Softmax(dim=-1)(last_token)
        # get the next token
        next_token = torch.multinomial(last_token, num_samples=1)
        # append the next token to the context window
        X = torch.cat([X, next_token], dim=-1)
    
    return X

class Head(nn.Module):
    def __init__(self, head_size):
        super(Head, self).__init__()
        self.key = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.query = nn.Linear(embedding_dim, head_size)
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
        sotmax = nn.Softmax(dim=-1)
        wei = sotmax(wei)
        wei = self.dropout(wei)

        # compute attention values
        output = wei @ self.value(X)
        return output
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super(MultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.linear = nn.Linear(num_heads * head_size, embedding_dim)
    def forward(self, X):
        output = torch.cat([head(X) for head in self.heads], dim=-1)
        output = self.linear(output.view(output.shape[0], output.shape[1], -1))
        return output

# let's test this model
model = Embedding_model().to(device)
X = torch.randint(0, vocab_size, (32, max_seq_len), dtype=torch.long, device=device)
print("X.shape: ", X.shape)
# test the model
embeddings = model(X)
print("returned logits shape: ", embeddings.shape) # torch.shape(logits) == (32, 8, 1000)

generated_logits = generate(X, 8)
print(generated_logits.shape) # torch.shape(generated_logits) == (32, 16)

multihead = MultiHeadAttention(num_heads, head_size).to(device)
attention_scores = multihead(embeddings)
print('attention scores shape: ', attention_scores.shape)