import torch
import torch.nn as nn
from model import Embedding_model, MultiHeadAttention, BigramLanguageModel

class TreeOfThoughts(nn.Module):
    def __init__(self, bigram_model, num_branches=3, max_seq_len=20):
        super(TreeOfThoughts, self).__init__()
        self.bigram_model = BigramLanguageModel()
        self.num_branches = num_branches
        self.max_seq_len = max_seq_len

    def forward(self, seed_token):
        # Generate num_branches sequences of tokens
        branches = []
        for i in range(self.num_branches):
            sequence = [seed_token]
            for j in range(self.max_seq_len):
                # Predict the next token using the bigram model
                next_token = self.bigram_model.forward(sequence)
                if next_token == '<EOS>':
                    break
                sequence.append(next_token)
            branches.append(sequence)
        return branches

