import torch
import torch.nn as nn
from model import BigramLanguageModel

class TreeOfThoughts(nn.Module):
    def __init__(self, num_branches, max_token_sequence_length):
        super(TreeOfThoughts, self).__init__()
        self.bigram_model = BigramLanguageModel()
        self.num_branches = num_branches
        self.mtsl = max_token_sequence_length
        self.branch_seed = [None] * self.num_branches
        self.tree = [[None for _ in range(self.mtsl)] for _ in range(self.num_branches)]
    
    def forward(self, seed):
        for _ in range(self.num_branches):
            self.branch_seed[_] = self.bigram_model(seed)

        for i in range(self.num_branches):
            for j in range(self.mtsl):
                self.tree[i][j] = self.bigram_model(self.branch_seed[i][j])

        return self.tree

