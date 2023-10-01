import torch
import torch.nn as nn
from model import BigramLanguageModel

class TreeOfThoughts(nn.Module):
    def __init__(self, num_branches, max_token_sequence_length, model):
        super(TreeOfThoughts, self).__init__()
        self.bigram_model = model
        self.num_branches = num_branches
        self.mtsl = max_token_sequence_length
        
        # Initializing an empty tree structure
        self.tree = {}

    def grow_tree(self, seed, max_depth):
        '''
        This takes in a seed text, initializes the root node of the tree with the seed text, and then recursively
        grows the tree by generating child nodes.
        '''
        self.tree = {'text': seed, 'children': []}
        self._grow_tree_recursive(self.tree, max_depth)
        return self.tree
    
    def grow_tree_recursively(self, node, depth):
        '''
        This takes in the initial tree, extracts the seed, predicts the next token which we then attach to a new
        child node. We then append the child node to the current node's children and pass this child node to the
        generate_next_token function and get a new token which we then attach to the child node. This goes on until
        the depth of the tree hits any of the limits.

        Not tested yet btw.
        '''
        if depth <= 0 or depth > self.num_branches > self.num_branches:
            return
        
        next_token = self.bigram_model.generate_next_token(node['text'])
        child_node = {'text': next_token, 'children': []}
        node['children'].append(child_node)
        self._grow_tree_recursive(child_node, depth - 1)

    def visualize_tree(self, tree):
        '''
        This takes in a tree and prints it out in a nice format.
        '''
        self._visualize_tree_recursive(tree, 0)
    
    def _visualize_tree_recursive(self, node, indent=""):
        '''
        This takes in a node and prints it out in a nice format.
        '''
        new_str = indent + node['text'] + "\n"
        for child in node['children']:
            self._visualize_tree_recursive(child, indent + "  ")

# Assumptions as of writing this: (1) The bigram model is token level and not character level.
# (2) The bigram model is trained on a dataset that is properly processed to end sentences with a special token like
# <eos> or something.