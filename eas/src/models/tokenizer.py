"""
Tokenizer specialized for logic/syllogism corpora
Handles ~500 tokens for logical expressions
"""
import json
import re
from typing import List, Dict, Union


class LogicTokenizer:
    """A specialized tokenizer for logical expressions and syllogisms"""
    
    def __init__(self, vocab_size: int = 500):
        # Define special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<START>': 2,
            '<END>': 3,
            '<SEP>': 4,  # Separator between premises and conclusion
            '<MASK>': 5
        }
        
        # Initial vocabulary with special tokens
        self.vocab = self.special_tokens.copy()
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Define logical operators and quantifiers
        self.logical_ops = [
            'AND', 'OR', 'NOT', 'IF', 'THEN', 'IFF', 'FOR_ALL', 'EXISTS',
            'IMPLIES', 'EQUIV', 'CONTRADICTION', 'THEREFORE', 'BECAUSE'
        ]
        
        # Add logical operators to vocabulary
        current_idx = len(self.vocab)
        for op in self.logical_ops:
            self.vocab[op] = current_idx
            self.reverse_vocab[current_idx] = op
            current_idx += 1
            
        # Define quantifiers
        self.quantifiers = [
            'all', 'some', 'no', 'every', 'each', 'any', 'most', 'few', 'many', 'several',
            'yes', 'true', 'false', 'correct', 'incorrect'
        ]
        
        # Add quantifiers to vocabulary
        for q in self.quantifiers:
            self.vocab[q] = current_idx
            self.reverse_vocab[current_idx] = q
            current_idx += 1
            
        # Define common logical subjects and predicates
        self.common_entities = [
            'A', 'B', 'C', 'X', 'Y', 'Z', 'S', 'P', 'Q', 'R', 'M', 'N',
            'person', 'human', 'man', 'woman', 'child', 'animal', 'dog', 'cat', 
            'mammal', 'bird', 'fish', 'reptile', 'amphibian', 'animal',
            'object', 'thing', 'item', 'entity', 'individual', 'being',
            'student', 'teacher', 'doctor', 'lawyer', 'engineer', 'scientist',
            'book', 'car', 'house', 'tree', 'flower', 'water', 'fire', 'air', 'earth',
            'red', 'blue', 'green', 'big', 'small', 'fast', 'slow', 'old', 'young',
            'smart', 'wise', 'tall', 'short', 'strong', 'weak', 'good', 'bad'            
        ]
        
        # Add entities to vocabulary
        for entity in self.common_entities:
            self.vocab[entity] = current_idx
            self.reverse_vocab[current_idx] = entity
            current_idx += 1
            
        # Define logical relations
        self.relations = [
            'is', 'are', 'was', 'were', 'be', 'being', 'been',
            'has', 'have', 'had', 'having',
            'does', 'do', 'did', 'doing',
            'belongs_to', 'contains', 'includes', 'excludes',
            'equals', 'equal_to', 'same_as', 'different_from',
            'greater_than', 'less_than', 'more_than', 'fewer_than'
        ]
        
        # Add relations to vocabulary
        for rel in self.relations:
            self.vocab[rel] = current_idx
            self.reverse_vocab[current_idx] = rel
            current_idx += 1
            
        # Define connectives
        self.connectives = [
            'and', 'or', 'but', 'if', 'then', 'when', 'while', 'since', 'because',
            'before', 'after', 'during', 'until', 'unless', 'except',
            'in', 'on', 'at', 'by', 'with', 'without', 'through', 'under', 'over'
        ]
        
        # Add connectives to vocabulary
        for conn in self.connectives:
            self.vocab[conn] = current_idx
            self.reverse_vocab[current_idx] = conn
            current_idx += 1
            
        # Add numbers for token expansion
        for i in range(10):
            num_token = f'NUM_{i}'
            self.vocab[num_token] = current_idx
            self.reverse_vocab[current_idx] = num_token
            current_idx += 1
            
        # Set maximum vocabulary size
        self.vocab_size = vocab_size
        self.max_idx = min(vocab_size - 1, current_idx - 1)
        
        # Add remaining indices as placeholder tokens
        while len(self.vocab) < vocab_size:
            token = f'TOK_{len(self.vocab)}'
            self.vocab[token] = len(self.vocab)
            self.reverse_vocab[len(self.vocab) - 1] = token
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize a text string into logical tokens"""
        # Normalize the text
        text = text.lower()
        
        # Split on common delimiters while preserving them
        # Use regex to capture punctuation and spaces as separate tokens
        tokens = re.findall(r"[\w]+|[^\s\w]", text)
        
        # Process tokens to match our vocabulary
        processed_tokens = []
        for token in tokens:
            # Clean up the token - remove extra punctuation
            clean_token = token.strip(".,!?;:'\"()[]{}")
            if clean_token in self.vocab:
                processed_tokens.append(clean_token)
            elif clean_token.lower() in self.vocab:
                processed_tokens.append(clean_token.lower())
            else:
                # Try to find partial matches or known patterns
                if any(q in clean_token for q in self.quantifiers):
                    processed_tokens.append('<UNK>')
                else:
                    processed_tokens.append('<UNK>')
                    
        return processed_tokens
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self.tokenize(text)
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['<UNK>'])
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if token not in self.special_tokens:
                    tokens.append(token)
            else:
                tokens.append('<UNK>')
        return ' '.join(tokens)
    
    def get_vocab_size(self) -> int:
        """Return the size of the vocabulary"""
        return len(self.vocab)
    
    def save_vocab(self, filepath: str):
        """Save the vocabulary to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.vocab, f)
    
    def load_vocab(self, filepath: str):
        """Load the vocabulary from a JSON file"""
        with open(filepath, 'r') as f:
            self.vocab = json.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}


def create_logic_tokenizer(vocab_size: int = 500) -> LogicTokenizer:
    """Create and return a logic tokenizer with specified vocabulary size"""
    return LogicTokenizer(vocab_size=vocab_size)


def create_small_tokenizer(vocab_size: int = 200) -> LogicTokenizer:
    """Create a smaller tokenizer for rapid prototyping"""
    return LogicTokenizer(vocab_size=vocab_size)