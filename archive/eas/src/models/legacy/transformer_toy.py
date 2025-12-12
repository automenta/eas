"""
Minimal Autoregressive Transformer Implementation for EAS
Based on the specification: 2 layers, 8 heads, 512 hidden dim
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention module"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Linear projections
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Reshape and apply final linear transformation
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        output = self.W_o(attention_output)

        return output, attention_weights


class PositionWiseFFN(nn.Module):
    """Position-wise feed-forward network"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with layer normalization and residual connections"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionWiseFFN(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention with residual connection
        attn_output, _ = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        return x


class AutoregressiveTransformer(nn.Module):
    """Minimal Autoregressive Transformer for EAS"""
    def __init__(self, vocab_size, d_model=512, num_layers=2, num_heads=8, max_seq_len=512):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.middle_layer_idx = num_layers // 2  # For 2 layers, this will be 1 (second layer), for 1 layer, 0

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))

        # Create transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff=d_model*4)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialize position encodings
        self.pos_encoding.data = self._generate_positional_encodings(max_seq_len, d_model)

        # Hook storage for layer activations
        self.layer_activations = {}
        self.intervention_hooks = {}

        # Register forward hooks for each layer
        for i, layer in enumerate(self.layers):
            layer.register_forward_hook(self._make_layer_hook(i))

    def _generate_positional_encodings(self, max_len, d_model):
        """Generate positional encodings as in the original transformer paper"""
        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(math.log(10000.0) / d_model))

        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)

        return pos_enc.unsqueeze(0)

    def _make_layer_hook(self, layer_idx):
        """Make a hook function to capture activations at a specific layer"""
        def hook(module, input, output):
            # Store the output of the layer
            # Use input for hooking before layer processing
            self.layer_activations[layer_idx] = input[0].detach().clone()
            # If we have an intervention for this layer, apply it
            if layer_idx in self.intervention_hooks:
                # Apply the intervention function
                intervention_func = self.intervention_hooks[layer_idx]
                return intervention_func(input[0])
        return hook

    def register_intervention_hook(self, layer_idx, intervention_func):
        """Register an intervention function for a specific layer"""
        self.intervention_hooks[layer_idx] = intervention_func

    def remove_intervention_hook(self, layer_idx):
        """Remove an intervention hook from a specific layer"""
        if layer_idx in self.intervention_hooks:
            del self.intervention_hooks[layer_idx]

    def get_layer_activation(self, layer_idx):
        """Get the activation from a specific layer"""
        return self.layer_activations.get(layer_idx)

    def set_layer_activation(self, layer_idx, new_activation):
        """Set a new activation for a specific layer (for EAS intervention)"""
        self.layer_activations[layer_idx] = new_activation

    def forward(self, x, attention_mask=None):
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = x + self.pos_encoding[:, :x.size(1), :]

        # Pass through each transformer layer
        for i, layer in enumerate(self.layers):
            x = layer(x, attention_mask)

        # Final normalization and output projection
        x = self.norm(x)
        output = self.output_projection(x)

        return output


def create_small_model(vocab_size, d_model=128, num_layers=1, num_heads=4):
    """Create a smaller model variant for rapid prototyping"""
    return AutoregressiveTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads
    )


def create_standard_model(vocab_size):
    """Create the standard model as specified"""
    return AutoregressiveTransformer(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=2,
        num_heads=8
    )