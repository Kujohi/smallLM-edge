import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaModel
from typing import List, Optional, Tuple, Union
import math

class TokenLearner(nn.Module):
    """
    Implementation of TokenLearner module which dynamically extracts tokens from visual features.
    Based on the paper: "TokenLearner: Adaptive Space-Time Tokenization for Videos"
    """
    def __init__(
        self,
        in_channels=768,
        num_tokens=8,
        bottleneck_dim=64,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        
        self.norm = nn.LayerNorm(in_channels)
        
        # Input projection to reduce dimensions
        self.input_projection = nn.Sequential(
            nn.Linear(in_channels, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Token generation layers
        self.token_attention = nn.Sequential(
            nn.Linear(bottleneck_dim, num_tokens),
            nn.Sigmoid()
        )
        
        # Output projection to get back to original dimension
        self.output_projection = nn.Sequential(
            nn.Linear(bottleneck_dim, in_channels),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        """
        Args:
            x: Visual features with shape [batch_size, seq_len, hidden_dim]
        Returns:
            Learned tokens with shape [batch_size, num_tokens, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Apply layer normalization
        x_norm = self.norm(x)
        
        # Project to bottleneck dimension
        x_projected = self.input_projection(x_norm)
        
        # Generate token attention weights
        attention = self.token_attention(x_projected)  # [batch_size, seq_len, num_tokens]
        attention = attention.transpose(1, 2)  # [batch_size, num_tokens, seq_len]
        
        # Normalize attention weights
        attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Generate tokens by weighted aggregation of features
        tokens = torch.bmm(attention, x_projected)  # [batch_size, num_tokens, bottleneck_dim]
        
        # Project back to original dimension
        tokens = self.output_projection(tokens)  # [batch_size, num_tokens, hidden_dim]
        
        return tokens


# RMSNorm implementation similar to LlamaRMSNorm
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Custom LLaMA-style attention for QFormer
class LlamaStyleAttention(nn.Module):
    def __init__(
        self,
        hidden_size=576,
        num_heads=8,
        head_dim=None,
        bias=False,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        
        if (self.head_dim * num_heads) != self.hidden_size:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_heads {num_heads}")
            
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=bias)
        
        self.scale = 1 / math.sqrt(self.head_dim)
    
    def forward(self, queries, keys, values, attention_mask=None):
        batch_size, q_len, _ = queries.shape
        _, k_len, _ = keys.shape
        
        # Project queries, keys, values
        query_states = self.q_proj(queries).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = self.k_proj(keys).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = self.v_proj(values).view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Normalize attention weights
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.hidden_size)
        
        # Output projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output


# MLP similar to LlamaMLP
class LlamaStyleMLP(nn.Module):
    def __init__(
        self,
        hidden_size=576,
        intermediate_size=1536,
        bias=False,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
        
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# Custom LLaMA-style decoder layer
class LlamaStyleDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size=576,
        num_attention_heads=8,
        intermediate_size=1536,
        bias=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Self attention
        self.self_attn = LlamaStyleAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            bias=bias,
        )
        
        # Cross attention
        self.cross_attn = LlamaStyleAttention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            bias=bias,
        )
        
        # MLP
        self.mlp = LlamaStyleMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
        )
        
        # Layer norms
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.post_cross_attention_layernorm = RMSNorm(hidden_size)
        
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
        # Self attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, hidden_states, hidden_states)
        hidden_states = residual + hidden_states
        
        # Cross attention
        if encoder_hidden_states is not None:
            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.cross_attn(hidden_states, encoder_hidden_states, encoder_hidden_states)
            hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_cross_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


# Custom LLaMA-style QFormer 
class LlamaStyleQFormer(nn.Module):
    def __init__(
        self,
        hidden_size=576,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=1536,
        num_query_tokens=16,
        bias=False,
    ):
        super().__init__()
        
        # Create learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, hidden_size))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Create decoder layers
        self.layers = nn.ModuleList([
            LlamaStyleDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                bias=bias,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(hidden_size)
        
    def forward(self, input_tokens):
        """
        Args:
            input_tokens: Tokens from TokenLearner with shape [batch_size, num_tokens, hidden_dim]
        Returns:
            Processed query tokens with shape [batch_size, num_query_tokens, hidden_dim]
        """
        batch_size = input_tokens.shape[0]
        
        # Expand query tokens to batch size
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        hidden_states = query_tokens
        
        # Process through decoder layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, encoder_hidden_states=input_tokens)
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        return hidden_states


class MyTokenLearnerQFormerConnector(nn.Module):
    """
    Custom connector that combines TokenLearner and QFormer for more effective
    vision-to-language mapping.
    """
    def __init__(
        self,
        vision_hidden_size=768,
        text_hidden_size=576,
        token_learner_tokens=32,
        qformer_query_tokens=16,
    ):
        super().__init__()
        
        self.token_learner = TokenLearner(
            in_channels=vision_hidden_size,
            num_tokens=token_learner_tokens,
            bottleneck_dim=vision_hidden_size // 4,
        )
        
        # Use LlamaStyleQFormer instead of BertModel-based QFormer
        self.q_former = LlamaStyleQFormer(
            hidden_size=vision_hidden_size,
            num_query_tokens=qformer_query_tokens,
        )
        
        # Final projection to match text model dimension
        self.projection = nn.Linear(vision_hidden_size, text_hidden_size, bias=False)
    
    def forward(self, vision_features):
        """
        Convert vision features to format compatible with text model
        
        Args:
            vision_features: Features from vision encoder [batch_size, seq_len, vision_hidden_size]
        
        Returns:
            Projected features for text model [batch_size, qformer_query_tokens, text_hidden_size]
        """
        # Extract tokens with TokenLearner
        learned_tokens = self.token_learner(vision_features)
        
        # Process with QFormer
        qformer_output = self.q_former(learned_tokens)
        
        # Project to text model dimension
        projected_features = self.projection(qformer_output)
        
        return projected_features 