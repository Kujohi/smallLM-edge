import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from torch.nn import functional as F

class TokenLearner(nn.Module):
    """
    Implementation of TokenLearner module that learns a small set of tokens from the input features.
    """
    def __init__(self, input_dim=768, num_tokens=8, bottleneck_dim=64):
        super().__init__()
        self.num_tokens = num_tokens
        
        # Spatial downsampling with MLP
        self.token_selector = nn.Sequential(
            nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(bottleneck_dim, num_tokens, kernel_size=1),
        )
        
    def forward(self, x):
        # Assuming x is of shape [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, hidden_dim = x.shape
        
        # Reshape to 2D feature map (assuming square features)
        h = w = int(seq_len**0.5)
        x_2d = x.reshape(batch_size, h, w, hidden_dim).permute(0, 3, 1, 2)
        
        # Generate attention weights
        attention_weights = self.token_selector(x_2d)  # [batch_size, num_tokens, h, w]
        attention_weights = attention_weights.view(batch_size, self.num_tokens, -1)
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        # Reshape input to match attention
        x_flat = x.reshape(batch_size, -1, hidden_dim)  # [batch_size, h*w, hidden_dim]
        
        # Weighted sum to get tokens
        tokens = torch.bmm(attention_weights, x_flat)  # [batch_size, num_tokens, hidden_dim]
        
        return tokens


class QFormer(nn.Module):
    """
    A simplified Q-Former implementation using a BERT backbone
    """
    def __init__(self, input_dim=768, output_dim=576, num_query_tokens=32):
        super().__init__()
        self.num_query_tokens = num_query_tokens
        
        # Initialize learnable query tokens
        self.query_tokens = nn.Parameter(torch.zeros(1, num_query_tokens, input_dim))
        nn.init.normal_(self.query_tokens, std=0.02)
        
        # Projection for input tokens if dimensions don't match
        self.input_proj = nn.Linear(input_dim, input_dim)
        
        # BERT model for cross-attention
        config = BertConfig(
            hidden_size=input_dim,
            num_hidden_layers=2,
            num_attention_heads=12,
            intermediate_size=input_dim * 4,
        )
        self.bert = BertModel(config)
        
        # Final projection to output dimension
        self.output_proj = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        # x shape: [batch_size, num_tokens, input_dim]
        batch_size = x.shape[0]
        
        # Expand query tokens for batch
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        
        # Project input tokens
        x = self.input_proj(x)
        
        # Cross-attention with BERT
        # We need to concat and prepare attention mask
        concat_tokens = torch.cat([query_tokens, x], dim=1)
        attention_mask = torch.ones(batch_size, concat_tokens.shape[1], device=x.device)
        
        # We want queries to attend to inputs, but not to each other
        # But BERT expects a flat attention mask, so we'll just let everything attend to everything
        outputs = self.bert(inputs_embeds=concat_tokens, attention_mask=attention_mask)
        
        # Take only the query tokens' outputs
        query_outputs = outputs.last_hidden_state[:, :self.num_query_tokens]
        
        # Project to output dimension
        output = self.output_proj(query_outputs)
        
        return output


class MyTokenLearnerQFormerConnector(nn.Module):
    """
    Custom connector that combines TokenLearner and QFormer to process visual features
    for the language model.
    """
    def __init__(self, 
                 vision_hidden_size=768,
                 text_hidden_size=576,
                 token_learner_tokens=64,
                 qformer_query_tokens=32):
        super().__init__()
        
        self.token_learner = TokenLearner(
            input_dim=vision_hidden_size,
            num_tokens=token_learner_tokens,
            bottleneck_dim=vision_hidden_size // 4
        )
        
        self.q_former = QFormer(
            input_dim=vision_hidden_size,
            output_dim=text_hidden_size,
            num_query_tokens=qformer_query_tokens
        )
        
    def forward(self, vision_hidden_states):
        # Reduce tokens with TokenLearner
        token_learner_output = self.token_learner(vision_hidden_states)
        
        # Process with QFormer to get the final output
        qformer_output = self.q_former(token_learner_output)
        
        return qformer_output 