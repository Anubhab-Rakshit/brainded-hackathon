import torch
import torch.nn as nn
import torch.nn.functional as F

class TriangularAttentionBlock(nn.Module):
    """
    RCTA: Triangular Cognitive Attention Block.
    Implements the verification loop:
    1. Image-Query: Decoder state queries Clinical Text/Global Context.
    2. Hypothesis Generation: Context queries Predicted Disease Labels.
    3. Verification Loop: Hypothesis queries Hierarchical Image Features (Pixel/Region).
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TriangularAttentionBlock, self).__init__()
        self.attn_text = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_label = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_image = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim) # For self-attention
        
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_ff = nn.LayerNorm(embed_dim)

    def forward(self, query, text_memory, label_memory, image_memory, image_mask=None, text_mask=None, tgt_mask=None):
        # query: [B, T, D] (Decoder state)
        
        # 0. Self-Attention (Standard Decoder Part)
        q2 = self.norm4(query)
        # attn_mask argument is for the query-key pairs
        q2, _ = self.self_attn(q2, q2, q2, attn_mask=tgt_mask)
        query = query + q2
        
        # 1. Image-Query / Text Context
        x = self.norm1(query)
        context, _ = self.attn_text(x, text_memory, text_memory, key_padding_mask=text_mask)
        query = query + context 
        
        # 2. Hypothesis Generation
        x = self.norm2(query)
        hypothesis, _ = self.attn_label(x, label_memory, label_memory)
        query = query + hypothesis
        
        # 3. Verification Loop
        x = self.norm3(query)
        verified, _ = self.attn_image(x, image_memory, image_memory)
        query = query + verified
        
        # Feed Forward
        x = self.norm_ff(query)
        out = self.feed_forward(x)
        query = query + out
        
        return query

class CognitiveDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=768, num_layers=6, num_heads=8, max_len=100):
        super(CognitiveDecoder, self).__init__()
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        self.layers = nn.ModuleList([
            TriangularAttentionBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, input_ids, text_memory, label_memory, image_memory):
        # input_ids: [B, T]
        B, T = input_ids.shape
        x = self.embed_tokens(input_ids) + self.pos_embed[:, :T, :]
        
        # Causal Mask (prevent peeking at future)
        # 0 = unmasked, -inf = masked
        tgt_mask = torch.triu(torch.full((T, T), float('-inf'), device=input_ids.device), diagonal=1)
        
        for layer in self.layers:
            x = layer(x, text_memory, label_memory, image_memory, tgt_mask=tgt_mask)
            
        x = self.norm(x)
        logits = self.output_head(x)
        return logits
