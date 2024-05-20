import torch
import torch.nn as nn
import math
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, input):
        input = input + self.pe[:, :input.size(1)].detach()
        return self.dropout(input)

#the input shape (batch_size, seq_length)
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=5000):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.max_len = max_len

    def forward(self, input):
        # Truncate or pad the input sequences to max_len
        input = input[:, :self.max_len]
        return self.embedding(input)

#the input shape is (batch_size, num_heads, seq_len, head_dim)
class ScaledProductAttn(nn.Module):
    def __init__(self, dropout = 0.1):
        super(ScaledProductAttn, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, attn_mask = None):
        _, _, _, d_k= query.shape
        assert d_k != 0
        attn = torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(d_k)
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == False, float('-inf'))
        attn = self.dropout(self.softmax(attn))
        context = torch.matmul(attn, value)
        return context

class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model,dropout = 0.1):
        super(MultiHeadAttn, self).__init__()
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
        self.n_head = n_head
        self.scaled_dot_attn = ScaledProductAttn(dropout)
        self.dropout = nn.Dropout(p = dropout)
        self.norm = nn.LayerNorm(d_model)


    def forward(self, x,  attn_mask=None):
        batch_size, seq_len, d_model = x.shape
        h_dim = d_model // self.n_head
        assert h_dim * self.n_head == d_model
        Q = self.Q(x).view(batch_size, seq_len, self.n_head, h_dim).permute(0,2,1,3)
        K = self.K(x).view(batch_size, seq_len, self.n_head, h_dim).permute(0,2,1,3)
        V = self.V(x).view(batch_size, seq_len, self.n_head, h_dim).permute(0,2,1,3)
        
        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_size, self.n_head, seq_len, seq_len)  # Expanding to [batch_size, n_head, seq_len, seq_len]
        # print(f"attn_mask shape after expansion: {attn_mask.shape}")
        attn_score = self.scaled_dot_attn(Q, K, V, attn_mask)
        attn_score = attn_score.permute(0,2,1,3).reshape(batch_size, seq_len, -1)

        attn_score = self.dropout(attn_score)
        attn_score = self.norm(attn_score + x)

        return attn_score


class MultiHeadCrossAttn(nn.Module):
    def __init__(self, n_head, d_model, dropout=0.1):
        super(MultiHeadCrossAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model
        self.head_dim = d_model // n_head

        assert self.head_dim * n_head == d_model, "d_model must be divisible by n_head"

        # Initialize linear layers for Q, K, V transformations
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        # Attention mechanism
        self.attention = ScaledProductAttn(dropout)

        # Layer norm and dropout
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target, memory, attn_mask=None):
        # target is from decoder, memory is the encoder's output
        batch_size, target_len, _ = target.shape
        _, memory_len, _ = memory.shape

        # Project target and memory to query, key, value spaces
        Q = self.query(target).view(batch_size, target_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(memory).view(batch_size, memory_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(memory).view(batch_size, memory_len, self.n_head, self.head_dim).permute(0, 2, 1, 3)

        # Expand the attention mask if present
        if attn_mask is not None:
            attn_mask = attn_mask.expand(batch_size, self.n_head, target_len, memory_len)

        # Apply scaled dot product attention
        attn_output = self.attention(Q, K, V, attn_mask)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(batch_size, target_len, self.d_model)

        # Apply dropout, add residual and norm
        output = self.dropout(attn_output)
        output = self.norm(output + target)

        return output
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias = False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias = False)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, input):
        residual = input
        output = self.fc(input)
        #dropout then add & norm
        output = self.dropout(output)
        output = self.norm(output + residual)
        return output



class EncoderLayer(nn.Module):
    # we put the multi-head attention and feedforward together as a encoder layer
    def __init__(self, n_head, d_model, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multiheadattn = MultiHeadAttn(n_head, d_model, dropout)
        self.fnn = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        x = self.norm1(x)
        attn_output = self.multiheadattn(x, attn_mask)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        ff_output = self.fnn(x)
        x = x + self.dropout(ff_output)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, d_model, d_ff, dropout= 0.1):
        super(Encoder, self). __init__()
        self.enc_layer = nn.ModuleList([EncoderLayer(n_head, d_model, d_ff, dropout) for _ in range(n_layer)])
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

    def forward(self,x, attn_mask):
        emb = self.embedding(x)
        x = self.pos_encoding(emb)
        for layer in self.enc_layer:
            x = layer(x, attn_mask)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        # print('self attn in decoder-----------')
        self.self_attn = MultiHeadAttn(n_head, d_model, dropout)
        # print('cross attn in decoder ************')
        self.cross_attn = MultiHeadCrossAttn(n_head, d_model)
        self.feed_forward = PoswiseFeedForwardNet(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.norm1(x)
        x2 = self.self_attn(x, attn_mask=tgt_mask)
        x = x + self.dropout1(x2)
        x = self.norm2(x)
        x2 = self.cross_attn(x, memory, src_mask)
        x = x + self.dropout2(x2)
        x = self.norm3(x)
        x2 = self.feed_forward(x)
        x = x + self.dropout3(x2)
        return x

