from .encoder import *
import torch.nn as nn 

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

class Decoder(nn.Module):
    def __init__(self,vocab_size, n_layer, n_head, d_model, d_ff, dropout =0.1):
        super(Decoder, self).__init__()
        self.layers =nn.ModuleList([DecoderLayer(n_head, d_model, d_ff, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(d_model, vocab_size)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        # self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, memory, src_mask=None, tgt_mask = None):
        emd = self.embedding(x)
        x = self.pos_encoding(emd)
        for layer in self.layers:
            x = layer(x, memory, src_mask=src_mask, tgt_mask=tgt_mask)
        x = self.norm(x)
        #the final logits
        logits = self.linear(x)
        # return x
        return logits

