import torch
import torch.nn as nn
import math



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 计算固定的位置编码矩阵 pe，大小 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)   # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)   # 0,2,4...
        pe[:, 1::2] = torch.cos(position * div_term)   # 1,3,5...

        pe = pe.unsqueeze(0)  # [1, max_len, d_model] 方便和 batch 数据相加
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # 只取前 seq_len 个位置编码，加到 x 上
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性投影 + reshape + 多头
        query = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key   = self.W_k(key)  .view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 计算注意力分数 (QK^T / sqrt(d_k))
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 3. softmax 得到注意力权重
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # 4. 加权求和得到上下文表示 (attn * V)
        context = torch.matmul(attn, value)
        
        # 5. 变回 [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 6. 输出 projection
        output = self.W_o(context)
        return output, attn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attn_output, attn = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiheadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self-attention
        attn_output, self_attn = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, cross_attn = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x, self_attn, cross_attn

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        attentions = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attentions.append(attn)
        x = self.norm(x)
        return x, attentions

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        self_attentions = []
        cross_attentions = []
        for layer in self.layers:
            x, self_attn, cross_attn = layer(x, enc_output, src_mask, tgt_mask)
            self_attentions.append(self_attn)
            cross_attentions.append(cross_attn)
        x = self.norm(x)
        return x, self_attentions, cross_attentions

class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output, enc_attentions = self.encoder(src, src_mask)
        dec_output, self_attentions, cross_attentions = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        return dec_output, enc_attentions, self_attentions, cross_attentions
    


class AdditionModel(nn.Module):
    def __init__(self, vocab_size, pad_idx = 0, d_model=128, num_layers=2, num_heads=4, d_ff=256, max_len=50, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.positional_encoding = PositionalEncoding(d_model = d_model)
        self.transformer = Transformer(num_layers, d_model, num_heads, d_ff, dropout)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        # 先加位置编码
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        out, _, _, _ = self.transformer(src, tgt, src_mask, tgt_mask)
        return self.fc_out(out)

def create_src_mask(src, pad_idx, device):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2).to(device)

def create_tgt_mask(tgt, pad_idx, device):
    seq_len = tgt.size(1)
    look_ahead_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    look_ahead_mask = (~look_ahead_mask).to(device)
    padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2).to(device)
    tgt_mask = look_ahead_mask.unsqueeze(0).unsqueeze(0) & padding_mask.expand(-1, -1, seq_len, -1)
    return tgt_mask