import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encodings to input embeddings.

    This module produces a fixed sinusoidal positional encoding matrix and adds
    the appropriate slice to the input tensor. Supports optional time-step encoding
    for autoregressive inference.

    Args:
        d_model (int): Dimensionality of the embeddings.
        max_seq_len (int): Maximum sequence length supported.
        dropout_p (float): Dropout probability applied after adding encoding.
    """
    def __init__(self, d_model: int, max_seq_len: int, dropout_p: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout_p)

        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        i = torch.arange(0, d_model, 2, dtype=torch.float32)
        div_term = 10000 ** (i / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor, t: int = None) -> torch.Tensor:
        """
        Apply positional encoding to the input.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model).
            t (int, optional): If specified, treat input as single time step and use this index.
                               Sequence length of x must be 1 in this case.

        Returns:
            torch.Tensor: Encoded embeddings with same shape as input.

        Raises:
            ValueError: If seq_len exceeds max_seq_len or if t is out of bounds,
                        or if seq_len != 1 when t is specified.
        """
        if t is None:
            seq_len = x.size(1)
            if seq_len > self.max_seq_len:
                raise ValueError(f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}")
            return self.dropout(x + self.pe[:, :seq_len, :].to(x.device))
        else:
            if t >= self.max_seq_len:
                t = self.max_seq_len - 1
            if x.size(1) != 1:
                raise ValueError(f"When t is specified, seq_len must be 1, got {x.size(1)}")
            return self.dropout(x + self.pe[:, t:t+1, :].to(x.device))


class TransformerBlock(nn.Module):
    """
    Single transformer block consisting of multi-head self-attention and position-wise feedforward.

    Args:
        d_model (int): Dimensionality of the attention projections.
        embed_dim (int): Dimensionality of input embeddings.
        n_heads (int): Number of attention heads.
        dropout_p (float): Dropout probability applied in attention and feedforward.
        sliding_window (int, optional): Window size for sliding attention; if None, full sequence.
    """
    def __init__(self, d_model: int, embed_dim: int, n_heads: int,
                 dropout_p: float, sliding_window: int = None):
        super().__init__()
        self.n_heads = n_heads
        self.sliding_window = sliding_window

        self.linearQ = nn.Linear(embed_dim, d_model)
        self.linearK = nn.Linear(embed_dim, d_model)
        self.linearV = nn.Linear(embed_dim, d_model)

        self.MHSA = MultiHeadSelfAttention(d_model=d_model,
                                           dropout_p=dropout_p,
                                           sliding_window=sliding_window)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.pffn = PositionWiseFNN(d_model=d_model,
                                    dropout_p=dropout_p)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                _inference: bool = False,
                _first: bool = False) -> torch.Tensor:
        """
        Forward pass through transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).
            _inference (bool): If True, use cached K/V for autoregressive decoding.
            _first (bool): If True during inference, initialize cache with full context.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            ValueError: If sequence length constraints violated in inference mode.
        """
        if _inference and not _first:
            if x.shape[1] != 1:
                x = x[:, -1:, :] 
            x = self.layernorm1(x)
            q = self.linearQ(x)
            k = self.linearK(x)
            v = self.linearV(x)
        else:
            self.__c_seq_len = x.shape[1]
            x = self.layernorm1(x)
            q = self.linearQ(x)
            k = self.linearK(x)
            v = self.linearV(x)

        batch_size, seq_len, _ = q.shape
        head_dim = q.shape[-1] // self.n_heads

        q = q.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        context_len = self.sliding_window if self.sliding_window else self.__c_seq_len
        x_attn = self.MHSA(q, k, v,
                           _inference=_inference,
                           context_len=context_len,
                           _first=_first)
        x = x_attn + x
        x_res = x
        x = self.layernorm2(x)
        x = self.pffn(x) + x_res
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module with optional sliding-window masking and caching.

    Args:
        dropout_p (float): Dropout probability on attention weights and output.
        d_model (int): Dimensionality of attention input/output.
        sliding_window (int, optional): Maximum context length for each query position.
    """
    def __init__(self, dropout_p: float, d_model: int, sliding_window: int = None):
        super().__init__()
        self.sliding_window = sliding_window
        self.dropout = nn.Dropout(p=dropout_p)
        self.linearO = nn.Linear(d_model, d_model)

    def create_swa_mask(self, mask: torch.Tensor,
                        seq_len: int,
                        k_seq_len: int = None,
                        square: bool = True) -> torch.Tensor:
        """
        Apply sliding-window mask to a causal mask matrix.

        Args:
            mask (torch.Tensor): Initial mask of shape (seq_len, k_seq_len or seq_len).
            seq_len (int): Length of query sequence.
            k_seq_len (int, optional): Length of key sequence when square=False.
            square (bool): If True, use square mask (same seq_len for keys).

        Returns:
            torch.Tensor: Updated mask with positions outside the sliding window set to 1.
        """
        if square:
            for i in range(seq_len):
                if i >= self.sliding_window:
                    mask[i, : i - self.sliding_window] = 1
        else:
            for i in range(seq_len):
                for j in range(k_seq_len):
                    if j - self.sliding_window > 0:
                        mask[i, : j - self.sliding_window] = 1
        return mask

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                context_len: int = None,
                _inference: bool = False,
                _first: bool = None) -> torch.Tensor:
        """
        Compute multi-head self-attention with optional caching and sliding-window.

        Args:
            q, k, v (torch.Tensor): Query, Key, Value tensors of shape
                (batch_size, n_heads, seq_len, head_dim).
            context_len (int, optional): Context length for caching.
            _inference (bool): If True, perform incremental decoding using cache.
            _first (bool): If True, initialize key/value cache.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).

        Raises:
            RuntimeError: If seq_len != 1 when decoding with cache.
        """
        device = q.device
        batch_size, n_heads, seq_len, head_dim = q.shape
        if not _inference:
            mask = torch.ones(seq_len, seq_len, device=device)
            mask = torch.triu(mask, diagonal=1)
            if self.sliding_window:
                mask = self.create_swa_mask(mask, seq_len)
            attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
            attn_logits = attn_logits.masked_fill(mask == 1, float('-inf'))
            attn_scores = self.dropout(F.softmax(attn_logits, dim=-1))
            attn_output = torch.matmul(attn_scores, v)
            x = self.dropout(self.linearO(
                attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)))
        else:
            if _first:
                self.K_cache = k
                self.V_cache = v
                mask = torch.ones(seq_len, seq_len, device=device)
                mask = torch.triu(mask, diagonal=1)
                if self.sliding_window:
                    mask = self.create_swa_mask(mask, seq_len)
                attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_logits = attn_logits.masked_fill(mask == 1, float('-inf'))
                attn_scores = self.dropout(F.softmax(attn_logits, dim=-1))
                attn_output = torch.matmul(attn_scores, v)
                x = self.dropout(self.linearO(
                    attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)))
            else:
                if seq_len != 1:
                    raise RuntimeError('seq_len must be 1 during inference when _first=False')
                k_full = torch.cat([self.K_cache, k], dim=2)
                v_full = torch.cat([self.V_cache, v], dim=2)
                past_seq_len = k_full.shape[2]
                mask = torch.zeros(1, past_seq_len, device=device)
                if self.sliding_window and past_seq_len > self.sliding_window:
                    mask[:, : past_seq_len - self.sliding_window] = 1
                attn_logits = torch.matmul(q, k_full.transpose(-1, -2)) / math.sqrt(head_dim)
                attn_logits = attn_logits.masked_fill(mask == 1, float('-inf'))
                attn_scores = self.dropout(F.softmax(attn_logits, dim=-1))
                attn_output = torch.matmul(attn_scores, v_full)
                x = self.dropout(self.linearO(
                    attn_output.transpose(1, 2).contiguous().view(batch_size, 1, -1)))
                self.K_cache = k_full
                self.V_cache = v_full
                if self.K_cache.shape[2] > context_len:
                    self.K_cache = self.K_cache[:, :, -context_len:, :]
                    self.V_cache = self.V_cache[:, :, -context_len:, :]
        return x


class PositionWiseFNN(nn.Module):
    """
    Position-wise feedforward network consisting of two linear layers with GELU activation.

    Args:
        d_model (int): Input and output dimensionality.
        dropout_p (float): Dropout probability applied after second linear layer.
    """
    def __init__(self, d_model: int, dropout_p: float):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feedforward network.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output of same shape after applying feedforward.
        """
        x = F.gelu(self.linear1(x))
        return self.dropout(self.linear2(x))