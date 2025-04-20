import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks import PositionalEncoding, TransformerBlock

class SLM(nn.Module):
    """
    Sequence Language Model combining token embeddings, positional encoding, and transformer blocks.

    This model embeds input token indices, adds positional information (either fixed sinusoidal
    or learned embeddings), and processes the sequence through a stack of transformer blocks.
    Finally, it projects back to vocabulary logits using tied weights.

    Args:
        d_model (int): Dimensionality of transformer projection space.
        embed_dim (int): Dimensionality of token embeddings.
        max_seq_len (int): Maximum sequence length for positional encodings.
        dropout_p (float): Dropout probability applied in positional encoding and transformer.
        n_heads (int): Number of attention heads in each transformer block.
        n_blocks (int): Number of transformer blocks.
        context_len (int): Context length for sliding-window attention caching.
        vocab_size (int): Size of the vocabulary for token embeddings and output projection.
        pretrained_embeddings (torch.Tensor, optional): Pretrained embedding weights of shape
            (vocab_size, embed_dim). If provided, embeddings are loaded and optionally frozen.
        sliding_window (int, optional): Window size for local attention; if None, use full attention.
        learned_pe (bool, optional): If True, learn positional embeddings instead of fixed sinusoidal.
        freeze_embeddings (bool, optional): If True when using pretrained embeddings, they are frozen.
        **kwargs: Additional keyword arguments (unused).
    """
    def __init__(
        self,
        d_model: int,
        embed_dim: int,
        max_seq_len: int,
        dropout_p: float,
        n_heads: int,
        n_blocks: int,
        context_len: int,
        vocab_size: int,
        pretrained_embeddings: torch.Tensor = None,
        sliding_window: int = None,
        learned_pe: bool = False,
        freeze_embeddings: bool = False,
        verbose:bool = False,
        **kwargs
    ):
        super().__init__()
  
        self.verbose = verbose
        
        if verbose:
            print("Initializing Model")
            
        self.n_blocks = n_blocks
        self.learned_pe = learned_pe
        self.max_seq_len = max_seq_len
        self.context_len = context_len
        self.pretrained_embeddings = pretrained_embeddings
        self.t = None  

        if not isinstance(pretrained_embeddings, torch.Tensor):
            self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        else:
            if verbose:
                print('Loading pretrained embeddings')
            self.token_embedding_table = nn.Embedding.from_pretrained(
                pretrained_embeddings, freeze=freeze_embeddings
            )

        if not learned_pe:
            self.pos_emb = PositionalEncoding(
                d_model=embed_dim,
                max_seq_len=max_seq_len,
                dropout_p=dropout_p
            )
        else:
            self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_model,
                    embed_dim,
                    n_heads,
                    dropout_p,
                    sliding_window if i % 2 == 0 else None
                )
                for i in range(n_blocks)
            ]
        )

        self.linearO = nn.Linear(d_model, vocab_size)
        self.linearO.weight = self.token_embedding_table.weight

        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights.

        Applies Xavier normal initialization to all linear layers, zeros biases,
        and normal initialization to token embeddings if not pretrained.
        """
        
        if self.verbose: 
            print(f"Initializing weights using Xavier Uniform Init.")
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding) and self.pretrained_embeddings is None:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        _inference: bool = False,
        _first: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the sequence language model.

        Args:
            x (torch.Tensor): Input token indices, shape (batch_size, seq_len).
            _inference (bool): If True, run in autoregressive inference mode with caching.
            _first (bool): If True when _inference, initialize cache and use full context.

        Returns:
            torch.Tensor: Output logits over vocabulary, shape (batch_size, seq_len, vocab_size).

        Raises:
            ValueError: If inference and sequence length assumptions are violated.
        """
        seq_len = x.shape[1]
        if not _inference:
            self.t = None  

        if _inference and not _first:
            x = x[:, -1:]  

        x = self.token_embedding_table(x)

        if _inference and not _first:
            x = self._add_pos_encoding(x, self.t)
            self.t += 1
        else:
            x = self._add_pos_encoding(x, self.t)
            if _inference and _first:
                self.t = seq_len

        for i in range(self.n_blocks):
            x = self.transformer_blocks[i](x, _inference=_inference, _first=_first)

        return self.linearO(x)

    def _add_pos_encoding(
        self,
        x: torch.Tensor,
        t: int = None
    ) -> torch.Tensor:
        """
        Add positional information to embeddings.

        Uses fixed sinusoidal or learned positional embeddings depending on configuration.

        Args:
            x (torch.Tensor): Embedded tokens, shape (batch_size, seq_len, embed_dim).
            t (int, optional): Current timestep for autoregressive inference.

        Returns:
            torch.Tensor: Input embeddings with positional encodings added.
        """
        seq_len = x.size(1)
        if not self.learned_pe:
            return self.pos_emb(x, t)
        else:
            if t is None:
                positions = torch.arange(seq_len, device=x.device)
            else:
                if t >= self.max_seq_len:
                    t = self.max_seq_len - 1  
                positions = torch.tensor([t], device=x.device)
            pe = self.pos_emb(positions).unsqueeze(0)
            return x + pe

