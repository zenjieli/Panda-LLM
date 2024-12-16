def compute_total_parameters(embedding_dim: int,
                             head_dim: int,
                             num_key_value_heads: int,
                             num_decoder_blocks: int,
                             vocab_size: int) -> int:
    # Embedding generation
    embedding_params = vocab_size * embedding_dim

    # Decoder blocks (self-attention + FFN)
    decoder_blocks_params = (embedding_dim + 2 * embedding_dim * embedding_dim + 2 * embedding_dim * num_key_value_heads * head_dim +
                                 embedding_dim + 12 * embedding_dim * embedding_dim) * num_decoder_blocks
    # RMS norm
    rms_norm_params = embedding_dim

    # LM head
    lm_head_params = vocab_size * embedding_dim

    return embedding_params + decoder_blocks_params + rms_norm_params + lm_head_params
