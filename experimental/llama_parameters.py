from transformers import PretrainedConfig


def compute_total_parameters(config: PretrainedConfig, decoder_bias: bool) -> int:
    """
    Compute the total number of parameters in the model.
    Args:
        config: The configuration of the model.
        decoder_bias: A boolean indicating whether to include the bias in the decoder blocks."""
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    num_key_value_heads = config.num_key_value_heads
    num_attention_heads = config.num_attention_heads
    inter_size = config.intermediate_size
    head_dim = hidden_size // num_attention_heads

    # Embedding generation
    embedding_params = vocab_size * hidden_size

    # Decoder blocks (self-attention + FFN)
    decoder_blocks_params = (hidden_size +
                             2*hidden_size*hidden_size + 2*hidden_size*num_key_value_heads*head_dim +
                             hidden_size +
                             3*inter_size*hidden_size) * num_hidden_layers

    if decoder_bias:
        decoder_blocks_params += num_hidden_layers * (hidden_size + 2 * num_key_value_heads*head_dim)

    # RMS norm
    rms_norm_params = hidden_size

    # LM head
    lm_head_params = vocab_size * hidden_size

    return embedding_params + decoder_blocks_params + rms_norm_params + lm_head_params
