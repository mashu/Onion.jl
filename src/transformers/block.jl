"""
    TransformerBlock(dim::Int, n_heads::Int, n_kv_heads::Int = n_heads, ff_hidden_dim = 4 * dim; norm_eps=1f-5, qkv_bias=false)

Transformer block for GQAttention (as in Llama3).

# Forward Pass

The forward pass signature is:

```julia
(block::TransformerBlock)(x, xs...; cond=nothing, pair_feats=nothing, pair=block.pair_proj(pair_feats), kws...)
```

where:
- `x`: Input tensor of shape `(dim, seqlen, batch)`
- `xs...`: Additional positional arguments (e.g., RoPE indices)
- `cond`: Optional conditioning tensor for adaptive normalization. When provided, this is passed to both `attention_norm` and `ffn_norm`. 
  - For regular `RMSNorm` or `LayerNorm`, this parameter is ignored (can be `nothing`)
  - For `AdaLN` (Adaptive Layer Normalization), this is required and should have shape `(cond_dim, batch)` where `cond_dim` matches the dimension specified when creating `AdaLN`
  - When using `AdaLN`, the conditioning tensor is passed directly to the normalization layer. `AdaLN` uses learned linear transformations (Dense layers) to compute scale and shift parameters from the conditioning tensor, which then modulate the normalized output: `output = normalized(x) * (1 + scale(cond)) + shift(cond)`
- `pair_feats`: Optional pair features for attention
- `kws...`: Additional keyword arguments passed to attention, including:
  - `causal::Bool=false`: Enable causal masking
  - `kpad_mask`: Optional padding mask for keys

# Masking

The forward pass supports various masking options through keyword arguments:

## Without mask

```julia
dim = 64
n_heads = 8
n_kv_heads = 4
seqlen = 10

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads, n_kv_heads)
h = randn(Float32, dim, seqlen, 1)

# Forward pass without mask
h = t(h, 1, rope[1:seqlen])
```

## Causal mask

Use `causal=true` to enable causal masking:

```julia
dim = 64
n_heads = 8
seqlen = 10

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads)
h = randn(Float32, dim, seqlen, 1)

# Forward pass with causal mask
h = t(h, 1, rope[1:seqlen]; causal=true)
```

## Padding mask with self_att_padding_mask

Use `kpad_mask` with `self_att_padding_mask` for self-attention padding:

```julia
dim = 64
n_heads = 8
seqlen = 10
batch = 2

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads)
h = randn(Float32, dim, seqlen, batch)

# Create padding mask: sequence-level mask where 1 indicates valid position, 0 indicates padding
# Example: batch 1 has all positions valid, batch 2 has position 10 padded
padmask = ones(Float32, seqlen, batch)
padmask[10, 2] = 0  # Mark position 10 in batch 2 as padding

# Convert to attention mask
kpad_mask = Onion.self_att_padding_mask(padmask)

# Forward pass with padding mask
h = t(h, 1, rope[1:seqlen]; kpad_mask=kpad_mask)
```

## Padding mask with cross_att_padding_mask (for cross-attention scenarios)

Use `kpad_mask` with `cross_att_padding_mask` when using TransformerBlock with different query and key sequences:

```julia
dim = 64
n_heads = 8
q_seqlen = 10
k_seqlen = 12
batch = 2

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads)
q = randn(Float32, dim, q_seqlen, batch)
k = randn(Float32, dim, k_seqlen, batch)

# Create padding mask for keys: sequence-level mask where 1 indicates valid position
padmask = ones(Float32, k_seqlen, batch)
padmask[12, 2] = 0  # Mark position 12 in batch 2 as padding

# Convert to cross-attention mask: shape is (k_seqlen, q_seqlen, batch)
kpad_mask = Onion.cross_att_padding_mask(padmask, q_seqlen)

# Forward pass: pass key as positional argument after query
# Note: The attention layer supports cross-attention when different keys are provided
h = t(q, k; rope=rope[1:q_seqlen], krope=rope[1:k_seqlen], kpad_mask=kpad_mask)
```

## Combining causal and padding masks

You can combine both causal and padding masks:

```julia
dim = 64
n_heads = 8
seqlen = 10
batch = 2

rope = RoPE(dim ÷ n_heads, 1000)
t = TransformerBlock(dim, n_heads)
h = randn(Float32, dim, seqlen, batch)

# Create padding mask
padmask = ones(Float32, seqlen, batch)
padmask[10, 2] = 0
kpad_mask = Onion.self_att_padding_mask(padmask)

# Forward pass with both causal and padding masks
h = t(h, 1, rope[1:seqlen]; causal=true, kpad_mask=kpad_mask)
```

## Using with adaptive normalization (AdaLN)

When using `AdaLN` (Adaptive Layer Normalization) instead of `RMSNorm`, you must pass a conditioning tensor via the `cond` parameter. 
`AdaLN` takes the conditioning tensor and passes it through learned linear transformations (Dense layers) to compute scale and shift parameters. 
These learned parameters then modulate the normalized output according to: `output = normalized(x) * (1 + scale(cond)) + shift(cond)`.

The conditioning tensor is passed directly to the normalization layers (`attention_norm` and `ffn_norm`) during the forward pass.

Use `AdaTransformerBlock` for a convenience constructor that creates a TransformerBlock with `AdaLN`:

```julia
dim = 64
cond_dim = 32  # Dimension of conditioning tensor
n_heads = 8
seqlen = 10
batch = 2

rope = RoPE(dim ÷ n_heads, 1000)
t = AdaTransformerBlock(dim, cond_dim, n_heads)
h = randn(Float32, dim, seqlen, batch)

# Create conditioning tensor
cond = randn(Float32, cond_dim, batch)

# Forward pass with conditioning tensor
h = t(h, 1, rope[1:seqlen]; cond=cond)

# You can still use masks with adaptive normalization
h = t(h, 1, rope[1:seqlen]; cond=cond, causal=true)
```

Alternatively, you can manually create a TransformerBlock with AdaLN:

```julia
using Onion: AdaLN

dim = 64
cond_dim = 32
n_heads = 8
t = TransformerBlock(
    dim, n_heads;
    attention_norm = AdaLN(dim, cond_dim),
    ffn_norm = AdaLN(dim, cond_dim)
)

# Usage is the same
cond = randn(Float32, cond_dim, batch)
h = t(h, 1, rope[1:seqlen]; cond=cond)
```
"""
@concrete struct TransformerBlock
    attention
    feed_forward
    attention_norm
    ffn_norm
    pair_proj
end

@layer TransformerBlock

function TransformerBlock(
    in_dim::Int, n_heads::Int,
    n_kv_heads::Int = n_heads,
    ff_hidden_dim::Int = 4*in_dim;
    out_init_scale=1, norm_eps=1f-5,
    feed_forward = StarGLU(in_dim, ff_hidden_dim),
    attention_norm = RMSNorm(in_dim, eps=norm_eps),
    ffn_norm = RMSNorm(in_dim, eps=norm_eps),
    pair_proj = identity,
    kws...
)
    return TransformerBlock(
        Attention(in_dim, n_heads, n_kv_heads; out_init_scale, kws...),
        feed_forward, attention_norm, ffn_norm, pair_proj
    )
end

function (block::TransformerBlock)(
    x, xs...;
    cond=nothing, pair_feats=nothing,
    pair=block.pair_proj(pair_feats),
    kws...
)
    cond = isnothing(cond) ? () : (cond,)
    h = x + block.attention(
        block.attention_norm(x, cond...), xs...;
        pair, kws...)
    return h + block.feed_forward(block.ffn_norm(h, cond...))
end


function AdaTransformerBlock(
    dim::Int, cond_dim::Int, args...; kws...
)
    return TransformerBlock(
        dim, args...; kws...,
        attention_norm = AdaLN(dim, cond_dim),
        ffn_norm = AdaLN(dim, cond_dim),
    )
end


@concrete struct STRINGBlock
    block
    rope
end

@layer STRINGBlock

function STRINGBlock(block::TransformerBlock, d_coords::Int; kws...)
    rope = STRINGRoPE(block.attention.head_dim, block.attention.n_heads, d_coords; kws...)
    return STRINGBlock(block, rope)
end

function (layer::STRINGBlock)(args...; positions, kws...)
    return layer.block(args...; rope=x->layer.rope(x, positions), kws...)
end
