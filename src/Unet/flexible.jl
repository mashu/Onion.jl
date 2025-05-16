"""
    reverse_tuple(t::Tuple)

Helper function that reverses the order of elements in a tuple.
Use for maintaining type stability when reversing the order of skip connections.
"""
@generated function reverse_tuple(t::Tuple)
    N = length(t.parameters)
    args = [:(t[$i]) for i in N:-1:1]
    return :(tuple($(args...)))
end

"""
    apply_custom_bottleneck(x, t, custom_bottleneck)

Type-stable function to apply a custom bottleneck in FlexibleUNet.
This eliminates the conditional branching that causes type instability.
"""
function apply_custom_bottleneck(x::T, t, custom_bottleneck::F) where {T <: AbstractArray, F}
    return custom_bottleneck(x, t)::T
end

# Special case for identity bottleneck
function apply_custom_bottleneck(x::T, t, custom_bottleneck::typeof(identity)) where {T <: AbstractArray}
    return x  # identity returns x unchanged
end

# For bottlenecks that explicitly don't use time conditioning, add specific methods
# Example:
# function apply_custom_bottleneck(x::AbstractArray{T,N}, t, custom_bottleneck::YourCustomType) where {T,N}
#     return custom_bottleneck(x)::AbstractArray{T,N}
# end

"""
    FlexibleUNet(; in_channels=3, out_channels=3, base_channels=64, channel_multipliers=[1, 2, 4], ...)

A flexible UNet architecture with configurable depth and channel dimensions.
Supports optional time and class embeddings for diffusion models and conditional generation.
The network depth is determined by the length of the channel_multipliers array.
Supports an optional custom_bottleneck that gets applied after the standard bottleneck.
"""
struct FlexibleUNet{E,B,CB,D,FC,T}
    encoders::E
    bottleneck::B
    custom_bottleneck::CB
    decoders::D
    final_conv::FC
    time_embed::T
end

Flux.@layer FlexibleUNet

function FlexibleUNet(;
    in_channels=3,
    out_channels=3,
    base_channels=64,
    channel_multipliers=[1, 2, 4],
    time_embedding=false,
    num_classes=0,
    embedding_dim=128,
    time_emb_dim=64,
    dropout=0.0,
    dropout_depth=0,
    activation=relu,
    custom_bottleneck=identity
)
    depth = length(channel_multipliers)
    channels = [base_channels * m for m in channel_multipliers]
    dropout_depth = min(dropout_depth, 1 + depth)
    dropout_bottleneck = dropout_depth >= 1 ? dropout : 0.0

    encoders = []
    input_ch = in_channels
    for i in 1:depth
        encoder_dropout = dropout_depth >= (depth - i + 1) ? dropout : 0.0
        encoder = EncoderBlock(input_ch, channels[i],
                             time_emb=time_embedding,
                             emb_dim=time_emb_dim,
                             dropout=encoder_dropout,
                             activation=activation)
        push!(encoders, encoder)
        input_ch = channels[i]
    end

    bottleneck = Bottleneck(channels[end],
                          time_emb=time_embedding,
                          emb_dim=time_emb_dim,
                          dropout=dropout_bottleneck,
                          activation=activation)

    decoders = []
    for i in depth:-1:1
        out_ch = i > 1 ? channels[i-1] : channels[1]
        decoder_dropout = dropout_depth >= (depth - i + 1) ? dropout : 0.0
        decoder = DecoderBlock(channels[i], out_ch,
                             time_emb=time_embedding,
                             emb_dim=time_emb_dim,
                             dropout=decoder_dropout,
                             activation=activation)
        push!(decoders, decoder)
    end

    final_conv = Conv((1, 1), channels[1]=>out_channels)
    time_embed = time_embedding ?
                TimeEmbedding(time_emb_dim, num_classes, embedding_dim) : identity

    encoders_tuple = Tuple(encoders)
    decoders_tuple = Tuple(decoders)

    FlexibleUNet(encoders_tuple, bottleneck, custom_bottleneck, decoders_tuple, final_conv, time_embed)
end

"""
    process_encoders(x, encoders::Tuple)

Process encoders and collect skip connections without mutations.
"""
function process_encoders(x, encoders::Tuple)
    (final_x, skips) = foldl(encoders; init=(x, ())) do acc, encoder
        (curr_x, skip_connections) = acc
        skip, new_x = encoder(curr_x)
        (new_x, (skip_connections..., skip))
    end

    return final_x, skips
end

"""
    process_encoders(x, t, encoders::Tuple)

Process encoders with time embedding and collect skip connections without mutations.
"""
function process_encoders(x, t, encoders::Tuple)
    (final_x, skips) = foldl(encoders; init=(x, ())) do acc, encoder
        (curr_x, skip_connections) = acc
        skip, new_x = encoder(curr_x, t)
        (new_x, (skip_connections..., skip))
    end

    return final_x, skips
end

"""
    process_decoders(x, decoders::Tuple, skip_connections::Tuple)

Process decoders using skip connections without mutations.
"""
function process_decoders(x, decoders::Tuple, skip_connections::Tuple)
    if isempty(decoders) || isempty(skip_connections)
        return x
    end

    return foldl(zip(decoders, skip_connections); init=x) do acc_x, (decoder, skip)
        decoder(acc_x, skip)
    end
end

"""
    process_decoders(x, t, decoders::Tuple, skip_connections::Tuple)

Process decoders with time embedding and skip connections without mutations.
"""
function process_decoders(x::T, t, decoders::Tuple, skip_connections::Tuple) where {T <: AbstractArray}
    if isempty(decoders) || isempty(skip_connections)
        return x
    end

    return foldl(zip(decoders, skip_connections); init=x) do acc_x, (decoder, skip)
        decoder(acc_x, skip, t)
    end
end

# Standard forward pass without time embedding
function (model::FlexibleUNet)(x)
    x, skip_connections = process_encoders(x, model.encoders)
    x = model.bottleneck(x)
    x = model.custom_bottleneck(x)
    rev_skips = reverse_tuple(skip_connections)
    x = process_decoders(x, model.decoders, rev_skips)
    model.final_conv(x)
end

# Forward pass with time embedding
function (model::FlexibleUNet)(x::T, t::AbstractVector) where {T <: AbstractArray}
    t = model.time_embed(t)
    x, skip_connections = process_encoders(x, t, model.encoders)
    x = model.bottleneck(x, t)

    # Keep type information with explicit typing
    x = apply_custom_bottleneck(x, t, model.custom_bottleneck)

    rev_skips = reverse_tuple(skip_connections)
    x = process_decoders(x, t, model.decoders, rev_skips)
    model.final_conv(x)
end

# Forward pass with time embedding and class labels
function (model::FlexibleUNet)(x::T, t::AbstractVector, labels::AbstractVector) where {T <: AbstractArray}
    t = model.time_embed(t, labels)
    x, skip_connections = process_encoders(x, t, model.encoders)
    x = model.bottleneck(x, t)

    # Keep type information with explicit typing
    x = apply_custom_bottleneck(x, t, model.custom_bottleneck)

    rev_skips = reverse_tuple(skip_connections)
    x = process_decoders(x, t, model.decoders, rev_skips)
    model.final_conv(x)
end 