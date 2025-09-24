"""
    ShortConvolution(hidden_size::Int, kernel_size::Int;
        bias=false, activation=:silu, init=Flux.glorot_uniform)

Depthwise causal 1D convolution over the sequence/time dimension.

Input/Output shape: `(hidden_size, seq_len, batch...)`.

- Causality: uses left padding of `kernel_size-1`, so no future leakage
- Depthwise: one independent filter per channel
- Activation: `:silu`/`:swish`, `:identity`, or a custom function

# Examples

```julia
layer = ShortConvolution(256, 5)
x = randn(256, 16, 2)
y = layer(x)

# Streaming/step usage
state = ShortConvolution.init_state(layer, batch=2)
for t in axes(x, 2)
    yt, state = ShortConvolution.step(layer, view(x, :, t, :), state)
end
```
"""
@concrete struct ShortConvolution
    weight; bias; σ
    hidden_size::Int
    kernel_size::Int
end

@layer ShortConvolution

silu(x) = x .* σ.(x)

function ShortConvolution(hidden_size::Int, kernel_size::Int;
    bias::Bool=false, σ::Union{Function,Symbol}=silu, init=Flux.glorot_uniform)
    @assert kernel_size ≥ 1
    W = init(hidden_size, kernel_size)
    b = bias ? zeros_like(W, hidden_size) : nothing
    return ShortConvolution(W, b, σ, hidden_size, kernel_size)
end

"""
    (layer::ShortConvolution)(x;
        residual=nothing, mask=nothing)

Apply depthwise causal 1D convolution along dimension 2.
Returns an array of the same shape as `x`.

Arguments:
- `x`: `(D, T, batch...)`
- `mask` (optional): broadcastable to `(1, T, batch...)`; padded positions should be 0/false
- `residual` (optional): added to the output before activation
"""
function (layer::ShortConvolution)(x; residual=nothing, mask=nothing)
    D, T = size(x, 1), size(x, 2)
    @assert D == layer.hidden_size
    x = isnothing(mask) ? x : x .* glut(mask, ndims(x), 0)

    # To (T, D, B)
    x_tdB = permutedims(x, (2, 1, 3))
    # (D, W) → (W, 1, D)
    w_w1d = reshape(permutedims(layer.weight, (2, 1)), (layer.kernel_size, 1, D))

    y_tdB = NNlib.depthwiseconv(x_tdB, w_w1d; pad=(layer.kernel_size - 1, 0))
    y = permutedims(y_tdB, (2, 1, 3))

    y = y .⊞ layer.bias
    y = y .⊞ residual
    layer.σ.(y)
end


conv_state(layer::ShortConvolution; batch::Int=1) =
    zeros_like(layer.weight, layer.hidden_size, layer.kernel_size, batch)

function step(layer::ShortConvolution, xt, state; residual=nothing)
    D, W = layer.hidden_size, layer.kernel_size
    if ndims(xt) == 2
        xt2 = xt
    else
        @assert ndims(xt) == 3 && size(xt, 2) == 1
        xt2 = reshape(xt, D, size(xt, 3))
    end
    B = size(xt2, 2)
    @assert size(state) == (D, W, B)

    # New state without in-place mutation
    new_state = cat(@view(state[:, 2:end, :]), reshape(xt2, D, 1, B); dims=2)
    yt = dropdims(sum(new_state .* reshape(layer.weight, D, W, 1); dims=2), dims=2)
    yt = yt .⊞ layer.bias
    yt = yt .⊞ residual
    σ = layer.σ
    return σ.(yt), new_state
end


