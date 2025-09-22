"""
    Modulator(in_dim => out_dim; σ=sigmoid, op=*)

Takes an input `Y` and a conditioning input `X` and applies a gate to `Y` based on `X`.

See [Gated Attention for Large Language Models](https://arxiv.org/pdf/2505.06708)

# Examples

```jldoctest
julia> gate = Modulator(32 => 64);

julia> Y = randn(Float32, 64);

julia> X = randn(Float32, 32);

julia> gate(Y, X) |> size
(64,)
```
"""
@concrete struct Modulator
    W; σ; op
end

Flux.@layer Modulator

function Modulator((in_dim, out_dim)::Pair{Int,Int}; σ=sigmoid, op=*)
    W = Dense(in_dim => out_dim, bias=false)
    return Modulator(W, σ, op)
end

function (m::Modulator)(Y, X)
    W, σ, * = m.W, m.σ, m.op
    Y′ = Y .* σ.(W(X))
    return Y′
end
