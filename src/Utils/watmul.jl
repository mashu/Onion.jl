function watmul(W::DenseArray{T,3}, x::DenseArray{T}) where T
    x′ = rearrange(x,  einops"(s1 k) ... -> s1 k ..."; k=size(W, 3))
    y′ = einsum(W, x′, einops"s2 s1 k, s1 k ... -> s2 k ...")
    y  = rearrange(y′, einops"s2 k ... -> (s2 k) ...")
    return y::typeof(x)
end

const ⨝ = watmul

function watmul_mix(
    W::AbstractArray{T,3}, x₁::AbstractArray{T};
    W_in::AbstractMatrix{T}
) where T
    x₁′ = rearrange(x₁, einops"(s1 k1) ... -> s1 k1 ..."; k1=size(W_in, 1))
    x₂′ = einsum(x₁′, W_in, einops"s1 k1 ..., k1 k2 -> s1 k2 ...")
    y₂′ = einsum(W, x₂′, einops"s2 s1 k2, s1 k2 ... -> s2 k2 ...")
    y₂ = rearrange(y₂′, einops"s2 k2 ... -> (s2 k2) ...")
    return y₂
end
