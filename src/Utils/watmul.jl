function watmul(W::AbstractArray{T,3}, x::AbstractArray{T}; r=1) where T
    x′ = rearrange(x,  einops"(s1 k r) ... -> s1 k r ..."; k=size(W, 3), r)
    y′ = einsum(W, x′, einops"s2 s1 k, s1 k r ... -> s2 k r ...")::typeof(x′)
    y  = rearrange(y′, einops"s2 k r ... -> (s2 k r) ...")
    return y
end

const ⨝ = watmul
