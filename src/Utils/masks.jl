function pad_lengths(lengths::AbstractArray{Int}, max_length=maximum(lengths))
    padmask = zeros_like(lengths, Bool, max_length, size(lengths)...)
    for (l, i) in zip(lengths, CartesianIndices(lengths))
        padmask[1:l, i] .= true
    end
    return padmask
end

function causal_mask(x::AbstractArray{<:AbstractFloat})
    n = size(x, 2)
    mask = like(-Inf, x, n, n)
    tril!(mask, -1) # This is swapped because we're using the slightly more efficient dim setup
    return mask
end

@non_differentiable pad_lengths(::Any...)
@non_differentiable causal_mask(::Any...)
