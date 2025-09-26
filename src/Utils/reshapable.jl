reshapable(x::AbstractArray) = restructure(x, x)

function hardreshape(x, args...)
    y = reshape(reshapable(x), args...)
    y isa Base.ReshapedArray && throw(ArgumentError("Result of `$hardreshape(::$(typeof(x)), ...)` is a `$(Base.ReshapedArray)`. Please define `$reshapable(::$(typeof(x)))`."))
    return y
end
