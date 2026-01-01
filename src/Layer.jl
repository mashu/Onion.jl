using Flux: @layer

abstract type Layer end
@layer Layer

function input_size end
function output_size end


abstract type LayerStyle end
struct EagerStyle <: LayerStyle end
struct FusedStyle <: LayerStyle end

function fuse end

LayerStyle(::Type{<:Layer}) = EagerStyle()
LayerStyle(::T) where T<:Layer = LayerStyle(T)

# always get an array; use `fuse` method if layer uses FusedStyle
# AND does not define a normal call method 
function (layer::Layer)(args...; kws...)
    LayerStyle(layer) isa EagerStyle && throw(MethodError(layer, args))
    return materialize(fuse(layer, args...; kws...))
end

# get a lazy object if fuse is defined, otherwise array
function fuse(layer::Layer, args...; kws...)
    LayerStyle(layer) isa FusedStyle && throw(MethodError(fuse, (layer, args...)))
    return layer(args...; kws...)
end


const UNICODE_PROPERTY_MAP = (
    :α => :alpha,   :β => :beta,    :γ => :gamma,
    :δ => :delta,   :ε => :epsilon, :ζ => :zeta,
    :η => :eta,     :θ => :theta,   :ι => :iota,
    :κ => :kappa,   :λ => :lambda,  :μ => :mu,
    :ν => :nu,      :ξ => :xi,      :π => :pi,
    :ρ => :rho,     :σ => :sigma,   :τ => :tau,
    :υ => :upsilon, :φ => :phi,     :χ => :chi,
    :ψ => :psi,     :ω => :omega,
)

function _getproperty(layer::T, s::Symbol, ::Val{:name}) where T<:Layer
    i = findfirst(==(s), first.(UNICODE_PROPERTY_MAP))
    unicode = i isa Int ?
        last(UNICODE_PROPERTY_MAP[i]) : nothing
    unicode isa Symbol ?
        getfield(layer, unicode) : throw(FieldError(T, s))
end

function _getproperty(layer::T, s::Symbol, ::Val{:unicode}) where T<:Layer
    i = findfirst(==(s), last.(UNICODE_PROPERTY_MAP))
    name = i isa Int ?
        first(UNICODE_PROPERTY_MAP[i]) : nothing
    name isa Symbol ?
        getfield(layer, name) : _getproperty(layer, s, Val(:name))
end

function Base.getproperty(layer::T, s::Symbol) where T<:Layer
    hasfield(T, s) ?
        getfield(layer, s) : _getproperty(layer, s, Val(:unicode))
end
