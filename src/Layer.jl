using Flux: @layer

abstract type Layer end
@layer Layer


abstract type LayerStyle end

LayerStyle(::T) where T<:Layer = LayerStyle(T)

apply_with(::S, ::S, ::L, args...; kws...) where {S<:LayerStyle,L<:Layer} = error("$(L.name.name) does not implement $(S)")
apply(style::LayerStyle, layer::Layer, args...; kws...) = apply_with(style, LayerStyle(layer), layer, args...; kws...)

struct EagerStyle <: LayerStyle end
(layer::Layer)(args...; kws...) = apply(EagerStyle(), layer, args...; kws...)
LayerStyle(::Type{<:Layer}) = EagerStyle()

struct FusedStyle <: LayerStyle end
fuse(layer::Layer, args...; kws...) = apply(FusedStyle(), layer, args...; kws...)

apply_with(::EagerStyle, ::FusedStyle, layer::Layer, args...; kws...) = materialize(fuse(layer, args...; kws...))
apply_with(::FusedStyle, ::EagerStyle, layer::Layer, args...; kws...) = layer(args...; kws...)


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
