"""
    ⊞(xs...)
    a ⊞ b

Adds the arguments together, ignoring `Nothing`s.

"⊞" can be typed by `\\boxplus<tab>`

# Examples

```jldoctest
julia> using Onion.Utils

julia> 1 ⊞ 2
3

julia> 1 ⊞ nothing
1

julia> (rand(Float32, 10) .⊞ nothing) isa Vector{Float32}
true
```
"""
⊞(args...) = +(filter(!isnothing, args)...)
