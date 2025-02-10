struct AdaLN{A,B,C}
    norm::A
    shift::B
    scale::C  
end
Flux.@layer AdaLN
AdaLN(dim::Int, cond_dim::Int) = AdaLN(LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))
(l::AdaLN)(x, cond) = l.norm(x) .* (1 .+ glut(l.scale(cond), ndims(x), 1)) .+ glut(l.shift(cond), ndims(x), 1)

#ALN = AdaLN(5, 3)
#ALN(randn(5,10,1), randn(3,1))