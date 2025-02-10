struct AdaLN{A,B,C}
    norm::A
    shift::B
    scale::C  
end
Flux.@layer AdaLN
AdaLN(dim::Int, cond_dim::Int) = AdaLN(LayerNorm(dim), Dense(cond_dim, dim), Dense(cond_dim, dim))
(aln::AdaLN)(x, cond) = aln.norm(x) .* (1 .+ glut(aln.scale(cond), ndims(x), 1)) .+ glut(aln.shift(cond), ndims(x), 1)

#ALN = AdaLN(5, 3)
#ALN(randn(5,10,1), randn(3,1))