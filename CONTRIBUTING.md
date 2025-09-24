# Contributing

We welcome contributions to Onion.jl. Please follow these guidelines when contributing.

## Guidelines

- Layers should be structs with concrete field types.
  - They do not necessarily have to be defined with [ConcreteStructs.jl](https://github.com/jonniedie/ConcreteStructs.jl).
- Low-level and performance-critical methods should have statically inferrable return types (Tip: use `@code_warntype` or `Test.@inferred`).
- Layers should preferrably work with any number (including zero) batch dims (Tip: use e.g. [Einops.jl](https://github.com/MurrellGroup/Einops.jl), `Onion.glut`)
- Layer implementations should be put into a separate file inside a relevant subdirectory of the src directory.
- Write docstrings for all user-facing functions.
  - Helper functions should only have comments.
- Cite sources. If from a paper/code include a link/ref in a comment or docstring.
