# Contributing

Follow the [Julia Style Guide](https://docs.julialang.org/en/v1/manual/style-guide/)
when writing new code.

## Formatting

Code is formatted with
[JuliaFormatter.jl](https://domluna.github.io/JuliaFormatter.jl/stable/) using the
settings in the repository's `.JuliaFormatter.toml`. The easiest way to stay
formatted is to enable format-on-save in your editor (the Julia VS Code extension
picks the config up automatically); alternatively run

```julia
using JuliaFormatter
format(".")
```

from the repository root before committing.

## Tests and docstrings

- New test files are plain scripts registered in `test/runtests.jl`.
- Exported functions get docstrings; API documentation is generated from them with
  Documenter.jl.
