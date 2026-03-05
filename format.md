# Julia Quality/Formatting Tooling Guide

This file documents the tooling recently added to `Project.toml` and a lightweight, robust way to enforce their usage in a Julia project.

## Tool Links

- [JET.jl](https://github.com/aviatesk/JET.jl)
- [Aqua.jl](https://github.com/JuliaTesting/Aqua.jl)
- [JuliaFormatter.jl](https://github.com/domluna/JuliaFormatter.jl)
- [StaticLint.jl](https://github.com/julia-vscode/StaticLint.jl)
- [AllocCheck.jl](https://github.com/JuliaLang/AllocCheck.jl)
- [DispatchDoctor.jl](https://github.com/MilesCranmer/DispatchDoctor.jl)

## 1) JET.jl

### What it is
Static analysis based on Julia type inference to detect possible errors and type instability.

### How to run
```julia
using Pkg
Pkg.add("JET")
using JET
```

```julia
@report_opt foldl(+, Any[]; init=0)
@report_call foldl(+, Char[])
```

```julia
using YourPackage
report_package(YourPackage)
report_package(YourPackage; target_modules=(YourPackage,))
```

### Key feature examples
`@report_opt`, `@report_call`, and `report_package` are the core workflows from the JET docs.

## 2) Aqua.jl

### What it is
Automated package quality checks: ambiguities, stale deps, compat coverage, undefined exports, piracy checks, etc.

### How to run
```julia
using Pkg
Pkg.add("Aqua")
using Aqua
using YourPackage

Aqua.test_all(YourPackage)
```

### Key feature examples
`Aqua.test_all(YourPackage)` runs the full QA suite on your package.

## 3) JuliaFormatter.jl

### What it is
Code formatter for Julia files, with API and CLI usage.

### How to run
```julia
using Pkg
Pkg.add("JuliaFormatter")
using JuliaFormatter

format(".")
format_file("src/YourFile.jl")
```

CLI:
```bash
jlfmt --check -v src/
jlfmt --inplace src/file.jl
jlfmt --diff src/file.jl
```

### Key feature examples
Use `format(".")` for project-wide formatting and `jlfmt --check` in CI for enforcement.

## 4) StaticLint.jl

### What it is
Static code analysis engine used by Julia language tooling (for example editor/language server workflows).

### How to run
```julia
using Pkg
Pkg.add("StaticLint")
using StaticLint
```

### Key feature examples
StaticLint is usually integrated through tooling (LanguageServer/editor) rather than as a simple package-level CI command.

## 5) AllocCheck.jl

### What it is
Static allocation analysis based on LLVM IR for a function call and its callees.

### How to run
```julia
using Pkg
Pkg.add("AllocCheck")
using AllocCheck
```

```julia
@check_allocs multiply(x, y) = x * y
multiply(1.5, 2.5)
```

```julia
@check_allocs ignore_throw=false mysin2(x) = sin(x)
mysin2(1.5)
```

### Key feature examples
`@check_allocs` to guard critical routines and `ignore_throw=false` when exception-path allocations matter.

## 6) DispatchDoctor.jl

### What it is
Type-stability enforcement via `@stable` and selective opt-out with `@unstable`.

### How to run
```julia
using Pkg
Pkg.add("DispatchDoctor")
using DispatchDoctor: @stable, @unstable
```

```julia
@stable function relu(x)
    if x > 0
        return x
    else
        return 0.0
    end
end
```

```julia
@stable begin
    f() = rand(Bool) ? 0 : 1.0
    f(x) = x
end
```

### Key feature examples
`@stable` on functions or blocks, with project-level behavior controlled through Preferences.

## Recommended Enforcement in a Julia Project

## Robust + Simple + Lightweight Setup

1. Keep checks in Julia scripts inside `test/`
2. Wrap them with optional `Makefile` targets for developer ergonomics
3. Enforce in CI (required)
4. Optionally add pre-commit hooks for local fast feedback

This keeps the source of truth in Julia, avoids heavy extra infrastructure, and gives both local and CI workflows.

## Suggested commands/scripts

Create quality scripts:

- `test/quality_aqua.jl`
- `test/quality_jet.jl`
- `test/quality_alloc.jl`
- `test/quality_dispatchdoctor.jl`

Run each with:
```bash
julia --project=. test/quality_aqua.jl
julia --project=. test/quality_jet.jl
julia --project=. test/quality_alloc.jl
julia --project=. test/quality_dispatchdoctor.jl
```

Formatting:
```bash
jlfmt --check src/ test/
jlfmt --inplace src/ test/
```

## Optional Makefile wrapper

```makefile
quality:
	julia --project=. test/quality_aqua.jl
	julia --project=. test/quality_jet.jl
	julia --project=. test/quality_alloc.jl
	julia --project=. test/quality_dispatchdoctor.jl

fmt:
	jlfmt --inplace src/ test/

fmt-check:
	jlfmt --check src/ test/
```

## Pros/Cons of Enforcement Options

### A) Julia scripts (`test/*.jl`) as source of truth

Pros:
- Native Julia, no extra toolchain
- Easy to version and review
- Works the same locally and in CI

Cons:
- Requires a little upfront scripting

### B) Makefile wrapper

Pros:
- Very simple UX (`make quality`, `make fmt-check`)
- Easy standardization for contributors

Cons:
- Extra thin layer to maintain
- Less portable for contributors who do not use `make`

### C) CI enforcement (GitHub Actions or similar)

Pros:
- Most robust gate
- Prevents regressions before merge
- Independent of developer local setup

Cons:
- Slower feedback than local checks

### D) Pre-commit hooks

Pros:
- Fast local feedback
- Prevents obvious formatting/lint misses

Cons:
- Can be bypassed
- Adds local setup friction

## Practical Recommendation

Use **B + A + C**, and optionally D:

1. Implement checks in Julia scripts under `test/`
2. Add a tiny `Makefile` wrapper
3. Run all checks in CI as required status checks
4. Add pre-commit hooks only if your team wants stricter local automation

This combination is usually the best balance of robustness, simplicity, and low maintenance for Julia projects.
