# PR #20 review responses

These replies correspond to the inline comment IDs in PR #20. “Accepted” means
the finding is valid and changed on this branch. “Acknowledged” means the
observation is valid, but the present design is intentional and the response
states the bounded contract rather than claiming an unimplemented rewrite.

| Comment | Assessment | Proposed GitHub reply |
|---|---|---|
| `3383915829` | Accepted | Changed the default square inverse strategy to `:pluq` and added explicit correctness coverage for both `:pluq` and `:augmented` across basecase/block boundaries. |
| `3383915830` | Accepted, partially mitigated | Replaced per-iteration host allocation with one reused pinned `Int32` buffer at every pivot-read site. The host dependency still serializes pivots; the benchmark report records this remaining architectural cost instead of claiming the synchronization is gone. |
| `3383915833` | Accepted | Parameterized inverse tests over both strategies at sizes 4, 16, 32, 33, 64, and 129, and added a direct `inverse_pluq_new` check. |
| `3383915838` | Accepted | Added singular square throw tests for both strategies and singular tests for every fixed-size batched inverse wrapper. |
| `3383915841` | Accepted | Added the prime-field precondition and `check_prime=true` guidance to public inverse/PLUQ docstrings and the inverse README. |
| `3383915847` | Accepted | `_resolve_options` now validates the modulus when requested and throws `CuModMatrixModulusNotPrimeException`; tests cover composite rejection and the unchecked compatibility mode. |
| `3383915852` | Accepted | Warp TRSM reductions now remain `Int64` and reduce modulo `N` after every shuffle addition. |
| `3383915853` | Accepted via explicit contract | CUDA inverse kernels now reject moduli above `typemax(Int32)`, and the composed PLUQ inverse also rejects moduli unsafe for its modular matmul. This prevents silent overflow and is covered by tests. |
| `3383915854` | Accepted | A deficient panel now returns before TRSM/Schur, so no zero diagonal is inverted. The regression checks a 64×64 duplicated-row matrix and verifies that PLUQ inverse throws. |
| `3385291140` | Accepted | All public paths resolve options through `_resolve_options`; `is_invertible_new` delegates to `pluq_new`, which performs that resolution. |
| `3385291146` | Acknowledged | The augmented route remains a reference Gauss–Jordan implementation; PLUQ is now the production default. Reworking the reference into forward elimination plus back substitution would create a second triangular-solve pipeline without affecting the default. |
| `3385291148` | Accepted | Public docstrings now state behavior, field preconditions, concrete results, and failure conditions; the inverse README summarizes the contracts. |
| `3385291152` | Accepted | Added tests for non-square `inverse_new` and wrong-orientation right/left inverse calls. |
| `3385291157` | Acknowledged | Canonical `inverse` and `is_invertible` remain compatibility entry points while the explicit experimental PLUQ APIs retain `_new` during this PR. Renaming the full public surface in the same performance/correctness change would be a separate breaking API change. |
| `3385291160` | Acknowledged | The augmented allocation/copy is confined to the explicit reference strategy; default PLUQ does not allocate `[A I]`. A view-backed `CuModMatrix` is not supported by the current padded storage invariant. |
| `3385291164` | Accepted | Scale testing reproduced a launch failure when autotune selected the 32×32 Schur kernel (1024 requested threads versus a 640-thread compiled limit on RTX 3060). Automatic and explicit selection now safely use the portable 16×16 geometry; the benchmark records device and element type. |
| `3385291167` | Accepted as measured limitation | PLUQ is now the default, avoiding the augmented per-column launch sequence. The remaining launch/synchronization costs are called out in the benchmark report and are not hidden behind an unverified fusion rewrite. |
| `3385291170` | Accepted | Added documentation for `inverse_pluq_new` and the fixed-size batched PLUQ/inverse wrapper families, including their shared preconditions and errors. |
| `3385291174` | Acknowledged | The transpose implementation is retained because it shares the tested right-inverse algorithm and is correct. A direct left solve is a distinct performance kernel and requires its own benchmark and tests. |
| `3385291176` | Accepted | Added tests for `pluq_new_batch`, two-stream square inverse fallback, and mixed wide/tall rectangular batch dispatch. |
| `3385291177` | Accepted | Added a validated copy constructor for `PLUQOptions`; autotune branches now specify only their changed fields. |
| `3385291180` | Partly accepted | Documented that `mod_backend` is currently a compatibility preference while kernels select a safe backend from `N`. The large-modulus path is bounded by explicit overflow checks; backend specialization remains a performance follow-up. |
| `3385291182` | Accepted | Rounded the configurable basecase block size up to a power of two (capped at 256), preserving the shared reduction invariant for values such as `nftb=6`. |
| `3385291188` | Acknowledged | The one-thread tiny kernel is correct and targets batch-level parallelism (one block per matrix). Warp-cooperative elimination is a separate performance implementation; current tests cover all four sizes and singular failure. |
| `3385291193` | Acknowledged | The repeated diagonal inverse is a valid optimization opportunity. It is retained until a shared-memory implementation is profiled because panel depth, occupancy, and synchronization trade off differently across modes. |
| `3385291197` | Accepted | Tile-specific kernels remain private and coupled to matching block shapes. Scale testing also found that the 32×32 variant can exceed the compiled thread limit, so dispatch now caps the active tile at 16 rather than launching an invalid configuration. |
| `3385291199` | Acknowledged | The duplication is real, but the tile sizes are compile-time shared-memory shapes in CUDA.jl. Consolidating them requires generated/`Val` kernels and should be isolated from this correctness pass. |
| `3385291202` | Accepted | Expanded `PLUQOptions` documentation to all 15 fields, valid values, defaults, and constructor errors. |
| `3385291205` | Accepted | Added a top-level README inverse/one-sided-inverse bullet linking to the focused inverse README, which lists algorithms, papers, entry points, and the prime-modulus contract. |
| `3385291208` | Accepted | Added `N=2` coverage for both inverse strategies, composite-modulus validation, and explicit overflow-limit behavior. |
| `3385291214` | Accepted in bounded form | Existing regime tests exercise the basecase and 256 boundary; the benchmark specification covers larger regimes. A 1600×1600 correctness product is intentionally kept out of routine CI because it is a multi-gigabyte-work GPU test. |

The submitted review about stray planning Markdown is addressed by `CLEAN.sh`:
it removes the downloaded review data, local plans, and untracked paper copies
only when the maintainer runs it.

The final full-suite run also exposed a PLUQ permutation bug not called out in
the review: for permutation cycles longer than two, inverse composition applied
the stored gather vectors instead of their inverses. The implementation now
uses inverse gather maps and includes the De Rham matrix as a regression.
