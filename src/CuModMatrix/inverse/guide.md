# Combined HPDC×ICCS PLUQ over a finite field on GPU (CuModMatrix)

This report formalizes a **block-recursive PLUQ** factorization for a large dense matrix over a finite field, using:

- **HPDC’14** for the _block-recursive structure_ (“recursive block LU decomposition” and Schur complement recursion).
- **ICCS’17** for a _high-performance base-case LU kernel_ for very small matrices, plus the key trick of **delaying swaps (“lazy swap”)** to reduce overhead.
- The GPU package already has a fast modular GEMM / arithmetic via `CuModMatrix` (your assumption).

The main mathematical correction to keep in mind (and the papers strongly suggest this): **a large LU/PLUQ does not equal LU of all blocks independently**. HPDC’s method factors a diagonal block, then uses **triangular solves** and a **Schur complement** update, then recurses.

---

## 1. Mathematical foundations

### 1.1 Field and notation

Let (\mathbb{F}=\mathbb{F}\_p) be a prime field. All operations are in (\mathbb{F}).

- A **permutation matrix** (P) corresponds to a permutation vector (\pi) such that (P e*i = e*{\pi(i)}).
- For column permutations, (Q) corresponds to a permutation vector (\sigma) such that (Q^T e*j = e*{\sigma(j)}) (equivalently, (AQ) permutes the columns of (A)).

### 1.2 Goal: PLUQ

A **PLUQ decomposition** of (A\in \mathbb{F}^{n\times n}) (not necessarily full rank) is:
[
P,A,Q = L,U
]
where:

- (P,Q) are permutation matrices,
- (L) is (block) unit lower triangular (or lower trapezoidal in rank-deficient case),
- (U) is upper triangular (or upper trapezoidal),
- and the product matches exactly over (\mathbb{F}).

For full-rank square (A), (L) and (U) can be taken as (n\times n) triangular with ones on (\mathrm{diag}(L)).

### 1.3 Pivoting over a finite field

ICCS’17 uses partial pivoting by max-abs in a column (IDAMAX) for floating point.
Over (\mathbb{F}\_p), “max abs” is meaningless. You replace it with a **nonzero pivot strategy**, typically one of:

- **First-nonzero** in the active column (fast, deterministic).
- **Random-nonzero** in the active column (good to avoid worst-cases).
- **Column swap when column is all-zero** (this is where (Q) becomes essential).

To get **PLUQ**, you permit both row and column swaps so that each elimination step finds a nonzero pivot in the remaining submatrix when possible.

---

## 2. HPDC-style block recursion, extended to include both P and Q

### 2.1 Block partition

At some recursion node, assume we have an active submatrix (A) of size (n\times n). Split into quadrants:
[
A =
\begin{pmatrix}
A_{11} & A_{12}\
A_{21} & A_{22}
\end{pmatrix},
\quad
A_{11}\in\mathbb{F}^{k\times k},\ A_{22}\in\mathbb{F}^{(n-k)\times(n-k)}.
]

HPDC’s recursive LU uses exactly this pattern (they describe “recursive block LU decomposition” and Schur complement recursion).

### 2.2 Local PLUQ on the diagonal block

Compute a PLUQ on the diagonal block (A*{11}):
[
P_1,A*{11},Q*1 = L*{11},U*{11}
]
where (P_1,Q_1) are (k\times k) permutations, (L*{11}) unit lower triangular, (U\_{11}) upper triangular (or trapezoidal if rank-deficient).

> This is the one place where you can directly drop in an ICCS-style “tiny LU” kernel when (k) is small.

### 2.3 Threading the permutations into the coupled blocks

Row/column permutations on (A\_{11}) must be applied consistently to the blocks that share those rows/columns:
[
\widetilde{A}*{12} := P_1 A*{12},
\qquad
\widetilde{A}*{21} := A*{21} Q_1.
]

This is the key “(Q) coupling”: if you permute columns of (A\_{11}), you must also permute the same columns in the block row underneath.

### 2.4 Off-diagonal blocks via triangular solves

HPDC’s equations compute off-diagonal blocks via triangular systems (their (L*1U_2=P_1A_2) and (L_2' U_1 = A_3)).
In PLUQ form:
[
\boxed{L*{11},U*{12} = \widetilde{A}*{12}}
\qquad
\boxed{L*{21},U*{11} = \widetilde{A}\_{21}}
]
so:

- (U*{12} = L*{11}^{-1}\widetilde{A}\_{12}) (left solve / TRSM-like)
- (L\_{21} = \widetilde{A}_{21}U_{11}^{-1}) (right solve / TRSM-like)

### 2.5 Schur complement and recursion

Define the Schur complement:
[
S := A_{22} - L_{21}U_{12}.
]
HPDC explicitly recurses on (A_4 - L_2' U_2).

Now recurse on (S):
[
P_2,S,Q_2 = L_{22},U_{22}.
]

### 2.6 Assembling global factors

Define block-diagonal permutations:
[
P :=
\begin{pmatrix}
P_1 & 0\
0 & P_2
\end{pmatrix},
\qquad
Q :=
\begin{pmatrix}
Q_1 & 0\
0 & Q_2
\end{pmatrix}.
]

Define:
[
L :=
\begin{pmatrix}
L_{11} & 0\
L_{21} & L_{22}
\end{pmatrix},
\qquad
U :=
\begin{pmatrix}
U_{11} & U_{12}\
0 & U_{22}
\end{pmatrix}.
]

Then the block-recursive identity holds:
[
\boxed{P,A,Q = L,U}.
]

---

## 3. Correctness proof (clean induction)

### Theorem

Assume:

1. The recursive calls return correct PLUQ factorizations on (A\_{11}) and on (S).
2. (U*{11}) and (L*{11}) are invertible on their triangular structure for the pivots chosen (full-rank case; see rank-deficient note below).

Then the assembled (P,Q,L,U) satisfy (P A Q = L U).

### Proof (block algebra)

Start with:
[
P_1 A_{11} Q_1 = L_{11}U_{11}.
]
Apply the same permutations to full (A):
[
\begin{pmatrix}
P*1 & 0\
0 & I
\end{pmatrix}
\begin{pmatrix}
A*{11} & A*{12}\
A*{21} & A\_{22}
\end{pmatrix}
\begin{pmatrix}
Q_1 & 0\
0 & I
\end{pmatrix}
=============

\begin{pmatrix}
P*1A*{11}Q*1 & P_1A*{12}\
A*{21}Q_1 & A*{22}
\end{pmatrix}
=============

\begin{pmatrix}
L*{11}U*{11} & \widetilde A*{12}\
\widetilde A*{21} & A\_{22}
\end{pmatrix}.
]

Define (U*{12}) and (L*{21}) as the triangular-solve solutions:
[
L_{11}U_{12}=\widetilde A_{12},\quad L_{21}U_{11}=\widetilde A_{21}.
]
Then:
[
\begin{pmatrix}
L*{11} & 0\
L*{21} & I
\end{pmatrix}
\begin{pmatrix}
U*{11} & U*{12}\
0 & A*{22}-L*{21}U\_{12}
\end{pmatrix}
=============

\begin{pmatrix}
L*{11}U*{11} & L*{11}U*{12}\
L*{21}U*{11} & L*{21}U*{12}+A*{22}-L*{21}U\_{12}
\end{pmatrix}
=============

\begin{pmatrix}
L*{11}U*{11} & \widetilde A*{12}\
\widetilde A*{21} & A\_{22}
\end{pmatrix}.
]

So, after this elimination, the bottom-right block becomes (S = A*{22}-L*{21}U*{12}). By the inductive hypothesis on (S), there exist (P_2,Q_2,L*{22},U*{22}) with (P_2 S Q_2 = L*{22}U\_{22}). Insert them into block-diagonal permutations and factors to obtain the final (P,Q,L,U) and the identity (P A Q = L U).

∎

### Rank-deficient note

If (A) is singular over (\mathbb{F}\_p), the algorithm continues until no nonzero pivot exists in the remaining submatrix. Then one obtains a **rank-revealing PLUQ** where (U) has a zero diagonal tail and (L,U) are trapezoidal. This is standard for PLUQ; implementation-wise you return the rank and stop pivoting past rank.

---

## 4. How ICCS’ “lazy swap” fits into the combined method

ICCS emphasizes that intermediate row swaps are expensive; they **delay** swaps and write each row directly to its final destination (“lazy swap”), producing the same LU result.
This idea is extremely useful in the combined method at two levels:

1. **Inside the base-case diagonal block factorization**
   Return pivot vectors (p) (rows) (ICCS stores pivots as IPIV).
   Perform swaps logically, only materializing at the end of the kernel.

2. **For global/tiled permutations**
   Maintaining (P) and (Q) as **permutation vectors** and applying them lazily to tiles avoids strided device memory movement (especially for (Q), which is otherwise painful in column-major storage).

---

# 5. CUDA.jl implementation plan for a modular Julia package (CuModMatrix)

Assumptions:

- `CuModMatrix{T,p}` (or similar) stores a dense matrix on GPU and supports fast modular GEMM and elementwise ops.
- You can launch custom CUDA kernels via CUDA.jl.
- You want a package-quality design: clean APIs, testable pieces, minimal cross-coupling.

## 5.1 Package structure (modules)

**Package name (suggestion):** `CuModPLUQ.jl`

### Modules

1. `CuModPLUQ.Types`
   - `CuModMatrix` integration
   - `PermVec` type for permutation vectors on device + host mirror
   - `PLUQFactorization` result struct

2. `CuModPLUQ.Pivoting`
   - pivot selection policies for (\mathbb{F}\_p):
     - `FirstNonzeroPivot`
     - `RandomNonzeroPivot`
     - `ColumnSearchPivot` (search over remaining columns; produces Q)

3. `CuModPLUQ.BaseCase`
   - ICCS-style tiny-matrix PLUQ kernel(s) for sizes up to a threshold (e.g., 32 or 64)
   - returns local factors + local pivots
   - adopts “lazy swap” internally

4. `CuModPLUQ.Triangular`
   - modular TRSM-like routines specialized to:
     - `solve_left_lower_unit!(X, L, B)` (solve (LX=B))
     - `solve_right_upper!(X, U, B)` (solve (XU=B))

   - if your matrix-mul is strong, you can implement TRSM recursively too, but TRSM kernels are usually worthwhile

5. `CuModPLUQ.Blocked`
   - the HPDC-style blocked recursion / iteration:
     - panel factorization of diagonal tile
     - apply pivots to coupled tiles (logically or physically)
     - triangular solves for (U*{12}), (L*{21})
     - Schur complement update (GEMM)

6. `CuModPLUQ.Tests`
   - correctness tests: compare (P A Q) to (L U) mod (p)
   - rank-deficient cases
   - randomized matrices

---

## 5.2 Core public API

### `pluq(A; blocksize, basecase, pivotpolicy, return_rank)`

**Input:** `A::CuModMatrix` (square)
**Output:** `PLUQFactorization` with fields:

- `L::CuModMatrix` (or compact storage inside overwritten `A`)
- `U::CuModMatrix`
- `p::Vector{Int}` row permutation (host) + optionally device copy
- `q::Vector{Int}` col permutation
- `rank::Int` (optional)

Recommended storage: LAPACK-style compact `LU` in one matrix plus separate diagonal metadata; but since you asked for clean modular planning, either is fine. If `CuModMatrix` is fast for arithmetic, compact LU reduces memory traffic.

---

## 5.3 Mathematical ↔ method mapping (what each method computes)

### Method 1: `base_pluq!(A11) -> (LU11, p1, q1, rank1)`

**Math:** (P*1 A*{11} Q*1 = L*{11}U*{11})
**GPU:** ICCS-style: keep (A*{11}) in registers/shared; delayed swaps; output LU packed + pivot vectors.

**Outputs**

- `LU11`: packed storage of (L*{11}) (unit lower) and (U*{11}) (upper)
- `p1`, `q1`: pivot vectors (length (k))
- `rank1`

**Keywords / tricks**

- register blocking vs shared memory tradeoff (ICCS discusses both; shared makes swaps easy).
- warp shuffle for pivot search / reductions (if doing register layout)
- “lazy swap”: compute final destination and write once

### Method 2: `apply_row_perm_tiles!(A12, p1)` and `apply_col_perm_tiles!(A21, q1)`

**Math:** (\widetilde A*{12}=P_1A*{12}), (\widetilde A*{21}=A*{21}Q_1)

**GPU options**

- **Lazy application:** don’t move data; represent the active tile-row/column through an index map and apply it in kernels that read tiles.
- **Materialized application:** physically permute rows in (A*{12}) tiles (coalesced), columns in (A*{21}) tiles (not coalesced in column-major unless you use a tiled layout).

Smart approach for Julia/CUDA: **materialize row swaps** (fast), **lazy column swaps** (often best), unless you store in tile-major layout.

### Method 3: `trsm_left_lower_unit!(U12, L11, A12_tilde)`

**Math:** solve (L*{11}U*{12}=\widetilde A\_{12})

**GPU tricks**

- if (k) is small-ish (basecase), you can do this inside the same kernel (fusion).
- otherwise TRSM kernel; or recursive TRSM using GEMM.

### Method 4: `trsm_right_upper!(L21, U11, A21_tilde)`

**Math:** solve (L*{21}U*{11}=\widetilde A\_{21})

Same ideas as above.

### Method 5: `schur_update!(A22, L21, U12)`

**Math:** (S = A*{22} - L*{21}U\_{12})

This is your **money kernel**: GEMM mod (p). HPDC explicitly centers the recursion around producing and reusing this Schur complement.

### Method 6: `recurse!(S)` or iterate over panels

**Math:** (P*2 S Q_2 = L*{22}U\_{22})

---

## 5.4 Overall blocked algorithm (device-side view)

Given `A` and a blocksize (k):

For (t=1,2,\dots) over diagonal blocks:

1. Extract view `A11 = A[t:t+k-1, t:t+k-1]`
2. `(LU11, p1, q1, r1) = base_pluq!(A11)` (ICCS-style)
3. Apply permutations to the coupled blocks:
   - `A12 = A[t:t+k-1, t+k:n]` gets row perm `p1`
   - `A21 = A[t+k:n, t:t+k-1]` gets col perm `q1`

4. `U12 = L11^{-1} * (P1*A12)`
5. `L21 = (A21*Q1) * U11^{-1}`
6. `A22 -= L21 * U12`
7. Continue on `A22` (next panel / recursion)

In recursion form, steps 1–6 are exactly HPDC’s “block computation style” and recursion on Schur complement.

---

## 5.5 What each component should output (clear contracts)

### `base_pluq!`

- packed `LU11` in-place in `A11`
- pivot vectors `p1`, `q1`
- `rank1` + a flag if singular pivot encountered

### `blocked_pluq!`

- overwrites `A` with packed `LU`
- returns global `p`, `q` (length (n))
- returns `rank`
- optionally returns explicit `L,U` matrices (but avoid if performance matters)

### `extract_LU(LU, p, q)`

- materializes `L` and `U` if needed for downstream tasks, otherwise keep compact

---

## 5.6 “Smart GPU” checklist (concrete keywords to lean on)

- **Tile / block layout** for coalesced GEMM and reduced permutation pain (HPDC discusses tiled layouts as important to locality in other systems; their own partitioning is explicitly layout-oriented).
- **Lazy swap / delayed permutation** (ICCS) to avoid repeated row swaps.
- **Permutation vectors not matrices** (ICCS stores (P) via pivot vector IPIV).
- **Kernel fusion** for small blocks: factor + partial solves in one pass (ICCS fuses work in inversion via augmented matrix handling).
- **Register blocking / shared memory staging** tradeoffs (ICCS details this design choice).
- **Avoid global memory traffic**: “read/write once” is the ideal for basecases.
- **Minimize column permutations materialization** (column swaps are strided in column-major); prefer lazy (Q) or tile-major format.

---

# 6. Small end-to-end test spec (inputs/outputs)

### Test 1: Full-rank random matrix

- Choose prime (p) (e.g., 101).
- Generate random (A\in \mathbb{F}\_p^{8\times 8}) on host, copy to `CuModMatrix`.
- Run `F = pluq(A; blocksize=4, basecase=32, pivotpolicy=FirstNonzeroPivot())`.
- Check on GPU (or copy back):
  [
  (P A Q) \bmod p \stackrel{?}{=} (L U)\bmod p
  ]
  where `P` and `Q` are applied using `p` and `q` vectors.

**Expected output shape**

- `p`, `q` are permutations of `1:8`
- `rank = 8`
- `LU` packed: strictly lower part is multipliers, diagonal+upper is U.

### Test 2: Rank-deficient

Take (A) with two equal rows over (\mathbb{F}\_p). Expect:

- `rank = 7` (or less depending on construction)
- Correctness check still holds with trapezoidal (L,U) (packed LU still works; diagonal of (U) will contain zeros after rank).

---

# 7. Practical scope boundaries (so the plan stays non-duct-taped)

1. **Basecase PLUQ on GPU over (\mathbb{F}\_p)** is doable and matches ICCS’s style, but you must implement modular inverses and a nonzero-pivot search. ICCS’s exact IDAMAX logic is not reusable as-is.
2. For large matrices, performance will be dominated by `schur_update!` (modular GEMM), which you already have fast in `CuModMatrix`. This is exactly why the HPDC recursion is attractive: it organizes the computation around Schur complements.
3. The highest leverage “new” engineering is **permutation handling** (especially (Q)): use lazy application or tile-major storage.

---

If you want one extra deliverable: I can write the exact **permutation-composition formulas** you’ll need to lift each local ((p_1,q_1)) at each block into the global ((p,q)) (so you can maintain (P,Q) without ever building permutation matrices), and I’ll express it as precise vector operations that map directly to CUDA kernels.

## 1) Permutations as vectors: exact formulas you can implement

We represent permutations by **index vectors** (length (n)).

### 1.1 Row permutation vectors

A row permutation vector (p\in [n]^n) encodes a permutation matrix (P) such that
[
(PA)_{i,j} = A_{p[i],,j}.
]
Equivalently, (P) “gathers” rows of (A) in the order (p).

### 1.2 Column permutation vectors

A column permutation vector (q\in [n]^n) encodes (Q) such that
[
(AQ)_{i,j} = A_{i,,q[j]}.
]
So (AQ) gathers columns of (A) in the order (q).

### 1.3 Composition rules (critical)

If you apply (p*1) then (p_2) on the left:
[
P_2(P_1A) = (P_2P_1)A,
]
then the composed gather vector is:
[
(p_2\circ p_1)[i] = p_1[p_2[i]].
]
Reason:
[
(P_2(P_1A))*{i,j}=(P_1A)_{p_2[i],j} = A_{p_1[p_2[i]],j}.
]

Similarly on the right for columns:
[
(AQ_1)Q_2 = A(Q_1Q_2),
]
and the composed gather vector is:
[
(q*1\circ q_2)[j] = q_1[q_2[j]].
]
Reason:
[
((AQ_1)Q_2)*{i,j}=(AQ_1)_{i,q_2[j]}=A_{i,q_1[q_2[j]]}.
]

These two identities are the “never get wrong” rules for maintaining global (p,q).

---

## 2) Local (block) pivots lifted to global permutations

At a given blocked step (t) (0-based) with block size (b), define the active index set
[
I = {t, t+1, \dots, t+b-1}.
]
(Use 1-based indices in Julia; I’ll write math 0-based for clarity—convert mechanically.)

Your base-case PLUQ on the diagonal block returns **local** pivot vectors:

- (p^{(loc)}\in [b]^b) (row gather on the block)
- (q^{(loc)}\in [b]^b) (col gather on the block)

These represent permutation matrices (P_1,Q_1) acting **only inside** the diagonal block.

### 2.1 Lift local pivots to global “swap lists”

Define the lifted global index vectors (length (b)):
[
p^{(g)}[r] = t + p^{(loc)}[r],\qquad q^{(g)}[c] = t + q^{(loc)}[c]
]
(where (p^{(loc)}[r]\in{0,\dots,b-1}) etc).

These (p^{(g)}), (q^{(g)}) are the actual **global row/col indices** involved in the pivoting within the active block.

---

## 3) Two equivalent ways to maintain global (p,q)

You can either maintain:

### Option A (recommended): global gather maps (p,q) for the entire matrix

Initialize:
[
p = [0,1,\dots,n-1],\qquad q=[0,1,\dots,n-1].
]

When you finish block (t), you **update only the active segment** (I) by composition:

#### Row update

[
p[I] \leftarrow p[I]\circ p^{(loc)}
]
meaning:
[
p[t+r] \leftarrow p[t + p^{(loc)}[r]]\quad \text{for } r=0,\dots,b-1.
]
This is exactly (p \leftarrow p \circ \widehat p) where (\widehat p) is identity outside (I) and equals the local block permutation inside.

#### Column update

[
q[I] \leftarrow q[I]\circ q^{(loc)}
]
meaning:
[
q[t+c] \leftarrow q[t + q^{(loc)}[c]]\quad \text{for } c=0,\dots,b-1.
]

This is the cleanest, fastest on GPU: it’s just a small gather kernel for segments.

### Option B: maintain “swap operations” (sequence of transpositions)

ICCS-style implementations often keep swaps to apply lazily. But for PLUQ you’ll likely still want final (p,q) as vectors. You can store transpositions from the basecase and later fold them into a vector using parallel permutation application.

Option A is simpler and already “lazy” (you do not physically permute data unless you choose to).

---

## 4) How the block math uses these permutations (full detail)

At block step (t), you conceptually want:
[
P_1 A_{11} Q_1 = L_{11}U_{11}.
]
But your global factorization is:
[
P A Q = L U.
]

To avoid physically permuting the whole matrix at each pivot, you keep (p,q) and interpret every access to (A) through them (lazy permutations).

### 4.1 Lazy-permuted views

Define the lazily permuted matrix:
[
A^{(pq)}_{i,j} := A_{p[i],,q[j]}.
]
Then “working on the active block” means you work on submatrices of (A^{(pq)}), not (A).

In code terms: a tile load kernel for tile rows (R) and cols (C) reads
[
\text{tile}[r,c] \leftarrow A[p[R_r],,q[C_c]].
]
This avoids ever doing strided physical column swaps.

### 4.2 Block formulas with lazy permutations

Let the active block partition of (A^{(pq)}) be:
[
A^{(pq)} =
\begin{pmatrix}
A_{11} & A_{12}\
A_{21} & A_{22}
\end{pmatrix}
]
(where these are _views_ into (A^{(pq)}) at indices ([t:t+b-1]) etc).

Basecase returns local (p^{(loc)},q^{(loc)}) such that:
[
P_1 A_{11} Q_1 = L_{11}U_{11}.
]

You now update the global permutation vectors **on the active segment**:
[
p[t:t+b-1] \leftarrow p[t:t+b-1]\circ p^{(loc)},\qquad
q[t:t+b-1] \leftarrow q[t:t+b-1]\circ q^{(loc)}.
]
This is exactly the lift of the local pivot to the global (P,Q).

After this update, the _definition_ of (A^{(pq)}) changes (because (p,q) changed), and that automatically applies:
[
\widetilde A_{12} = P_1 A_{12},\qquad
\widetilde A_{21} = A_{21}Q_1
]
without moving data.

Then compute (in (\mathbb{F}_p)):
[
U_{12} = L*{11}^{-1},\widetilde A*{12},\qquad
L*{21} = \widetilde A*{21},U*{11}^{-1},
]
and update:
[
A*{22} \leftarrow A*{22} - L*{21}U\_{12}.
]

That is HPDC’s block recursion, with (Q) handled properly and lazily.

---

## 5) GPU implementation mapping (CuModMatrix)

### 5.1 Data you maintain

- `A::CuModMatrix` (either overwritten in-place with packed LU, or separate buffers)
- `p::CuDeviceVector{Int32}` length (n) (row gather)
- `q::CuDeviceVector{Int32}` length (n) (col gather)
- Temporary tile buffers in shared memory/registers for basecase.

### 5.2 Kernels / methods and exactly what they compute

#### (1) `compose_segment!(p, t, p_loc)`

Implements:
[
p[t+r] \gets p[t+p^{(loc)}[r]]
]
for (r=0..b-1). Same for `q`.

GPU: one block handles one segment; threads load old segment into shared memory then write composed segment. Coalesced.

#### (2) `load_tile_pq!(tile, A, p, q, row_ids, col_ids)`

Implements:
[
\text{tile}[r,c] \gets A[p[row_ids[r]],\ q[col_ids[c]]].
]
This is your “lazy permutation” primitive. Everything else can be written in terms of tile loads/stores.

#### (3) `base_pluq_tile!(tile) -> (tile_LU, p_loc, q_loc, rank)`

ICCS-style kernel on `tile` in shared/registers. Returns local pivot vectors and LU packed in `tile`.

Over (\mathbb{F}\_p):

- pivot search finds a nonzero pivot; if pivot column is all zeros, search a later column ⇒ produces `q_loc` updates.
- scaling uses modular inverse.
- rank-1 update is mod (p).

You can still use ICCS’s “lazy swap” internally: don’t physically swap rows every step; track mapping and write final at end.

#### (4) `trsm_left!` and `trsm_right!`

If you have strong GEMM, consider TRSM via recursive blocking; but a direct TRSM kernel for modular arithmetic is usually needed.

- Left solve: (L*{11}U*{12}=\widetilde A\_{12})
- Right solve: (L*{21}U*{11}=\widetilde A\_{21})

#### (5) `schur_update!`

Uses your fast modular GEMM:
[
A_{22} \gets A_{22} - L_{21}U_{12}.
]
This must dominate runtime.

### 5.3 Output contract

Return a factorization object containing:

- packed `LU` (in `A` or separate)
- `p`, `q` (device + host copy)
- `rank`
  Optionally: methods to apply (P) and (Q) to vectors/matrices without materializing permutation matrices.

---

## 6) Testing: precise identities and how to check them

### 6.1 Construct explicit P and Q on CPU (for tests only)

From gather vectors (p,q), define:
[
(PA)_{i,j}=A_{p[i],j},\quad (AQ)_{i,j}=A_{i,q[j]}.
]
So:
[
(PAQ)_{i,j} = A_{p[i],,q[j]}.
]

### 6.2 Extract (L,U) from packed LU

If `LU` is packed in-place:

- (L) has ones on diagonal and strictly-lower entries from `LU`.
- (U) has diagonal+upper entries from `LU`.

### 6.3 Correctness check

Compute on CPU or GPU:
[
\Delta := (PAQ - LU)\bmod p.
]
Accept if (\Delta) is the zero matrix.

### 6.4 Small deterministic example

Let (p=7), (A\in \mathbb{F}\_7^{4\times 4}) chosen so that first pivot needs a column swap (forces (Q\neq I)), e.g. make first column of the active submatrix zero but some later column nonzero. Then:

- `q` should not be identity
- the identity (PAQ=LU) should still hold
- `rank` should match expected.

---

## 7) Performance sanity checks (non-math but essential)

- Validate that most time is in `schur_update!` (GEMM) for large (n).
- Ensure `compose_segment!` and `load_tile_pq!` are not dominating.
- Confirm that you are not materializing column swaps frequently; prefer lazy `q`.
