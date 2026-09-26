"""Certified lower bounds on the shortest kernel vector, in exact arithmetic.

The float Gram-Schmidt in lattice.py overflows float64 well below the magnitudes
that arise at non-dyadic factors (it emits 'invalid value encountered in multiply'
and returns nonsense), so every bound here is computed over Fraction and converted
to a float only in the final log.  The reported bound is the MINIMUM over windows:
if the smallest window bound already exceeds the 255 an integer pixel offset can
carry, then no window admits a realizable perturbation.
"""
import sys, math, time
import numpy as np
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from fractions import Fraction
from aliasforge.exact import pil_coeffs
from aliasforge.lattice import kernel_basis

def gs_bound_exact(B):
    Bs = []
    for b in B:
        v = [Fraction(x) for x in b]
        for u, uu in Bs:
            if uu == 0: continue
            c = sum(Fraction(x) * y for x, y in zip(b, u)) / uu
            v = [a - c * y for a, y in zip(v, u)]
        Bs.append((v, sum(x * x for x in v)))
    n2 = [uu for _, uu in Bs if uu > 0]
    if not n2: return None
    m2 = min(n2); W = len(B[0])
    return math.exp(0.5 * (math.log(m2.numerator) - math.log(m2.denominator))
                    - 0.5 * math.log(W))

def main():
    N_OUT, W, STRIDE = 128, 32, 16
    FACTORS = [("2x", 2), ("4x", 4), ("8x", 8),
               ("3x", 3), ("6x", 6), ("1.5x", 1.5), ("2.5x", 2.5), ("1.41x", 181/128)]

    print(f"config: n_out={N_OUT}, window W={W}, stride {STRIDE}, Pillow bicubic fixed-point kernel")
    print(f"{'factor':7s} {'n_in':>6s} {'windows':>8s} {'certified LB (min)':>19s} {'shortest found':>15s} {'realizable':>11s}")
    for name, f in FACTORS:
        t0 = time.time()
        n_in = int(round(N_OUT * f))
        K = pil_coeffs(n_in, N_OUT)
        lbs, sh = [], []
        for c0 in range(0, max(1, n_in - W), STRIDE):
            B = kernel_basis(K, c0, W)
            if not B: continue
            b = gs_bound_exact(B)
            if b is not None: lbs.append(b)
            sh.append(min(max(abs(x) for x in v) for v in B if any(v)))
        if not lbs:
            print(f"{name:7s} {n_in:6d} {'--':>8s}   (kernel empty in every window)"); continue
        lo, s0 = min(lbs), min(sh)
        print(f"{name:7s} {n_in:6d} {len(lbs):8d} {lo:19.3e} {s0:15.3e} "
              f"{'YES' if lo <= 255 else 'no':>11s}   [{time.time()-t0:.0f}s]")


# ---------------------------------------------------------------------------
# Sound bounds: the FULL integer kernel lattice, not the free-variable sublattice.
#
# `kernel_basis` clears denominators of the rational free-variable vectors, so
# its Z-span is a SUBLATTICE of {v in Z^W : K v = 0} -- of astronomical index at
# non-dyadic ratios (measured: index^2 ~ 1e100 at bicubic 4x).  A Gram-Schmidt
# bound on a sublattice says nothing about the lattice, so `main()` above (kept
# unchanged for the record) certifies only the sublattice.  The functions below
# compute a Z-basis of the full integer kernel by unimodular row reduction on
# [M^T | I], reduce it with float-guided LLL (every operation is an exact integer
# unimodular step, so the output is ALWAYS a basis of the same lattice whatever
# the float decisions were), and only then take the exact Gram-Schmidt bound.
# ---------------------------------------------------------------------------

def integer_kernel_basis(K, c0, width):
    """Z-basis of {v in Z^width : K[:, c0:c0+width] v = 0}, by Hermite-style row ops.

    Rows of K that do not touch the window impose no constraint and are dropped.
    Returns [] when the window admits only the zero vector.
    """
    rows = [i for i in range(K.shape[0]) if any(K[i, c0:c0 + width] != 0)]
    m, W = len(rows), width
    A = [[int(K[rows[j], c0 + i]) for j in range(m)]
         + [1 if k == i else 0 for k in range(W)] for i in range(W)]
    r = 0
    for col in range(m):
        if r == W:
            break
        while True:
            nz = [i for i in range(r, W) if A[i][col] != 0]
            if not nz:
                break
            p = min(nz, key=lambda i: abs(A[i][col]))
            A[r], A[p] = A[p], A[r]
            done = True
            for i in range(r + 1, W):
                if A[i][col] != 0:
                    q = A[i][col] // A[r][col]
                    A[i] = [a - q * b for a, b in zip(A[i], A[r])]
                    if A[i][col] != 0:
                        done = False
            if done:
                break
        if A[r][col] != 0:
            r += 1
    return [row[m:] for row in A[r:]]


def lll_reduce(basis, delta=0.99, max_iter=20000, passes=8):
    """Float-guided LLL on an INTEGER basis, kept exact.

    The Gram-Schmidt data is recomputed from a float copy (numpy QR) at every step,
    and the integer basis is refreshed into that copy each iteration, so precision
    recovers as entries shrink.  Runs up to `passes` sweeps until a sweep changes
    nothing.  Returns (basis, {"iters", "passes", "converged"}).
    """
    B = [list(map(int, b)) for b in basis]
    n = len(B)
    if n < 2:
        return B, {"iters": 0, "passes": 0, "converged": True}
    total_it, used = 0, 0
    converged = False
    for p in range(passes):
        used = p + 1
        before = [list(b) for b in B]
        k, it = 1, 0
        while k < n and it < max_iter:
            it += 1
            Bf = np.array(B, dtype=float)
            Q, R = np.linalg.qr(Bf.T)
            d = np.diag(R)
            mu = (R / np.where(d == 0, 1.0, d)[:, None]).T   # mu[i, j] = R[j, i] / R[j, j]
            for j in range(k - 1, -1, -1):
                q = int(round(mu[k, j]))
                if q:
                    B[k] = [a - q * b for a, b in zip(B[k], B[j])]
                    mu[k, :j + 1] -= q * mu[j, :j + 1]
            Bf = np.array(B, dtype=float)
            Q, R = np.linalg.qr(Bf.T)
            d = np.diag(R)
            mu = (R / np.where(d == 0, 1.0, d)[:, None]).T
            if d[k] ** 2 >= (delta - mu[k, k - 1] ** 2) * d[k - 1] ** 2:
                k += 1
            else:
                B[k], B[k - 1] = B[k - 1], B[k]
                k = max(k - 1, 1)
        total_it += it
        if B == before:
            converged = it < max_iter
            break
    return B, {"iters": total_it, "passes": used, "converged": converged}


def certified_window_bound(K, c0, width, reduce=True):
    """Exact Gram-Schmidt lower bound on ||v||_inf over the FULL integer kernel of a window.

    Returns None when the window's kernel is trivial, else a dict with `bound`
    (float; sound for every nonzero integer kernel vector supported on the window),
    `basis_size`, `witness` (the shortest reduced basis vector, a list of ints -- an
    upper witness, verified K v == 0), `witness_maxabs`, and the LLL meta.
    """
    S = integer_kernel_basis(K, c0, width)
    if not S:
        return None
    meta = {"iters": 0, "passes": 0, "converged": True}
    if reduce:
        S, meta = lll_reduce(S)
    Kw = K[:, c0:c0 + width]
    for v in S:
        if any(int(x) != 0 for x in (Kw.astype(object) @ np.array(v, dtype=object))):
            raise AssertionError("reduced basis vector is not in the kernel")
    wit = min((b for b in S if any(b)), key=lambda b: max(abs(x) for x in b))
    return {"bound": gs_bound_exact(S), "basis_size": len(S), "witness": wit,
            "witness_maxabs": max(abs(x) for x in wit), "lll": meta}


if __name__ == "__main__":
    main()
