#!/usr/bin/env python3
"""Lattice reduction for the resampler's integer kernel, and honest bounds on it.

This module exists because a negative search result that nobody can re-run is
worth nothing.  An earlier version of this work reported that lattice reduction
"does not help" at non-dyadic downscale factors on the strength of code that ran
inline in a shell and was never saved -- and whose first invocation reduced a
TRUNCATED basis (10 of 29 vectors), i.e. searched a sublattice rather than the
lattice.  Both the claim and the method are recorded here so the finding can be
checked and, if it is wrong, refuted.

Three tools, in increasing order of how much they license:

  lll                  float LLL.  Fast, and adequate when entries are small.
                       UNRELIABLE above ~1e12, which is exactly the regime the
                       non-dyadic factors live in, so a null result from it is
                       weak evidence and is reported as such.
  exact_kernel_check   verifies K @ v == 0 over the integers.  Any vector that
                       is going to be believed must pass this, because float
                       reduction can and does produce near-kernel vectors.
  gram_schmidt_bound   a CERTIFIED lower bound on the shortest vector,
                       lambda_inf >= min_i ||b*_i|| / sqrt(W).  This is the only
                       tool here that can support a claim of the form "no short
                       vector exists"; searching and failing cannot.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def kernel_basis(K: np.ndarray, c0: int, width: int) -> list[list[int]]:
    """Integer basis of {v supported on [c0, c0+width) : K v = 0}, by exact RREF.

    The vectors this returns are free-variable back-substitutions with denominators
    cleared, so they can be enormous -- that largeness is a property of the BASIS,
    not of the lattice, which is the whole reason reduction is needed afterwards.
    """
    rows = [i for i in range(K.shape[0]) if any(K[i, c0:c0 + width] != 0)]
    M = [[Fraction(int(K[i, j])) for j in range(c0, c0 + width)] for i in rows]
    r, c = len(M), width
    piv, row = [], 0
    for col in range(c):
        p = next((i for i in range(row, r) if M[i][col] != 0), None)
        if p is None:
            continue
        M[row], M[p] = M[p], M[row]
        pv = M[row][col]
        M[row] = [x / pv for x in M[row]]
        for i in range(r):
            if i != row and M[i][col] != 0:
                f = M[i][col]
                M[i] = [a - f * b for a, b in zip(M[i], M[row])]
        piv.append(col)
        row += 1
        if row == r:
            break
    out = []
    for fj in [j for j in range(c) if j not in piv]:
        v = [Fraction(0)] * c
        v[fj] = Fraction(1)
        for i, pc in enumerate(piv):
            v[pc] = -M[i][fj]
        den = 1
        for x in v:
            den = den * x.denominator // math.gcd(den, x.denominator)
        iv = [int(x * den) for x in v]
        g = 0
        for x in iv:
            g = math.gcd(g, abs(x))
        out.append([x // g for x in iv] if g else iv)
    return out


def lll(basis: list[list[int]], delta: float = 0.99, max_iter: int = 20000):
    """Float LLL, returning the reduced INTEGER basis and the iteration count.

    Operates on the FULL basis -- passing a subset silently searches a sublattice,
    which is the error this module was written to prevent.  Returns the iteration
    count so a caller can tell "converged" from "hit the cap", since a capped run
    is not a completed reduction and must not be reported as one.
    """
    n = len(basis)
    Bf = [list(map(float, b)) for b in basis]
    Bi = [list(map(int, b)) for b in basis]

    def gso():
        Bs, mu = [], [[0.0] * n for _ in range(n)]
        for i in range(n):
            v = list(Bf[i])
            for j in range(i):
                d = sum(x * x for x in Bs[j])
                mu[i][j] = (sum(a * b for a, b in zip(Bf[i], Bs[j])) / d) if d > 1e-300 else 0.0
                v = [a - mu[i][j] * b for a, b in zip(v, Bs[j])]
            Bs.append(v)
        return Bs, mu

    Bs, mu = gso()
    k, it = 1, 0
    while k < n and it < max_iter:
        it += 1
        for j in range(k - 1, -1, -1):
            if abs(mu[k][j]) > 0.5:
                r = round(mu[k][j])
                Bf[k] = [a - r * b for a, b in zip(Bf[k], Bf[j])]
                Bi[k] = [a - r * b for a, b in zip(Bi[k], Bi[j])]
                Bs, mu = gso()
        if sum(x * x for x in Bs[k]) >= (delta - mu[k][k - 1] ** 2) * sum(x * x for x in Bs[k - 1]):
            k += 1
        else:
            Bf[k], Bf[k - 1] = Bf[k - 1], Bf[k]
            Bi[k], Bi[k - 1] = Bi[k - 1], Bi[k]
            Bs, mu = gso()
            k = max(k - 1, 1)
    return Bi, {"iters": it, "converged": it < max_iter, "n": n}


def exact_kernel_check(K: np.ndarray, v_full: np.ndarray) -> bool:
    """K @ v == 0 over the integers.  Nothing is believed without this."""
    return int(max(abs(int(x)) for x in (K @ v_full.astype(object)))) == 0


def gram_schmidt_bound(basis: list[list[int]]) -> float:
    """Certified lower bound on the shortest nonzero vector: min||b*|| / sqrt(W).

    This is the ONLY function here that can support "no short vector exists".
    Failing to find one by search says nothing; this says something.
    """
    B = [np.array(b, dtype=float) for b in basis]
    Bs = []
    for i, b in enumerate(B):
        v = b.copy()
        for u in Bs:
            d = float(u @ u)
            if d > 0:
                v = v - (float(b @ u) / d) * u
        Bs.append(v)
    norms = [float(np.linalg.norm(v)) for v in Bs if np.linalg.norm(v) > 0]
    if not norms:
        return 0.0
    return min(norms) / math.sqrt(len(basis[0]))


def shortest_in_window(K: np.ndarray, c0: int, width: int, reduce: bool = True):
    """Best integer kernel vector found in a window, with provenance attached."""
    B = kernel_basis(K, c0, width)
    if not B:
        return None, {"reason": "empty kernel"}
    naive = min(max(abs(x) for x in b) for b in B)
    info = {"naive_min": naive, "basis_size": len(B),
            "certified_lower_bound": gram_schmidt_bound(B)}
    best = min(((max(abs(x) for x in b), b) for b in B if any(b)), key=lambda t: t[0])
    if reduce:
        R, meta = lll(B)
        info.update(meta)
        cand = min(((max(abs(x) for x in b), b) for b in R if any(b)), key=lambda t: t[0])
        if cand[0] < best[0]:
            best = cand
    info["found_min"] = best[0]
    v = np.zeros(K.shape[1], dtype=object)
    v[c0:c0 + width] = best[1]
    info["exact_kernel"] = exact_kernel_check(K, v)
    return v, info
