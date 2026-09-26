#!/usr/bin/env python3
"""Check the polyphase theorem (existence and size of integer kernel vectors) on the reconstructed operators.
For p = q+1: build H(z) from the reconstructed fixed-point taps, take the signed maximal minors,
divide by their gcd in Z[z] (content included), and report beta(K), ||u||_inf, and whether the
resulting vector is annihilated by the actual operator K."""
import sys, math, itertools
from fractions import Fraction
from functools import reduce
import numpy as np
import pathlib
_here = pathlib.Path(__file__).resolve()
for _c in (_here.parents[3] / "Code" / "src" if len(_here.parents) > 3 else None, _here.parents[1] / "src"):
    if _c is not None and (_c / "aliasforge").is_dir():
        sys.path.insert(0, str(_c)); break
from aliasforge.exact import KERNELS, pil_coeffs

def trim(a):
    a = list(a)
    while a and a[-1] == 0: a.pop()
    return a
def pmul(a, b):
    if not a or not b: return []
    r = [0]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b): r[i+j] += x*y
    return trim(r)
def psub(a, b):
    n = max(len(a), len(b)); return trim([(a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0) for i in range(n)])
def content(a): return reduce(math.gcd, [abs(int(x)) for x in a], 0)
def primitive(a):
    a = trim(a)
    if not a: return []
    den = reduce(lambda x, y: x*y//math.gcd(x, y), [Fraction(x).denominator for x in a], 1)
    b = [int(Fraction(x)*den) for x in a]; c = content(b); b = [x//c for x in b]
    return b if b[-1] > 0 else [-x for x in b]
def prem(a, b):                      # remainder over Q
    a = [Fraction(x) for x in a]; b = [Fraction(x) for x in b]
    while len(a) >= len(b) and a:
        f = a[-1]/b[-1]; s = len(a)-len(b)
        for i, y in enumerate(b): a[s+i] -= f*y
        a = trim(a)
    return a
def pgcd(a, b):                      # gcd in Z[z], content included
    a, b = trim(a), trim(b)
    if not a: return b
    if not b: return a
    c = math.gcd(content(a), content(b)); x, y = primitive(a), primitive(b)
    while y: x, y = y, primitive(prem(x, y)) if prem(x, y) else []
    return [c*t for t in x]
def pdiv_exact(a, g):
    a = [Fraction(x) for x in a]; q = [Fraction(0)]*(len(a)-len(g)+1)
    while a and len(a) >= len(g):
        f = a[-1]/g[-1]; s = len(a)-len(g); q[s] = f
        for i, y in enumerate(g): a[s+i] -= f*y
        a = trim(a)
    assert not a and all(x.denominator == 1 for x in q); return [int(x) for x in q]
def det(M):                           # polynomial determinant by Laplace expansion (q <= 3)
    n = len(M)
    if n == 1: return M[0][0]
    out = []
    for j in range(n):
        minor = [row[:j]+row[j+1:] for row in M[1:]]
        term = pmul(M[0][j], det(minor))
        out = psub(out, term) if j % 2 else psub(out, [-x for x in term])
    return out

def gate(name, K, n_in, n_out):
    """Bit-exact reconstruction gate: the integer operator must reproduce Pillow's resize on random rows."""
    import numpy as np
    from PIL import Image
    mode = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "box": Image.BOX, "lanczos": Image.LANCZOS}[name]
    img = np.random.default_rng(0).integers(0, 256, (4, n_in), dtype=np.uint8)
    ref = np.asarray(Image.fromarray(img).resize((n_out, 4), mode)).astype(int)
    half = 1 << 21
    mine = [[min(255, max(0, (sum(k*int(x) for k, x in zip(row, line) if k) + half) >> 22)) for row in K] for line in img]
    return int((ref == np.array(mine)).all())

def analyse(name, n_in, n_out):
    f, supp = KERNELS[name]; g0 = math.gcd(n_in, n_out); p, q = n_in//g0, n_out//g0
    K = pil_coeffs(n_in, n_out, support=supp, filt=f); K = [[int(x) for x in row] for row in K]
    gate_ok = gate(name, K, n_in, n_out)
    # periodicity: K[j+q][c+p] == K[j][c]
    bad = [j for j in range(n_out-q) if any(K[j+q][c+p] != K[j][c] for c in range(n_in-p)) or any(K[j][c] for c in range(n_in-p, n_in)) and False]
    interior = [j for j in range(n_out-q) if j not in bad]
    lo = next(j for j in range(n_out) if j not in bad); hi = max(j for j in interior)
    nonper = [j for j in range(lo, hi) if j in bad]
    n0 = (n_out//q)//2
    rows = []
    for phi in range(q):
        j = q*n0+phi; cols = [c for c in range(n_in) if K[j][c]]
        ms = sorted({(c - 0)//p - n0 for c in cols}); mmin, mmax = min(ms), max(ms)
        row = []
        for s in range(p):
            h = {m: K[j][p*(n0+m)+s] for m in range(mmin, mmax+1)}
            row.append(trim([h[mmax-e] for e in range(mmax-mmin+1)]))   # coefficient of z^e is h[mmax-e]
        rows.append(row)
    assert p == q+1
    minors = []
    for s in range(p):
        sub = [r[:s]+r[s+1:] for r in rows]; d = det(sub); minors.append(d if s % 2 == 0 else [-x for x in d])
    rank_full = any(minors)
    g = reduce(pgcd, [m for m in minors if m]); u = [pdiv_exact(m, g) if m else [] for m in minors]
    beta = max(max(abs(us[0]), abs(us[-1])) for us in u if us); unorm = max(abs(x) for us in u for x in us)
    # place u in the interior and test against the real operator
    deg = max(len(us) for us in u); v = [0]*n_in; base = n0 - deg
    for s, us in enumerate(u):
        for e, x in enumerate(us): v[p*(base+e)+s] = x
    ok = all(sum(K[j][c]*v[c] for c in range(n_in)) == 0 for j in range(n_out))
    Mrow = max(sum(abs(x) for x in row) for row in K)
    seq = [v[c] for c in range(n_in) if any(v)][p*base:p*(base+deg)]
    print(f"{name:8s} {n_in}->{n_out} p/q={p}/{q} border_cols L={lo} R={n_out-1-hi-q+1} nonperiodic_interior={len(nonper)} "
          f"full_rank={int(rank_full)} deg_gcd={len(g)-1} content_gcd=2^{math.log2(content(g)):.1f} beta={beta:.4g} ||u||={unorm:.4g} "
          f"Ku=0:{int(ok)} gate={gate_ok} M={Mrow:.3g} beta/M={beta/Mrow:.4g}" + (f" u={seq}" if unorm < 100 else ""), flush=True)

for n_in, n_out in ((896, 448), (672, 448), (2048, 1536)):
    for name in ("bicubic", "bilinear", "box", "lanczos"):
        try: analyse(name, n_in, n_out)
        except Exception as e: print(f"{name:8s} {n_in}->{n_out} ERROR {type(e).__name__}: {e}", flush=True)


def pairwise(name, n_in, n_out):
    """Integer ratios (q = 1): the explicit two-class witnesses V_a = H_b/g, V_b = -H_a/g, g = gcd(H_a, H_b).
    Reports the smallest sup-norm over class pairs, and checks that witness against the real operator."""
    f, supp = KERNELS[name]; p = n_in // n_out
    K = pil_coeffs(n_in, n_out, support=supp, filt=f); K = [[int(x) for x in row] for row in K]
    gate_ok = gate(name, K, n_in, n_out)
    n0 = n_out // 2; cols = [c for c in range(n_in) if K[n0][c]]
    ms = sorted({c // p - n0 for c in cols}); mmin, mmax = min(ms), max(ms)
    H = [trim([K[n0][p*(n0+mmax-e)+s] for e in range(mmax-mmin+1)]) for s in range(p)]
    best = None
    for a, b in itertools.combinations(range(p), 2):
        if not H[a] or not H[b]: continue
        g = pgcd(H[a], H[b]); Va, Vb = pdiv_exact(H[b], g), [-x for x in pdiv_exact(H[a], g)]
        norm = max(max(map(abs, Va)), max(map(abs, Vb)))
        if best is None or norm < best[0]: best = (norm, a, b, Va, Vb)
    norm, a, b, Va, Vb = best
    # V_s(z) = sum_n v_s[n] z^n, and H's coefficient of z^e is the tap at block mmax-e: place accordingly
    v = [0]*n_in; base = n0 - 8
    for e, x in enumerate(Va): v[p*(base+e)+a] = x
    for e, x in enumerate(Vb): v[p*(base+e)+b] = x
    ok = all(sum(K[j][c]*v[c] for c in range(n_in) if v[c]) == 0 for j in range(n_out))
    print(f"pairwise {name:8s} {n_in}->{n_out} ratio={p}x gate={gate_ok} best_pair=({a},{b}) sup_norm={norm} Kv=0:{int(ok)}", flush=True)

for name in ("bicubic", "bilinear", "lanczos"):
    for n_in in (1344, 1792, 2688, 3584):
        try: pairwise(name, n_in, 448)
        except Exception as e: print(f"pairwise {name:8s} {n_in}->448 ERROR {type(e).__name__}: {e}", flush=True)
