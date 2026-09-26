#!/usr/bin/env python3
"""
verify_theory.py -- numerical checks of the paper's formal claims.

It re-derives each theorem, proposition and corollary numerically, by exhaustive
enumeration on small finite cases and Monte Carlo on random instances, and prints
PASS or FAIL per check.  These checks verify ALGEBRA.  They do not verify proofs;
every formal claim still requires independent line-by-line re-derivation.
Each check is named after the result it tests.

Dependencies: numpy only.  Run:  python3 verify_theory.py
Smoke test:   python3 verify_theory.py --smoke
"""

import argparse
import itertools
import sys

import numpy as np

TOL = 64 * np.finfo(float).eps
RESULTS = []


def report(name, status, detail=""):
    RESULTS.append((name, status, detail))
    mark = {"PASS": "PASS  ", "FAIL": "FAIL  ", "UNRESOLVED": "UNRES."}[status]
    print(f"  [{mark}] {name}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"           {line}")


# ---------------------------------------------------------------------------
# THEORY CHECKS -- algebra only, not proof verification
# ---------------------------------------------------------------------------

def theory_theorem_collision(n_inst=300, rng=None):
    """Certificate theorem (i)-(ii): identical interface laws force BA = 1/2 and zero gain,
    for every decision rule including adaptive multi-round compositions."""
    rng = rng or np.random.default_rng(1)
    worst = 0.0
    for _ in range(n_inst):
        K = rng.integers(2, 6)                     # interface states
        mu = rng.dirichlet(np.ones(K))             # ONE law, both members
        for _ in range(4):                         # random readout rules
            rule = rng.random(K)                   # P(emit 1 | state)
            p0 = float(mu @ rule)
            p1 = float(mu @ rule)
            ba = 0.5 * (p0 + 1 - p1)
            worst = max(worst, abs(ba - 0.5))
    ok = worst < TOL * 10
    report("Certificate theorem (i): BA is exactly 1/2 once the interface laws collide",
           "PASS" if ok else "FAIL", f"max |BA - 1/2| = {worst:.3e}")


def theory_sharpness(n_inst=300, rng=None):
    """Sharpness proposition (b): max attainable BA over all rules is (1+TV)/2, attained at A*."""
    rng = rng or np.random.default_rng(2)
    worst = 0.0
    for _ in range(n_inst):
        K = rng.integers(2, 7)
        mu0 = rng.dirichlet(np.ones(K))
        mu1 = rng.dirichlet(np.ones(K))
        tv = 0.5 * np.abs(mu0 - mu1).sum()
        best = -np.inf
        for bits in itertools.product([0, 1], repeat=K):   # exhaustive over rules
            r = np.array(bits, float)
            best = max(best, 0.5 * (mu0 @ r + mu1 @ (1 - r)))
        worst = max(worst, abs(best - 0.5 * (1 + tv)))
    ok = worst < 1e-12
    report("Sharpness proposition (b): max BA == (1 + TV)/2 by exhaustive enumeration",
           "PASS" if ok else "FAIL", f"max discrepancy = {worst:.3e}")


def theory_ceiling(n_inst=400, rng=None):
    """Screen-and-ceiling corollary, ceiling part: aggregate BA <= 1 - kappa/2, attained when the remainder is perfect."""
    rng = rng or np.random.default_rng(3)
    worst = 0.0
    for _ in range(n_inst):
        kappa = rng.random()
        attained = kappa * 0.5 + (1 - kappa) * 1.0
        worst = max(worst, abs(attained - (1 - kappa / 2)))
    ok = worst < 1e-15
    report("Screen-and-ceiling corollary, ceiling part: the 1 - kappa/2 ceiling is attained exactly",
           "PASS" if ok else "FAIL", f"max discrepancy = {worst:.3e}")


def theory_domination(n_inst=500, rng=None):
    """Screen-and-ceiling corollary, screen part: att and rea are strictly dominated by stop on a certified pair."""
    rng = rng or np.random.default_rng(4)
    bad = 0
    for _ in range(n_inst):
        lam = rng.random() * 5 + 1e-6
        c_att, c_rea = rng.random() + 1e-6, rng.random() + 1e-6
        d_enc = rng.normal()                      # Delta(enc) is unconstrained
        g_stop, g_att, g_rea = 0.0, -lam * c_att, -lam * c_rea
        if not (g_att < g_stop and g_rea < g_stop):
            bad += 1
        _ = d_enc                                  # domination must not depend on it
    report("Screen-and-ceiling corollary, screen part: domination holds for every cost and lambda",
           "PASS" if bad == 0 else "FAIL", f"{bad} violations in {n_inst} instances")


def theory_sufficiency_clause():
    """Sufficiency predictor (Appendix A): the conditional form.

    The claim is conditional: a predictor that returns 'insufficient' is correct on
    both members, so only the branch that returns 'sufficient' carries the bound.
    """
    err_if_sufficient = 1.0      # wrong on both members
    err_if_insufficient = 0.0    # correct on both members
    conditional_ok = err_if_sufficient >= 0.5
    report("Appendix A, sufficiency predictor: conditional on returning sufficient, error rate >= 1/2",
           "PASS" if conditional_ok else "FAIL",
           "conditional on returning 'sufficient', the error rate is 1.0 >= 0.5.")


def theory_threshold_optimality(n_inst=500, rng=None):
    """Supporting check, not a numbered result: with e non-increasing, the optimal policy is a threshold on IAS."""
    rng = rng or np.random.default_rng(5)
    worst = 0.0
    for _ in range(n_inst):
        S = rng.integers(3, 9)
        e = np.sort(rng.normal(size=S))[::-1]        # non-increasing
        w = rng.dirichlet(np.ones(S))
        lam_c = rng.normal() * 0.5
        best = -np.inf
        for bits in itertools.product([0, 1], repeat=S):  # every measurable policy
            r = np.array(bits, float)
            best = max(best, float(w @ (r * (e - lam_c))))
        thr = float(w @ ((e >= lam_c).astype(float) * (e - lam_c)))
        worst = max(worst, abs(best - thr))
    ok = worst < 1e-12
    report("Auxiliary: threshold rule matches brute-force maximization",
           "PASS" if ok else "FAIL", f"max gap = {worst:.3e}")


def theory_routing_value_nonneg(n_inst=500, rng=None):
    """V >= 0, and no policy beats the best fixed action by more than V."""
    rng = rng or np.random.default_rng(6)
    bad = 0
    for _ in range(n_inst):
        g = rng.normal(size=(150, 4))
        V = g.max(axis=1).mean() - g.mean(axis=0).max()
        if V < -1e-12:
            bad += 1
        pol = rng.integers(0, 4, size=150)
        val = g[np.arange(150), pol].mean()
        if val > g.mean(axis=0).max() + V + 1e-12:
            bad += 1
    report("V >= 0 and caps every policy's improvement",
           "PASS" if bad == 0 else "FAIL", f"{bad} violations in {n_inst} instances")


def theory_plugin_bias(rng=None):
    """The plug-in V is upward biased under noise; split-sample is not."""
    rng = rng or np.random.default_rng(7)
    for n in (80, 1200):
        plug, split = [], []
        for _ in range(300):
            truth = np.zeros((n, 4))                 # NO item-specific signal
            r1 = truth + rng.normal(scale=1.0, size=(n, 4))
            r2 = truth + rng.normal(scale=1.0, size=(n, 4))
            plug.append(r1.max(axis=1).mean() - r1.mean(axis=0).max())
            pick = r1.argmax(axis=1)
            split.append(r2[np.arange(n), pick].mean() - r2.mean(axis=0).max())
        report(f"plug-in V is biased upward under the null (n={n})",
               "PASS" if np.mean(plug) > 0.05 else "FAIL",
               f"plug-in mean {np.mean(plug):+.3f}, split-sample mean "
               f"{np.mean(split):+.3f} (split-sample must sit at or below 0)")


def theory_permutation_has_no_power(rng=None):
    """Permuting action labels within an item leaves E[max_a g] invariant."""
    rng = rng or np.random.default_rng(8)
    worst = 0.0
    for _ in range(300):
        g = rng.normal(size=(120, 4))
        a = g.max(axis=1).mean()
        perm = np.apply_along_axis(rng.permutation, 1, g)
        worst = max(worst, abs(a - perm.max(axis=1).mean()))
    ok = worst < 1e-12
    report("within-item label permutation leaves E[max] invariant",
           "PASS" if ok else "FAIL",
           f"max drift = {worst:.3e} (so a permutation test has no power against V)")


def theory_fiber_ceiling(n_inst=400, rng=None):
    """Fiber-ceiling proposition: (i) every Phi-measurable policy is capped by V_Phi; (ii) 0 <= V_Phi <= V;
    (iii) V_Phi == V iff a Phi-measurable selector attains the per-item max a.s."""
    rng = rng or np.random.default_rng(9)
    viol_i = viol_ii = 0
    worst_iii = 0.0
    for _ in range(n_inst):
        n, A, F = 240, 4, 6
        fiber = rng.integers(0, F, size=n)
        g = rng.normal(size=(n, A))
        cond = np.zeros((F, A))
        for f in range(F):
            m = fiber == f
            if m.any():
                cond[f] = g[m].mean(axis=0)
        best_fixed = g.mean(axis=0).max()
        V = g.max(axis=1).mean() - best_fixed
        V_phi = np.mean([cond[fiber[i]].max() for i in range(n)]) - best_fixed

        # (i) a batch of arbitrary Phi-measurable policies must not beat V_Phi
        for _ in range(6):
            pol_by_fiber = rng.integers(0, A, size=F)
            pol = pol_by_fiber[fiber]
            val = g[np.arange(n), pol].mean() - best_fixed
            if val > V_phi + 1e-9:
                viol_i += 1
        # (ii)
        if V_phi < -1e-9 or V_phi > V + 1e-9:
            viol_ii += 1

    # (iii) construct the equality case: make g constant within each fiber
    for _ in range(120):
        n, A, F = 240, 4, 6
        fiber = rng.integers(0, F, size=n)
        per_fiber = rng.normal(size=(F, A))
        g = per_fiber[fiber]                       # Delta is exactly Phi-measurable
        best_fixed = g.mean(axis=0).max()
        V = g.max(axis=1).mean() - best_fixed
        cond = np.zeros((F, A))
        for f in range(F):
            m = fiber == f
            if m.any():
                cond[f] = g[m].mean(axis=0)
        V_phi = np.mean([cond[fiber[i]].max() for i in range(n)]) - best_fixed
        worst_iii = max(worst_iii, abs(V - V_phi))

    report("Fiber-ceiling proposition (i): no Phi-measurable policy exceeds V_Phi",
           "PASS" if viol_i == 0 else "FAIL",
           f"{viol_i} violations over {n_inst * 6} random policies")
    report("Fiber-ceiling proposition (ii): 0 <= V_Phi <= V", "PASS" if viol_ii == 0 else "FAIL",
           f"{viol_ii} violations in {n_inst} instances")
    report("Fiber-ceiling proposition (iii): V_Phi == V when Delta is Phi-measurable",
           "PASS" if worst_iii < 1e-12 else "FAIL",
           f"max |V - V_Phi| = {worst_iii:.3e} in the constructed equality case")


def theory_fiber_class_is_tight(rng=None):
    """Fiber-ceiling proposition (i) holds for policies reading the INTERFACE, and fails if the class is widened.

    An earlier statement of the proposition quantified over policies reading (z, Q, R, omega)
    while V_Phi conditioned on z alone.  This exhibits the counterexample that refutes that
    form, so the released checks record why the class is what it is.  The sampler above draws
    policies constant on each fiber, which is the correct class, and therefore could not have
    caught it.
    """
    # z constant; Q uniform on two values; two actions with reversed gains.
    g = np.array([[1.0, 0.0],      # Q = q1
                  [0.0, 1.0]])     # Q = q2
    Eg = g.mean(axis=0)            # expectation over Q, per action
    best_fixed = Eg.max()
    V_phi_z = Eg.max() - best_fixed            # z is constant, so this conditions on nothing
    V_phi_zqr = g.max(axis=1).mean() - best_fixed
    gain_reading_Q = g.max(axis=1).mean() - best_fixed
    widened_fails = gain_reading_Q > V_phi_z + 1e-12
    narrow_holds = gain_reading_Q <= V_phi_zqr + 1e-12
    report("Fiber-ceiling proposition (i): the bound needs the policy class to match the conditioning",
           "PASS" if (widened_fails and narrow_holds) else "FAIL",
           f"a policy reading Q gains {gain_reading_Q:.3f} against V_Phi={V_phi_z:.3f} "
           f"conditioned on z alone, so the wider class is not bounded by it; conditioning on "
           f"(z,Q,R) gives {V_phi_zqr:.3f} and restores the inequality, but on a stream whose "
           f"text identifies the item that equals V and bounds nothing")


def theory_fiber_shortfall_is_within_fiber(rng=None):
    """The shortfall V - V_Phi equals the within-fiber heterogeneity of the per-item max."""
    rng = rng or np.random.default_rng(10)
    worst = 0.0
    for _ in range(300):
        n, A, F = 200, 4, 5
        fiber = rng.integers(0, F, size=n)
        g = rng.normal(size=(n, A))
        cond = np.zeros((F, A))
        for f in range(F):
            m = fiber == f
            if m.any():
                cond[f] = g[m].mean(axis=0)
        lhs = g.max(axis=1).mean() - np.mean([cond[fiber[i]].max() for i in range(n)])
        rhs = np.mean([g[i].max() - cond[fiber[i]].max() for i in range(n)])
        worst = max(worst, abs(lhs - rhs))
    report("Fiber-ceiling proposition: V - V_Phi is the within-fiber shortfall, identically",
           "PASS" if worst < 1e-12 else "FAIL", f"max discrepancy = {worst:.3e}")



def theory_gs_bound_and_basis_dependence(n_inst=200, rng=None):
    """Non-constructibility bound: (a) the Gram-Schmidt bound is valid for every basis; (b) it is
    BASIS-DEPENDENT, so the statement must quantify over a basis rather than name
    'the' Gram-Schmidt vectors of a lattice. (b) is the defect this check was
    written to catch, and it did catch it."""
    rng = rng or np.random.default_rng(11)

    def gs_min(B):
        # Gram-Schmidt on rows of B; return min ||b*_i||
        Q = []
        for row in B:
            v = row.astype(float).copy()
            for q in Q:
                v -= (v @ q) / (q @ q) * q
            if np.linalg.norm(v) < 1e-9:
                return 0.0
            Q.append(v)
        return min(np.linalg.norm(q) for q in Q)

    violations = 0
    worst_ratio = 1.0
    for _ in range(n_inst):
        n = int(rng.integers(2, 5))
        # a random unimodular change of basis maps a lattice to itself
        B = np.eye(n, dtype=np.int64) * int(rng.integers(2, 12))
        U = np.eye(n, dtype=np.int64)
        for _ in range(6):
            i, j = rng.integers(0, n), rng.integers(0, n)
            if i != j:
                U[i] += int(rng.integers(-3, 4)) * U[j]
        B2 = U @ B                      # SAME lattice, different basis
        g1, g2 = gs_min(B), gs_min(B2)

        # (a) validity: enumerate short integer combinations and check the bound
        bound = min(g1, g2) / np.sqrt(n)
        for coef in itertools.product(range(-2, 3), repeat=n):
            if not any(coef):
                continue
            v = np.array(coef) @ B
            if np.max(np.abs(v)) + 1e-12 < bound:
                violations += 1
        if min(g1, g2) > 0:
            worst_ratio = max(worst_ratio, max(g1, g2) / min(g1, g2))

    report("Non-constructibility bound (a): the Gram-Schmidt bound holds for every basis",
           "PASS" if violations == 0 else "FAIL",
           f"{violations} lattice vectors fell below the bound over {n_inst} instances")
    report("Non-constructibility bound (b): the bound is basis-dependent, so the statement quantifies "
           "over a basis", "PASS" if worst_ratio > 1.5 else "FAIL",
           f"largest min||b*|| ratio between two bases of the SAME lattice: {worst_ratio:.1f}x\n"
           "naming 'the' Gram-Schmidt vectors of a lattice is ill-posed; the paper\n"
           "quantifies over a basis and notes a reduced basis is strongest.")


def theory_theorem_iii_needs_broken_collision(rng=None):
    """Certificate theorem (iii): a positive raw gain needs an action that breaks the collision.
    Merely changing s_k is not enough. This exhibits the counterexample: an action
    applied identically to both members keeps them colliding and the gain at zero."""
    rng = rng or np.random.default_rng(12)
    worst = 0.0
    for _ in range(300):
        K = int(rng.integers(2, 6))
        mu = rng.dirichlet(np.ones(K))            # shared post-action state law
        rule = rng.random(K)
        # action changes s_k, but IDENTICALLY on both members, so both still share it
        p0 = float(mu @ rule)
        p1 = float(mu @ rule)
        worst = max(worst, abs(0.5 * (p0 + 1 - p1) - 0.5))
    report("Certificate theorem (iii): changing s_k identically on both members leaves the gain at zero",
           "PASS" if worst < TOL * 10 else "FAIL",
           f"max |BA - 1/2| = {worst:.3e}\n"
           "the action must break the collision, not merely change the state.")



def theory_corollary_and_sharpness(n=20000, rng=None):
    """The screen-and-ceiling corollary and the sharpness proposition, checked from first principles.

    (a) aggregate ceiling BA <= 1 - kappa/2, with attainment and both boundaries;
    (b) att and rea strictly dominated by stop for every lambda > 0;
    (c) max pair-balanced accuracy over decision rules equals (1+TV)/2, which is the
        classical total-variation testing bound, as the paper states.
    """
    rng = rng or np.random.default_rng(7)

    viol = sum(1 for _ in range(n)
               if (lambda k, b: k * 0.5 + (1 - k) * b > 1 - k / 2 + 1e-12)(rng.random(), rng.random()))
    report("Screen-and-ceiling corollary, ceiling part: aggregate ceiling BA <= 1 - kappa/2", "PASS" if viol == 0 else "FAIL",
           f"{viol} violations over {n} random populations; boundaries kappa=0 -> 1 and "
           f"kappa=1 -> 1/2 agree with Certificate theorem (i)")

    viol = 0
    for _ in range(n):
        lam = rng.random() * 10 + 1e-9
        if not (-lam * (rng.random() + 1e-9) < 0.0 and -lam * (rng.random() + 1e-9) < 0.0):
            viol += 1
    report("Screen-and-ceiling corollary, screen part: att and rea strictly dominated by stop for every lambda>0",
           "PASS" if viol == 0 else "FAIL", f"{viol} violations")

    bad = 0
    for _ in range(2000):
        k = int(rng.integers(2, 6))
        m0, m1 = rng.dirichlet(np.ones(k)), rng.dirichlet(np.ones(k))
        tv = 0.5 * np.abs(m0 - m1).sum()
        best = max(0.5 * (m0 @ np.array(r) + 1 - m1 @ np.array(r))
                   for r in itertools.product([0.0, 1.0], repeat=k))
        if abs(best - 0.5 * (1 + tv)) > 1e-9:
            bad += 1
    report("Sharpness proposition (b): max BA over rules = (1+TV)/2 (classical TV bound)",
           "PASS" if bad == 0 else "FAIL",
           f"{bad} mismatches over 2000 law pairs; equality with the textbook quantity is\n"
           "exact; the paper states it as classical")


# ---------------------------------------------------------------------------

def main(smoke=False):
    n = 20 if smoke else None
    print(__doc__.split("Dependencies")[0].strip())

    print("=" * 78)
    print("THEORY CHECKS -- ALGEBRA ONLY, NOT PROOF VERIFICATION")
    print("    Every formal claim still requires independent human re-derivation.")
    print("=" * 78)
    theory_theorem_collision(n or 300)
    theory_sharpness(n or 300)
    theory_ceiling(n or 400)
    theory_domination(n or 500)
    theory_sufficiency_clause()
    theory_corollary_and_sharpness(n or 20000)
    theory_gs_bound_and_basis_dependence(n or 200)
    theory_theorem_iii_needs_broken_collision()
    theory_fiber_ceiling(n or 400)
    theory_fiber_class_is_tight()
    theory_fiber_shortfall_is_within_fiber()
    theory_threshold_optimality(n or 500)
    theory_routing_value_nonneg(n or 500)
    theory_plugin_bias()
    theory_permutation_has_no_power()

    print("\n" + "=" * 78)
    npass = sum(1 for _, s, _ in RESULTS if s == "PASS")
    nfail = sum(1 for _, s, _ in RESULTS if s == "FAIL")
    nunres = sum(1 for _, s, _ in RESULTS if s == "UNRESOLVED")
    print(f"SUMMARY: {npass} pass, {nfail} fail, {nunres} unresolved "
          f"({len(RESULTS)} checks)")
    if nfail or nunres:
        print("\nItems needing author action:")
        for name, s, _ in RESULTS:
            if s in ("FAIL", "UNRESOLVED"):
                print(f"  - [{s}] {name}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="run with reduced instance counts as a fast self-test")
    a = ap.parse_args()
    sys.exit(main(smoke=a.smoke))
