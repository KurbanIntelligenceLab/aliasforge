#!/usr/bin/env python3
"""Numerical verification of every formal claim in the manuscript.

This is the module the paper refers to when it says its algebra was checked
before it was written.  Each check below re-derives one claim by brute force --
enumerating every decision rule, simulating populations, or sampling random
instances -- and compares the brute-force answer against the closed form the
manuscript states.  Nothing here is evidence about a model; these checks verify
ALGEBRA ONLY.  The proofs still require independent line-by-line human reading.

Claims covered
--------------
  C1  attainable accuracy      sup_d BA(d) = (1 + TV(mu0, mu1)) / 2
                               and BA(d) = 1/2 for EVERY d when mu0 = mu1
                               [Theorem 1(i)(ii), Proposition 1(b)]
  C2  necessity                (every rule has zero gain) <=> mu0 = mu1
                               [Proposition 1(a)]
  C3  aggregate ceiling        population BA <= 1 - kappa/2, attained
                               [Corollary 1(a)]
  C4  domination               att, rea strictly dominated by stop on a
                               certified pair, for every lambda > 0; and a
                               sufficiency predictor errs at rate >= 1/2
                               [Corollary 1(b)]
  C5  regret bound             per-item regret <= 2 max_a |ghat - g|, and the
                               plug-in policy is invariant to a per-item shift
                               [Remark 2]
  C6  routing value            V >= 0; every policy's improvement over the best
                               fixed action is at most V; the oracle attains it
                               [Remark 1]
  C7  twin licensing           kappa_LB from sound-but-incomplete mining is a
                               valid one-sided lower bound on the interface-
                               limited fraction  [Remark: twin licensing]
  C9  V estimation             the plug-in routing value is upward biased under the
                               null and the bias does not vanish with n, so a
                               bootstrap interval on it can never fail; the
                               split-sample estimator gates correctly, with size
                               <= alpha and power at a real effect
                               [Remark 1, Remark: estimating V, E1 gate]
  C8  threshold optimality     with a monotone conditional gain the cost-
                               adjusted optimal policy on {stop, enc} is a
                               threshold rule on the score, and the threshold
                               depends on the population only through that
                               conditional  [Proposition: threshold optimality]

Results are emitted twice, per project convention: one CSV row per check to
$OUT (or --out), and a short parsable summary to stdout.

Usage
-----
    python -m aliasforge.metrics --out checks.csv [--seed 0] [--scale 1.0]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# Version of THIS checking module, bumped when a check changes meaning.  Printed
# with every result so a stored CSV identifies the code that produced it.
CHECKS_VERSION = "1.1"

# Every claim is exact algebra, so the only admissible discrepancy is floating
# point.  We allow a small multiple of machine epsilon rather than a loose
# tolerance: a check that needs slack is a check that is not verifying the
# identity the manuscript states.
EPS = float(np.finfo(np.float64).eps)  # 2.220446049250313e-16
ATOL = 64 * EPS


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def total_variation(mu0: np.ndarray, mu1: np.ndarray) -> float:
    """TV(mu0, mu1) = (1/2) * sum |mu0 - mu1|, for laws on a finite set."""
    return 0.5 * float(np.abs(mu0 - mu1).sum())


def pair_balanced_accuracy(d: np.ndarray, mu0: np.ndarray, mu1: np.ndarray) -> float:
    """BA of decision rule ``d`` on a pair whose members induce mu0 and mu1.

    ``d[z]`` is the probability the verifier emits label 1 after seeing
    interface state ``z``.  Member 0 carries ground-truth label 1 and member 1
    carries label 0, so

        BA = (1/2) * ( P(emit 1 | member 0) + P(emit 0 | member 1) ).

    Written in this order deliberately: the residual it leaves when mu0 == mu1
    is exactly the floating-point quantity C1 reports.
    """
    p0 = float(mu0 @ d)
    p1 = float(mu1 @ d)
    return 0.5 * (p0 + 1.0 - p1)


def all_deterministic_rules(m: int) -> np.ndarray:
    """Every deterministic rule on m interface states, as a (2**m, m) array.

    Randomized rules are convex combinations of these, so a linear objective
    over [0,1]^m attains its maximum on this set.  C1 checks that claim rather
    than assuming it, by also sampling randomized rules.
    """
    return np.array(list(itertools.product([0.0, 1.0], repeat=m)), dtype=float)


def max_ba_bruteforce(mu0: np.ndarray, mu1: np.ndarray) -> float:
    """Maximum BA over every deterministic decision rule, by enumeration."""
    rules = all_deterministic_rules(len(mu0))
    return float(max(pair_balanced_accuracy(d, mu0, mu1) for d in rules))


def random_law(rng: np.random.Generator, m: int) -> np.ndarray:
    """A random law on m states.  Dirichlet(1) gives a uniform simplex draw."""
    return rng.dirichlet(np.ones(m))


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    """One-sided Clopper-Pearson lower confidence bound for a binomial rate.

    Implemented from the Beta quantile identity; scipy is avoided so this module
    has no dependency beyond numpy.  Returns 0.0 when k == 0, which is the
    correct (vacuous) bound.
    """
    if k == 0:
        return 0.0
    if k == n:
        return float(alpha ** (1.0 / n))
    # Lower bound solves  P(Bin(n, p) >= k) = alpha  in p; bisection is ample
    # here and keeps the implementation obviously correct.
    def tail(p: float) -> float:
        # P(Bin(n,p) >= k), computed in log space for numerical safety.
        total = 0.0
        for j in range(k, n + 1):
            logc = (
                math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
            )
            total += math.exp(logc + j * math.log(p) + (n - j) * math.log1p(-p))
        return total

    lo, hi = 0.0, k / n
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if tail(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# check registry
# ---------------------------------------------------------------------------


@dataclass
class Result:
    """One verified claim."""

    check: str
    claim: str  # where it lives in the manuscript
    n: int  # instances enumerated / simulated
    statistic: str  # what `value` measures
    value: float
    tolerance: float
    passed: bool
    extra: dict = field(default_factory=dict)


CHECKS: list[tuple[str, str, Callable]] = []


def check(name: str, claim: str):
    def deco(fn):
        CHECKS.append((name, claim, fn))
        return fn

    return deco


# ---------------------------------------------------------------------------
# C1 -- attainable accuracy, and the certified case
# ---------------------------------------------------------------------------


@check("C1a_attainable_accuracy", "Thm 1(i), Prop 1(b)")
def c1a(rng: np.random.Generator, n: int) -> Result:
    """sup over EVERY decision rule of BA equals (1 + TV)/2.

    Brute force enumerates all 2**m deterministic rules; the closed form is the
    manuscript's.  Randomized rules are sampled too, and none may exceed the
    enumerated maximum -- that is what licenses restricting to the vertices.
    """
    worst = 0.0
    worst_rand = 0.0
    for _ in range(n):
        m = int(rng.integers(2, 8))
        mu0, mu1 = random_law(rng, m), random_law(rng, m)
        brute = max_ba_bruteforce(mu0, mu1)
        closed = 0.5 * (1.0 + total_variation(mu0, mu1))
        worst = max(worst, abs(brute - closed))
        # randomized rules must not beat the enumerated vertex maximum
        for _ in range(20):
            d = rng.random(m)
            worst_rand = max(worst_rand, pair_balanced_accuracy(d, mu0, mu1) - brute)
    return Result(
        "C1a_attainable_accuracy",
        "Thm 1(i), Prop 1(b)",
        n,
        "max|bruteforce_maxBA - (1+TV)/2|",
        worst,
        ATOL,
        worst <= ATOL and worst_rand <= ATOL,
        {"max_randomized_excess": worst_rand},
    )


@check("C1b_certified_is_exactly_half", "Thm 1(i)(ii)")
def c1b(rng: np.random.Generator, n: int) -> Result:
    """When mu0 == mu1, EVERY rule -- not merely the best -- has BA = 1/2.

    This is the whole force of Theorem 1: it pins the entire language-side class,
    at any critique budget, rather than bounding the best member of it.  The
    reported value is the largest |BA - 1/2| over every enumerated rule, and it
    is pure floating-point residue.
    """
    worst = 0.0
    worst_adaptive = 0.0
    for _ in range(n):
        m = int(rng.integers(2, 8))
        mu = random_law(rng, m)
        for d in all_deterministic_rules(m):
            worst = max(worst, abs(pair_balanced_accuracy(d, mu, mu) - 0.5))
        # An adaptive procedure composes finitely many maps, each reading every
        # earlier output; Definition 1 says the composition is again a function
        # of the shared state.  Simulate that directly: draw a random multi-round
        # rule whose later rounds depend on earlier draws, and confirm the
        # composed rule still lands at exactly 1/2.
        composed = np.ones(m)
        for _round in range(int(rng.integers(2, 6))):
            composed = composed * rng.random(m)  # each round reads the last
        worst_adaptive = max(
            worst_adaptive, abs(pair_balanced_accuracy(composed, mu, mu) - 0.5)
        )
    return Result(
        "C1b_certified_is_exactly_half",
        "Thm 1(i)(ii)",
        n,
        "max|BA - 1/2| over all rules, mu0 = mu1",
        worst,
        ATOL,
        worst <= ATOL and worst_adaptive <= ATOL,
        {"max_adaptive_multiround_dev": worst_adaptive, "eps": EPS},
    )


@check("C1c_deterministic_interface_separates", "Prop 1(b)")
def c1c(rng: np.random.Generator, n: int) -> Result:
    """A deterministic interface with z0 != z1 admits a verifier reaching BA = 1.

    The second half of Proposition 1(b): once the states differ at all, the
    attainable accuracy is unconstrained, so no tolerance can certify.
    """
    worst = 0.0
    for _ in range(n):
        m = int(rng.integers(2, 8))
        z0, z1 = rng.choice(m, size=2, replace=False)
        mu0, mu1 = np.zeros(m), np.zeros(m)
        mu0[z0], mu1[z1] = 1.0, 1.0
        worst = max(worst, abs(max_ba_bruteforce(mu0, mu1) - 1.0))
    return Result(
        "C1c_deterministic_interface_separates",
        "Prop 1(b)",
        n,
        "max|maxBA - 1| for deterministic distinct states",
        worst,
        ATOL,
        worst <= ATOL,
    )


@check("C1d_every_delta_attained", "Prop 1(b)")
def c1d(rng: np.random.Generator, n: int) -> Result:
    """For every delta in (0, TV] some verifier attains BA = (1 + delta)/2.

    Checked constructively: scale the TV-optimal rule toward the constant rule
    and confirm the realized BA hits the requested delta.
    """
    worst = 0.0
    for _ in range(n):
        m = int(rng.integers(2, 8))
        mu0, mu1 = random_law(rng, m), random_law(rng, m)
        t = total_variation(mu0, mu1)
        if t <= 0:
            continue
        d_opt = (mu0 > mu1).astype(float)  # attains TV
        delta = float(rng.uniform(1e-6, 1.0)) * t
        d = (delta / t) * d_opt  # convex mix with the all-zero rule
        worst = max(worst, abs(pair_balanced_accuracy(d, mu0, mu1) - 0.5 * (1 + delta)))
    return Result(
        "C1d_every_delta_attained",
        "Prop 1(b)",
        n,
        "max|BA(constructed) - (1+delta)/2|",
        worst,
        ATOL,
        worst <= ATOL,
    )


# ---------------------------------------------------------------------------
# C2 -- necessity
# ---------------------------------------------------------------------------


@check("C2_necessity", "Prop 1(a)")
def c2(rng: np.random.Generator, n: int) -> Result:
    """(every rule has exactly zero gain) <=> mu0 = mu1.

    Both directions are checked on the same random draws: half the instances are
    forced to collide so the equivalence is exercised in both directions rather
    than only refuted.
    """
    mismatches = 0
    for i in range(n):
        m = int(rng.integers(2, 7))
        mu0 = random_law(rng, m)
        mu1 = mu0.copy() if i % 2 == 0 else random_law(rng, m)
        rules = all_deterministic_rules(m)
        max_gain = max(abs(pair_balanced_accuracy(d, mu0, mu1) - 0.5) for d in rules)
        all_zero = max_gain <= ATOL
        collide = total_variation(mu0, mu1) <= ATOL
        if all_zero != collide:
            mismatches += 1
    return Result(
        "C2_necessity",
        "Prop 1(a)",
        n,
        "count(all-rules-zero-gain XOR laws-equal)",
        float(mismatches),
        0.0,
        mismatches == 0,
    )


# ---------------------------------------------------------------------------
# C3 -- aggregate ceiling
# ---------------------------------------------------------------------------


@check("C3_aggregate_ceiling", "Cor 1(a)")
def c3(rng: np.random.Generator, n: int) -> Result:
    """Population BA <= 1 - kappa/2, and the bound is attained.

    Each simulated population mixes a certified fraction kappa (whose members
    collide, so BA = 1/2 exactly) with an uncertified remainder.  The violation
    statistic must be non-positive; the attainment statistic checks that a
    population whose remainder separates perfectly meets the bound with equality.
    """
    worst_violation = -np.inf
    worst_attain = 0.0
    for _ in range(n):
        n_pairs = int(rng.integers(4, 25))
        kappa_count = int(rng.integers(0, n_pairs + 1))
        kappa = kappa_count / n_pairs
        bas, bas_attaining = [], []
        for j in range(n_pairs):
            m = int(rng.integers(2, 6))
            if j < kappa_count:
                mu = random_law(rng, m)
                bas.append(max_ba_bruteforce(mu, mu))
                bas_attaining.append(max_ba_bruteforce(mu, mu))
            else:
                mu0, mu1 = random_law(rng, m), random_law(rng, m)
                bas.append(max_ba_bruteforce(mu0, mu1))
                # a perfectly separating remainder: disjoint deterministic states
                d0, d1 = np.zeros(m), np.zeros(m)
                z0, z1 = rng.choice(m, size=2, replace=False)
                d0[z0], d1[z1] = 1.0, 1.0
                bas_attaining.append(max_ba_bruteforce(d0, d1))
        bound = 1.0 - kappa / 2.0
        worst_violation = max(worst_violation, float(np.mean(bas)) - bound)
        worst_attain = max(worst_attain, abs(float(np.mean(bas_attaining)) - bound))
    return Result(
        "C3_aggregate_ceiling",
        "Cor 1(a)",
        n,
        "max(populationBA - (1-kappa/2))",
        float(worst_violation),
        ATOL,
        worst_violation <= ATOL and worst_attain <= ATOL,
        {"max_attainment_gap": worst_attain},
    )


# ---------------------------------------------------------------------------
# C4 -- domination and the sufficiency screen
# ---------------------------------------------------------------------------


@check("C4a_domination", "Cor 1(b)")
def c4a(rng: np.random.Generator, n: int) -> Result:
    """On a certified pair, att and rea are strictly dominated by stop.

    Delta(att) = Delta(rea) = 0 exactly by Theorem 1, so their cost-adjusted
    gains are -lambda*c < 0 = g(stop) for every lambda > 0, whatever
    Delta(enc) turns out to be.  The statistic is the largest g(att) or g(rea)
    observed; it must stay strictly below zero.
    """
    worst = -np.inf
    stop_ever_beaten = 0
    for _ in range(n):
        lam = float(rng.uniform(1e-6, 100.0))
        c_att = float(rng.uniform(1e-6, 10.0))
        c_rea = float(rng.uniform(1e-6, 10.0))
        c_enc = float(rng.uniform(1e-6, 10.0))
        delta_enc = float(rng.uniform(-1.0, 1.0))  # unconstrained on purpose
        g_stop = 0.0 - lam * 0.0
        g_att = 0.0 - lam * c_att  # Delta = 0 exactly, by Theorem 1
        g_rea = 0.0 - lam * c_rea
        g_enc = delta_enc - lam * c_enc
        worst = max(worst, g_att, g_rea)
        if max(g_att, g_rea) >= g_stop:
            stop_ever_beaten += 1
        # stop must be optimal exactly when enc does not pay -- the manuscript's
        # reason for excluding stop from the screen
        assert (g_stop >= g_enc) == (delta_enc <= lam * c_enc)
    return Result(
        "C4a_domination",
        "Cor 1(b)",
        n,
        "max(g_att, g_rea) over configs  [must be < 0]",
        float(worst),
        0.0,
        worst < 0.0 and stop_ever_beaten == 0,
    )


@check("C4b_sufficiency_error", "Cor 1(b)")
def c4b(rng: np.random.Generator, n: int) -> Result:
    """A sufficiency predictor errs on at least half of a certified pair.

    Both members induce one shared interface state, so the predictor returns one
    shared answer for two members carrying opposite correct answers.  Answering
    "sufficient" is wrong on at least one of the two; answering "insufficient" is
    right on both.  The statistic is the minimum error rate over the two possible
    answers when the predictor says "sufficient".
    """
    worst = 0.0
    for _ in range(n):
        m = int(rng.integers(2, 8))
        mu = random_law(rng, m)  # collided: both members induce this law
        # Declaring "sufficient" asserts the current view determines the correct
        # answer, i.e. that SOME rule reading the shared state answers both
        # members correctly.  Enumerate every such rule and take its best
        # pair-balanced error; the minimum attainable error is what the claim
        # bounds below by 1/2.
        best_err = min(
            1.0 - pair_balanced_accuracy(d, mu, mu) for d in all_deterministic_rules(m)
        )
        worst = max(worst, abs(best_err - 0.5))
    return Result(
        "C4b_sufficiency_error",
        "Cor 1(b)",
        n,
        "max|min attainable error when answering 'sufficient' - 1/2|",
        float(worst),
        ATOL,
        worst <= ATOL,
    )


# ---------------------------------------------------------------------------
# C5 -- regret bound and shift invariance
# ---------------------------------------------------------------------------


@check("C5a_regret_bound", "Rem 2")
def c5a(rng: np.random.Generator, n: int) -> Result:
    """Per-item regret of the plug-in policy is at most 2 * max_a |ghat - g|."""
    worst = -np.inf
    for _ in range(n):
        k = int(rng.integers(2, 6))
        g = rng.normal(size=k)
        ghat = g + rng.normal(scale=float(rng.uniform(0.0, 1.0)), size=k)
        pi_hat = int(np.argmax(ghat))
        regret = float(g.max() - g[pi_hat])
        bound = 2.0 * float(np.abs(ghat - g).max())
        worst = max(worst, regret - bound)
    return Result(
        "C5a_regret_bound",
        "Rem 2",
        n,
        "max(regret - 2*max|ghat-g|)  [must be <= 0]",
        float(worst),
        ATOL,
        worst <= ATOL,
    )


@check("C5b_shift_invariance", "Rem 2")
def c5b(rng: np.random.Generator, n: int) -> Result:
    """Adding a per-item constant to ghat leaves the plug-in choice unchanged.

    This is why the manuscript reports action ORDERING accuracy and regret rather
    than gain-regression error: only gain differences are identified.
    """
    mismatches = 0
    for _ in range(n):
        k = int(rng.integers(2, 6))
        ghat = rng.normal(size=k)
        shift = float(rng.normal(scale=10.0))
        if int(np.argmax(ghat)) != int(np.argmax(ghat + shift)):
            mismatches += 1
    return Result(
        "C5b_shift_invariance",
        "Rem 2",
        n,
        "count(argmax changed by a constant shift)",
        float(mismatches),
        0.0,
        mismatches == 0,
    )


# ---------------------------------------------------------------------------
# C6 -- routing value
# ---------------------------------------------------------------------------


def routing_value(g: np.ndarray) -> float:
    """V = E_i[max_a g_i(a)] - max_a E_i[g_i(a)], for an (items, actions) array.

    WARNING -- this is the plug-in estimator and it is UPWARD BIASED whenever the
    per-item gains are measured with noise.  The first term averages a per-item
    MAX over noisy values, so it inherits the winner's curse: under a null with no
    item-specific signal at all, this returns ~0.14 for unit-ish noise and it does
    NOT shrink with the number of items, because the bias lives inside each item,
    not in the average.  Do not test V > 0 by bootstrapping this quantity; the
    interval essentially never covers zero.  Use `routing_value_permutation_p` or
    `routing_value_crossfit` below.  See C9.
    """
    return float(g.max(axis=1).mean() - g.mean(axis=0).max())


# NOTE -- a within-item permutation test was tried here and REMOVED, because it
# cannot test this null.  Permuting the action labels inside one item leaves that
# item's multiset of gains untouched, so it leaves max_a g_i(a) exactly invariant
# and therefore leaves E_i[max_a g_i(a)] -- the entire first term of V -- unchanged.
# All it perturbs is the second term, max_a E_i[g_i(a)], which it can only shrink
# by levelling the column means.  The permuted V is therefore stochastically LARGER
# than the observed one whether or not any routing signal exists, and the test has
# no power: measured 0.00 at every effect size.  The split-sample estimator below
# is the correct instrument, and C9/C9b check its size and power directly.


def routing_value_crossfit(g_train: np.ndarray, g_eval: np.ndarray) -> float:
    """Split-sample V: pick each item's action on one replicate, score it on another.

    The bias comes from choosing and evaluating the per-item maximum on the SAME
    noisy numbers.  With two independent measurements of the same items -- which
    E1 gets for free, since it runs every action and verifier decoding is
    stochastic -- choosing on `g_train` and scoring on `g_eval` removes it.  The
    result is unbiased for the true routing value and may legitimately come out
    negative when there is no signal, which is the honest behaviour a gate needs.
    """
    pick = g_train.argmax(axis=1)
    oracle = g_eval[np.arange(g_eval.shape[0]), pick].mean()
    best_fixed = g_eval.mean(axis=0).max()
    return float(oracle - best_fixed)


@check("C6a_routing_value_bounds_gain", "Rem 1")
def c6a(rng: np.random.Generator, n: int) -> Result:
    """V >= 0, no policy beats the best fixed action by more than V, oracle = V.

    Random policies are sampled alongside the oracle so the bound is tested, not
    merely the identity that defines it.
    """
    worst_neg = np.inf
    worst_excess = -np.inf
    worst_oracle = 0.0
    for _ in range(n):
        n_items = int(rng.integers(5, 60))
        k = int(rng.integers(2, 5))
        g = rng.normal(size=(n_items, k))
        V = routing_value(g)
        worst_neg = min(worst_neg, V)
        best_fixed = float(g.mean(axis=0).max())
        for _ in range(10):
            pi = rng.integers(0, k, size=n_items)
            improvement = float(g[np.arange(n_items), pi].mean()) - best_fixed
            worst_excess = max(worst_excess, improvement - V)
        oracle_improvement = float(g.max(axis=1).mean()) - best_fixed
        worst_oracle = max(worst_oracle, abs(oracle_improvement - V))
    return Result(
        "C6a_routing_value_bounds_gain",
        "Rem 1",
        n,
        "max(policy_improvement - V)  [must be <= 0]",
        float(worst_excess),
        ATOL,
        worst_excess <= ATOL and worst_neg >= -ATOL and worst_oracle <= ATOL,
        {"min_V": float(worst_neg), "max_oracle_gap": float(worst_oracle)},
    )


@check("C6b_routing_value_zero_case", "Rem 1")
def c6b(rng: np.random.Generator, n: int) -> Result:
    """V = 0 exactly when one fixed action is optimal for every item.

    The V = 0 arm is built by giving every item the same argmax; the V > 0 arm
    by giving items conflicting argmaxes.  Bootstrap intervals are reported for
    both because E1 uses this estimand as its gate.
    """
    worst_zero = 0.0
    min_positive = np.inf
    cover_zero = 0
    cover_pos = 0
    for _ in range(n):
        n_items, k = 200, 3
        # V = 0 construction: action 0 dominates pointwise
        g0 = rng.normal(size=(n_items, k))
        g0[:, 0] = g0.max(axis=1) + rng.uniform(0.01, 1.0, size=n_items)
        worst_zero = max(worst_zero, abs(routing_value(g0)))
        # V > 0 construction: the argmax alternates by item
        gp = rng.normal(scale=0.1, size=(n_items, k))
        idx = rng.integers(0, k, size=n_items)
        gp[np.arange(n_items), idx] += 2.0
        min_positive = min(min_positive, routing_value(gp))
        # clustered-free bootstrap, matching the estimand E1 gates on
        lo0, hi0 = bootstrap_ci(g0, routing_value, rng, n_boot=200)
        lop, hip = bootstrap_ci(gp, routing_value, rng, n_boot=200)
        cover_zero += int(lo0 <= 0.0 <= hi0 or abs(hi0) <= 1e-9)
        cover_pos += int(lop > 0.0)
    return Result(
        "C6b_routing_value_zero_case",
        "Rem 1",
        n,
        "max|V| under the pointwise-dominant construction",
        float(worst_zero),
        ATOL,
        worst_zero <= ATOL and min_positive > 0.0,
        {
            "min_V_positive_case": float(min_positive),
            "boot_ci_covers_zero_frac": cover_zero / n,
            "boot_ci_excludes_zero_frac": cover_pos / n,
        },
    )


def _paired_replicates(n_items: int, delta: float, rng: np.random.Generator):
    """Two independent measurements of the SAME items.

    delta = 0 is the null: no item-specific signal, so no per-item router can beat
    the best fixed action.  delta > 0 gives each item a latent best action, which
    is the signal a router would exploit.  Both replicates share that latent truth
    and differ only in noise -- which is exactly what E1 obtains by running every
    action twice under stochastic decoding.
    """
    base = np.array([0.0, -0.02, -0.04, -0.06])
    best = rng.integers(0, 4, size=n_items)

    def draw():
        g = base[None, :] + rng.normal(scale=0.15, size=(n_items, 4))
        g[np.arange(n_items), best] += delta
        return g

    return draw(), draw()


@check("C9_plugin_routing_value_is_biased", "Rem 1, E1 gate")
def c9(rng: np.random.Generator, n: int) -> Result:
    """The plug-in V is upward biased under the null and the bias does not vanish.

    This check exists because E1's gate depends on it.  A gate reading "stop if the
    95% interval for V covers 0" is only meaningful if the interval CAN cover 0
    when nothing is there.  With the plug-in estimator it cannot: averaging a
    per-item maximum over noisy values inherits the winner's curse, the bias lives
    inside each item rather than in the average, and so it does not shrink as items
    are added.  Reported value is the naive bootstrap's false-positive rate, which
    must NOT be near alpha -- this check passes when the naive gate is shown broken.
    """
    n_inst = max(20, n // 10)
    boot_rej = 0
    small, large = [], []
    for _ in range(n_inst):
        a_s, _ = _paired_replicates(80, 0.0, rng)
        a_l, _ = _paired_replicates(1200, 0.0, rng)
        small.append(routing_value(a_s))
        large.append(routing_value(a_l))
        lo, _hi = bootstrap_ci(a_s, routing_value, rng, n_boot=200)
        boot_rej += int(lo > 0.0)

    rate = boot_rej / n_inst
    ms, ml = float(np.mean(small)), float(np.mean(large))
    return Result(
        "C9_plugin_routing_value_is_biased",
        "Rem 1, E1 gate",
        n_inst,
        "naive bootstrap false-positive rate under the null  [pathological: ~1]",
        rate,
        0.0,
        rate > 0.5 and ml > 0.5 * ms,  # the bug reproduces: high FPR, bias persists
        {
            "plugin_V_under_null_n80": round(ms, 4),
            "plugin_V_under_null_n1200": round(ml, 4),
            "bias_shrinks_with_n": bool(ml < 0.5 * ms),
        },
    )


@check("C9b_splitsample_gate_is_valid", "Rem 1, E1 gate")
def c9b(rng: np.random.Generator, n: int) -> Result:
    """The split-sample gate has correct size under the null AND power at a real effect.

    Choosing each item's action on one replicate and scoring it on an independent
    one removes the winner's curse.  Under the null the statistic sits at or just
    below zero -- slightly negative, because the action chosen on noise regresses
    to the mean when scored fresh -- so a lower-bound gate can actually fail, which
    is what a gate must be able to do.  Reported value is the null rejection rate;
    the alternative's rejection rate is reported alongside so a test that is merely
    conservative-and-useless cannot pass.
    """
    n_inst = max(20, n // 10)
    n_items, n_boot = 320, 150
    null_rej = alt_rej = 0
    null_v, alt_v = [], []

    def gate(a, b):
        stat = [
            routing_value_crossfit(a[s], b[s])
            for s in (rng.integers(0, a.shape[0], size=a.shape[0]) for _ in range(n_boot))
        ]
        return float(np.quantile(stat, 0.025)) > 0.0

    for _ in range(n_inst):
        a0, b0 = _paired_replicates(n_items, 0.0, rng)     # null
        a1, b1 = _paired_replicates(n_items, 0.20, rng)    # real routing signal
        null_v.append(routing_value_crossfit(a0, b0))
        alt_v.append(routing_value_crossfit(a1, b1))
        null_rej += int(gate(a0, b0))
        alt_rej += int(gate(a1, b1))

    size = null_rej / n_inst
    power = alt_rej / n_inst
    return Result(
        "C9b_splitsample_gate_is_valid",
        "Rem 1, E1 gate",
        n_inst,
        "split-sample gate size under the null  [must be <= alpha]",
        float(size),
        0.05,
        size <= 0.05 and power >= 0.80,
        {
            "power_at_delta_0.20": power,
            "mean_V_under_null": round(float(np.mean(null_v)), 4),
            "mean_V_under_alt": round(float(np.mean(alt_v)), 4),
        },
    )


def bootstrap_ci(
    g: np.ndarray,
    stat: Callable[[np.ndarray], float],
    rng: np.random.Generator,
    n_boot: int = 1000,
    alpha: float = 0.05,
    clusters: np.ndarray | None = None,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling clusters when they are supplied.

    E1 and E3 cluster at the render template and source image because items nest
    inside templates and item-level resampling is anti-conservative there; the
    ``clusters`` argument is what those experiments pass.
    """
    n_items = g.shape[0]
    vals = np.empty(n_boot)
    if clusters is None:
        for b in range(n_boot):
            idx = rng.integers(0, n_items, size=n_items)
            vals[b] = stat(g[idx])
    else:
        uniq = np.unique(clusters)
        members = [np.flatnonzero(clusters == c) for c in uniq]
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[p] for p in pick])
            vals[b] = stat(g[idx])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# C7 -- twin licensing (stated in Cor 1(a), verified here)
# ---------------------------------------------------------------------------


@check("C7_twin_licensing", "Cor 1(a), twin licensing")
def c7(rng: np.random.Generator, n: int) -> Result:
    """kappa_LB from sound-but-incomplete mining lower-bounds the true rate.

    Mining is SOUND (a constructed twin proves the item interface-limited) and
    INCOMPLETE (it can fail on an interface-limited item).  So the success rate
    estimates q * kappa_IL <= kappa_IL, and a Clopper-Pearson lower bound on the
    success rate is a valid one-sided lower bound for kappa_IL.  The statistic is
    the empirical non-coverage rate, which must not exceed the nominal alpha.
    """
    alpha = 0.05
    non_coverage = 0
    over_estimates = 0
    for _ in range(n):
        n_items = int(rng.integers(50, 400))
        kappa_true = float(rng.uniform(0.05, 0.95))
        q = float(rng.uniform(0.2, 1.0))  # mining recall: incompleteness
        is_limited = rng.random(n_items) < kappa_true
        mined = is_limited & (rng.random(n_items) < q)  # soundness: never fires otherwise
        k = int(mined.sum())
        khat = k / n_items
        lower = clopper_pearson_lower(k, n_items, alpha)
        if khat > kappa_true + 1e-12:
            over_estimates += 1  # point estimate should not exceed the truth often
        if lower > kappa_true:
            non_coverage += 1  # the BOUND must not exceed the truth
    rate = non_coverage / n
    return Result(
        "C7_twin_licensing",
        "Cor 1(a), twin licensing",
        n,
        "one-sided lower-bound non-coverage rate  [must be <= alpha]",
        float(rate),
        alpha,
        rate <= alpha,
        {"alpha": alpha, "point_estimate_exceeds_truth_frac": over_estimates / n},
    )


# ---------------------------------------------------------------------------
# C8 -- threshold optimality
# ---------------------------------------------------------------------------


@check("C8_threshold_optimality", "Prop: threshold optimality")
def c8(rng: np.random.Generator, n: int) -> Result:
    """The cost-adjusted optimal {stop, enc} policy is a threshold on the score.

    Let e(s) = E[Delta(enc) | IAS = s] be non-increasing.  Then choosing enc
    exactly when e(s) >= lambda*c(enc) is optimal among ALL functions of the
    score, and that rule is a threshold in s.  Verified by brute force over every
    subset-policy on a discretized score grid: the best unrestricted policy is
    compared against the best threshold policy, and against the closed-form tau.
    """
    worst_gap = 0.0
    worst_tau = 0.0
    for _ in range(n):
        # Score grid, indexed in ASCENDING score order, so "a prefix of the grid"
        # is exactly "the low-score items" -- i.e. a threshold rule.
        m = int(rng.integers(3, 11))
        e = np.sort(rng.uniform(-0.5, 1.0, size=m))[::-1]  # non-increasing in s
        w = rng.dirichlet(np.ones(m))  # score distribution
        lam = float(rng.uniform(0.01, 2.0))
        c_enc = float(rng.uniform(0.05, 2.0))
        gain = e - lam * c_enc  # cost-adjusted gain of enc; stop gains 0

        # best unrestricted policy: choose enc wherever it pays
        best_any = float((w * np.maximum(gain, 0.0)).sum())
        # best threshold policy: enc on a prefix of the (ascending-s) grid
        best_thr = max(float((w[: j + 1] * gain[: j + 1]).sum()) for j in range(-1, m))
        worst_gap = max(worst_gap, best_any - best_thr)

        # closed form: enc iff e(s) >= lambda*c(enc); with e non-increasing this
        # is a prefix, so the induced value must match the unrestricted optimum
        take = e >= lam * c_enc
        closed = float((w * gain * take).sum())
        worst_tau = max(worst_tau, abs(best_any - closed))
    return Result(
        "C8_threshold_optimality",
        "Prop: threshold optimality",
        n,
        "max(best_unrestricted - best_threshold)  [must be 0]",
        float(worst_gap),
        ATOL,
        worst_gap <= ATOL and worst_tau <= ATOL,
        {"max_closed_form_gap": float(worst_tau)},
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

# Instance counts, chosen so every check runs in seconds and matches the counts
# the manuscript reports.  --scale multiplies them all.
DEFAULT_N = {
    "C1a_attainable_accuracy": 300,
    "C1b_certified_is_exactly_half": 300,
    "C1c_deterministic_interface_separates": 300,
    "C1d_every_delta_attained": 300,
    "C2_necessity": 300,
    "C3_aggregate_ceiling": 400,
    "C4a_domination": 500,
    "C4b_sufficiency_error": 500,
    "C5a_regret_bound": 500,
    "C5b_shift_invariance": 500,
    "C6a_routing_value_bounds_gain": 300,
    "C6b_routing_value_zero_case": 100,
    "C7_twin_licensing": 300,
    "C8_threshold_optimality": 500,
    "C9_plugin_routing_value_is_biased": 300,
    "C9b_splitsample_gate_is_valid": 300,
}


def run_all(seed: int = 0, scale: float = 1.0) -> list[Result]:
    """Run every registered check with an independent, reproducible stream."""
    results = []
    for i, (name, claim, fn) in enumerate(CHECKS):
        rng = np.random.default_rng(seed + 1000 * i)  # independent per check
        n = max(1, int(DEFAULT_N.get(name, 100) * scale))
        results.append(fn(rng, n))
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="", help="CSV path (default: $OUT or stdout only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args(argv)

    import os

    out = args.out or os.environ.get("OUT", "")

    results = run_all(seed=args.seed, scale=args.scale)

    # --- machine-readable form -------------------------------------------
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "check", "claim", "n", "statistic", "value", "tolerance",
                    "passed", "extra", "checks_version", "seed",
                ]
            )
            for r in results:
                w.writerow(
                    [
                        r.check, r.claim, r.n, r.statistic, repr(r.value),
                        repr(r.tolerance), int(r.passed),
                        ";".join(f"{k}={v}" for k, v in r.extra.items()),
                        CHECKS_VERSION, args.seed,
                    ]
                )

    # --- human/LLM-readable form -----------------------------------------
    n_pass = sum(r.passed for r in results)
    print(
        f"exp=theory_checks version={CHECKS_VERSION} seed={args.seed} "
        f"scale={args.scale} eps={EPS:.6e} atol={ATOL:.3e} out={out or '-'}"
    )
    for r in results:
        print(
            f"check={r.check} claim=\"{r.claim}\" n={r.n} "
            f"stat=\"{r.statistic}\" value={r.value:.6e} tol={r.tolerance:.3e} "
            f"pass={int(r.passed)}"
            + ("".join(f" {k}={v}" for k, v in r.extra.items()) if r.extra else "")
        )
    print(f"RESULT checks={len(results)} passed={n_pass} failed={len(results)-n_pass}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
