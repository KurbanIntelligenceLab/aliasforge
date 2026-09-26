#!/usr/bin/env python3
"""Unit tests for the invariants the experiments depend on.

These are cheap, deterministic and local -- they guard the properties that, if
they silently broke, would produce results that look fine and are wrong.  Run:

    python tests/test_core.py
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from aliasforge.aliasforge import (  # noqa: E402
    box_carrier, cell_sums_match, glyph_mask, make_s1_pair, nullspace,
    probe_row_operator, upsample_nearest,
)
from aliasforge.influence import (  # noqa: E402
    dead_pixel_mask, largest_dead_rectangle, make_dead_region_pair,
)
from aliasforge.interface import field_digest, per_field_report, state_digest  # noqa: E402
from aliasforge.metrics import ATOL, run_all  # noqa: E402

FAILURES = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- digests -----------------------------------------------------------------

def test_digest_sensitivity():
    a = {"pixel_values": np.zeros((2, 3), np.float32)}
    b = {"pixel_values": np.zeros((2, 3), np.float32)}
    check("digest: identical states collide", state_digest(a) == state_digest(b))

    b2 = {"pixel_values": np.zeros((2, 3), np.float64)}   # dtype only
    check("digest: dtype change is detected", state_digest(a) != state_digest(b2))

    b3 = {"pixel_values": np.zeros((3, 2), np.float32)}   # shape only
    check("digest: shape change is detected", state_digest(a) != state_digest(b3))

    b4 = {"pixel_values": np.zeros((2, 3), np.float32), "grid": (1, 2)}  # extra field
    check("digest: an added field is detected", state_digest(a) != state_digest(b4))

    b5 = {"pixel_value": np.zeros((2, 3), np.float32)}    # rename only
    check("digest: a renamed field is detected", state_digest(a) != state_digest(b5))

    v = np.zeros((2, 3), np.float32)
    v2 = v.copy(); v2[1, 2] = np.float32(1e-7)            # one tiny value
    check("digest: a single 1e-7 change is detected", field_digest(v) != field_digest(v2))

    # -0.0 vs 0.0 compare equal numerically but are different bits; the digest
    # must follow the BITS, since that is what the model actually receives.
    z, nz = np.zeros(3, np.float32), np.array([-0.0, 0.0, 0.0], np.float32)
    check("digest: -0.0 vs 0.0 is detected (bitwise, not numeric)",
          field_digest(z) != field_digest(nz))


def test_per_field_report():
    a = {"x": np.zeros(3), "input_ids": np.ones(2)}
    b = {"x": np.ones(3), "input_ids": np.ones(2)}
    r = per_field_report(a, b)
    check("report: names the differing image field", r["image_fields_differ"] == ["x"])
    check("report: does not certify when a field differs", r["collides"] is False)
    r2 = per_field_report(a, dict(a))
    check("report: certifies identical states", r2["collides"] is True)


# --- S1 construction ---------------------------------------------------------

def test_box_carrier_sums_to_zero():
    for s in (2, 3, 4, 5, 8):
        c = box_carrier(s, s * 6)
        sums = c.reshape(-1, s).sum(axis=1)
        check(f"carrier: every cell sums to zero exactly (s={s})", bool((sums == 0).all()),
              f"sums={sums[:4]}")


def test_glyph_and_upsample():
    g = glyph_mask("3", 12, 8, scale=2, top=0, left=0)
    check("glyph: renders non-empty content", g.sum() > 0)
    check("glyph: differs between digits",
          not np.array_equal(glyph_mask("3", 12, 8, scale=2), glyph_mask("8", 12, 8, scale=2)))
    u = upsample_nearest(np.array([[1, 0]]), 2, 3)
    check("upsample: exact replication", u.shape == (2, 6) and u[0, 0] == 1 and u[0, 3] == 0)


def test_s1_pair_invariants():
    H = W = 64
    base = np.full((H, W, 3), 128, np.uint8)
    x0, x1, meta = make_s1_pair(base, "3", "8", cell=4, amplitude=30, scale=2, top=1, left=1)
    check("S1: pair is constructed", x0 is not None)
    if x0 is None:
        return
    check("S1: cell sums match exactly", cell_sums_match(x0, x1, 4))
    check("S1: members actually differ at full resolution", meta["pix_diff"] > 0)
    check("S1: difference is a small fraction of pixels", meta["pix_diff"] < 0.25 * meta["pix_total"],
          f"{meta['pix_diff']}/{meta['pix_total']}")

    # saturation must be REJECTED, not silently clipped -- a clip is nonlinear
    # and would break the very cancellation the construction relies on.
    bright = np.full((H, W, 3), 250, np.uint8)
    x0b, _x1b, metab = make_s1_pair(bright, "3", "8", cell=4, amplitude=60, scale=2)
    check("S1: saturating candidate is rejected, not clipped",
          x0b is None and metab["saturates"] is True)


def test_nullspace_is_annihilated():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(6, 20))
    N = nullspace(A)
    check("nullspace: correct dimension", N.shape[1] == 14, f"{N.shape}")
    check("nullspace: A @ N is zero", float(np.abs(A @ N).max()) < 1e-10)


def test_probe_recovers_box_operator():
    # An exact box downscale, so the probed operator has a known null space.
    H, W, cell = 24, 24, 4
    def rs(X):
        return X.reshape(H // cell, cell, W // cell, cell).mean((1, 3))
    A = probe_row_operator(rs, H, W)
    check("probe: recovers operator of right shape", A.shape == (H // cell, H))
    c = box_carrier(cell, H).astype(float)
    check("probe: the integer cell carrier lies in the probed null space",
          float(np.abs(A @ c).max()) < 1e-10)


# --- dead-region construction ------------------------------------------------

def test_dead_region_pair():
    H = W = 64
    img = np.full((H, W, 3), 100, np.uint8)
    infl = np.ones((4, 4), np.uint8)
    infl[2:, 2:] = 0                       # bottom-right quadrant ignored
    dead = dead_pixel_mask(infl, 16, H, W)
    check("dead mask: expands to full resolution", dead.shape == (H, W))
    check("dead mask: marks exactly the ignored quadrant", dead.sum() == 32 * 32)

    a, b, meta = make_dead_region_pair(img, dead, "3", "8")
    check("dead pair: constructed", a is not None)
    if a is None:
        return
    check("dead pair: members differ", meta["pix_diff"] > 0)
    # The whole point: nothing outside the ignored region may move.
    live = ~dead
    check("dead pair: NO live pixel is touched",
          np.array_equal(a[live], img[live]) and np.array_equal(b[live], img[live]))
    check("dead pair: members agree outside the dead region",
          np.array_equal(a[live], b[live]))

    empty, _, m2 = make_dead_region_pair(img, np.zeros((H, W), bool), "3", "8")
    check("dead pair: refuses when there is no dead region",
          empty is None and "no dead region" in m2.get("reason", ""))


def test_largest_dead_rectangle():
    m = np.zeros((10, 12), bool)
    m[2:8, 3:10] = True
    check("max-rect: recovers an exact rectangle", largest_dead_rectangle(m) == (2, 3, 6, 7))

    # A center crop leaves two disjoint side bands.  The bounding box of the dead
    # SET spans the whole image and is mostly live, so the constructor must pick
    # a band -- this is the bug that silently produced identical "certified" pairs.
    bands = np.zeros((333, 500), bool)
    bands[:, :80] = True
    bands[:, 420:] = True
    y, x, h, w = largest_dead_rectangle(bands)
    check("max-rect: picks one band, not the bounding box", (h, w) == (333, 80), f"{(y,x,h,w)}")
    check("max-rect: every pixel inside is dead", bool(bands[y:y + h, x:x + w].all()))

    img = np.full((333, 500, 3), 100, np.uint8)
    a, b, meta = make_dead_region_pair(img, bands, "3", "8")
    check("dead pair: built from side bands", a is not None)
    if a is not None:
        live = ~bands
        check("dead pair: difference lies ONLY in the dead region",
              int(((a != b).any(-1) & live).sum()) == 0)
        check("dead pair: members genuinely differ", meta["pix_diff"] > 0)


def test_degenerate_pair_is_rejected():
    """A pair whose members are identical must NOT be offered for certification.

    Equal digests are then a tautology.  This is the failure the first
    dead-region run produced: two geometries reported certified=1 with
    pix_diff=0, which is a vacuous pass, not a certified alias.
    """
    img = np.full((64, 64, 3), 100, np.uint8)
    # A dead region large enough to pass the size gate but only 1 px wide in the
    # columns where '3' and '8' differ -- engineered so the glyphs collide.
    dead = np.zeros((64, 64), bool)
    dead[0:64, 30:31] = True
    a, b, meta = make_dead_region_pair(img, dead, "3", "8")
    check("degenerate: identical members are refused",
          a is None and b is None,
          f"pix_diff={meta.get('pix_diff')}")
    check("degenerate: refusal states a reason", bool(meta.get("reason")))


# --- action set --------------------------------------------------------------

class _FakeVerifier:
    """Deterministic, image-dependent, no GPU -- enough to pin the contracts."""

    def p_correct(self, img, q, s):
        return float(np.clip(0.5 + 0.3 * np.tanh((np.asarray(img).mean() - 128) / 20.0), 0, 1))

    def interface_digest(self, img, q, s):
        return str(hash(np.asarray(img).tobytes()))[:12]

    def critique(self, img, q, s, temperature=0.8, seed=None, max_new_tokens=96):
        # a seeded stand-in for generation: different seeds -> different text
        r = np.random.default_rng(seed)
        return {"text": f"critique-{r.integers(1<<30)}", "prefill_tokens": 12,
                "decode_tokens": 7}

    def verdict_after_critique(self, img, q, s, text):
        # the critique perturbs the verdict, as a generated one would
        r = np.random.default_rng(abs(hash(text)) % (1 << 31))
        return float(np.clip(self.p_correct(img, q, s) + r.normal(0, 0.05), 0, 1))


def test_action_class_membership():
    """Definition 1: stop/att/rea must not change the interface; enc must."""
    from aliasforge.actions import ActionRunner, gains

    r = ActionRunner(_FakeVerifier(), rea_passes=3, sampled=False)
    img = np.full((448, 448, 3), 120, np.uint8)
    img[100:200, 100:200] = 200
    res = r.run_all(img, "How many marks?", "There are three.", (100, 100, 100, 100))

    d_stop = res["stop"].interface_digest
    check("actions: att is language-side (interface unchanged)",
          res["att"].interface_digest == d_stop)
    check("actions: rea is language-side (interface unchanged)",
          res["rea"].interface_digest == d_stop)
    check("actions: enc leaves the class (interface changes)",
          res["enc"].interface_digest != d_stop)

    g = gains(res, label=1)
    check("actions: stop has exactly zero gain over itself", g["stop"] == 0.0)
    # Theorem 1 forces this one: a re-encode to an identical interface cannot pay.
    check("actions: ctrl_same_interface gain is exactly zero (Thm 1)",
          abs(g["ctrl_same_interface"]) < 1e-12, f"{g['ctrl_same_interface']}")
    check("actions: same-interface control reports a stable digest",
          res["ctrl_same_interface"].extra["digest_stable"] is True)


def test_excluding_crop_is_a_real_control():
    """The target-excluding crop must not land on the target region."""
    from aliasforge.actions import ActionRunner, gains

    r = ActionRunner(_FakeVerifier(), sampled=False)
    img = np.full((448, 448, 3), 120, np.uint8)
    img[20:120, 20:120] = 220                      # decisive evidence, top-left
    box = (20, 20, 100, 100)
    res = r.run_all(img, "q", "s", box)
    ex = res["ctrl_excluding_crop"].extra["box"]
    check("control: excluding crop is not the target box", ex != box, str(ex))
    y0, x0, h, w = ex
    ty0, tx0, th, tw = box
    overlap = max(0, min(y0 + h, ty0 + th) - max(y0, ty0)) * \
              max(0, min(x0 + w, tx0 + tw) - max(x0, tx0))
    check("control: excluding crop does not overlap the target", overlap == 0, f"{overlap}px")
    g = gains(res, label=1)
    check("control: oracle crop gains more than the excluding crop",
          g["ctrl_oracle_crop"] > g["ctrl_excluding_crop"],
          f"{g['ctrl_oracle_crop']:.3f} vs {g['ctrl_excluding_crop']:.3f}")


def test_gain_uses_the_label():
    """Correctness is p when the label is 1 and 1-p when it is 0."""
    from aliasforge.actions import ActionResult, gains

    res = {"stop": ActionResult("stop", 0.4), "enc": ActionResult("enc", 0.9)}
    check("gains: label=1 rewards a higher probability",
          abs(gains(res, 1)["enc"] - 0.5) < 1e-12)
    check("gains: label=0 rewards a LOWER probability",
          abs(gains(res, 0)["enc"] + 0.5) < 1e-12)


def test_sampled_critiques_give_independent_replicates():
    """The fix for the degenerate pilot: two runs must NOT be bit-identical.

    Under deterministic scoring every replicate came back identical (80/80),
    which silently collapsed the split-sample estimator onto the plug-in one and
    made `rea` a bare prompt variant.  Sampled critiques are what restore the
    independence the estimator assumes.
    """
    from aliasforge.actions import ActionRunner, gains

    img = np.full((448, 448, 3), 120, np.uint8)
    img[100:200, 100:200] = 200
    a = ActionRunner(_FakeVerifier(), rea_passes=3, sampled=True, seed=0)
    b = ActionRunner(_FakeVerifier(), rea_passes=3, sampled=True, seed=1)
    ga = gains(a.run_all(img, "q", "s", (100, 100, 100, 100)), 1)
    gb = gains(b.run_all(img, "q", "s", (100, 100, 100, 100)), 1)
    moved = [k for k in ("att", "rea", "enc") if ga[k] != gb[k]]
    check("sampled: replicates differ on the critique-based actions",
          len(moved) == 3, f"moved={moved}")

    det = ActionRunner(_FakeVerifier(), rea_passes=3, sampled=False, seed=0)
    d1 = gains(det.run_all(img, "q", "s", (100, 100, 100, 100)), 1)
    d2 = gains(det.run_all(img, "q", "s", (100, 100, 100, 100)), 1)
    check("deterministic: replicates are identical (the degenerate case)",
          all(d1[k] == d2[k] for k in ("att", "rea", "enc")))

    # rea aggregates several passes, so under sampling they must actually spread
    res = a.run_all(img, "q", "s", (100, 100, 100, 100))
    check("sampled: rea passes have non-zero spread",
          res["rea"].extra["spread"] > 0, f"{res['rea'].extra['spread']}")
    check("sampled: critique cost is recorded (decode tokens > 0)",
          res["rea"].decode_tokens > 0 and res["att"].decode_tokens > 0)


# --- theory checks -----------------------------------------------------------

def test_metrics_all_pass():
    res = run_all(seed=123, scale=0.15)     # small but non-trivial
    bad = [r.check for r in res if not r.passed]
    check(f"metrics: all {len(res)} checks pass at seed 123", not bad, f"failed={bad}")
    c1a = next(r for r in res if r.check == "C1a_attainable_accuracy")
    check("metrics: attainable-accuracy discrepancy is at float noise",
          c1a.value <= ATOL, f"{c1a.value:.3e}")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"\n--- {fn.__name__} ---")
        fn()
    print(f"\nRESULT tests_failed={len(FAILURES)} {FAILURES if FAILURES else ''}")
    sys.exit(1 if FAILURES else 0)
