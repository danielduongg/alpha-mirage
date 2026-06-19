"""Tests for alpha-mirage: the data-snooping mirage must reproduce."""
import numpy as np
from experiment import search

def test_search_structure():
    r = search("randomwalk", seed=1)
    for k in ("in_sample_sharpe","out_of_sample_sharpe","deflated_sharpe","K","best_fast","best_slow"):
        assert k in r
    assert r["best_fast"] < r["best_slow"]

def test_in_sample_beats_out_of_sample():
    rows = [search("randomwalk", seed=s) for s in range(25)]
    mean_in = np.mean([r["in_sample_sharpe"] for r in rows])
    mean_out = np.mean([r["out_of_sample_sharpe"] for r in rows])
    assert mean_in > 0.4          # cherry-picked winners look good
    assert mean_in - mean_out > 0.4   # and the edge evaporates out-of-sample

def test_random_walk_has_no_real_edge():
    rows = [search("randomwalk", seed=s) for s in range(25)]
    mean_out = np.mean([r["out_of_sample_sharpe"] for r in rows])
    assert abs(mean_out) < 0.3    # OOS Sharpe ~ 0 (no edge by construction)

def test_deflated_sharpe_fails_significance():
    rows = [search("randomwalk", seed=s) for s in range(25)]
    assert np.mean([r["deflated_sharpe"] for r in rows]) < 0.95
