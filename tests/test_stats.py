import math

from presup.stats import (as_float, benjamini_hochberg, bootstrap, ols_slope, permutation_p,
                          spearman)


def test_as_float():
    assert as_float("0.25") == 0.25
    assert as_float("") is None
    assert as_float(None) is None


def test_bootstrap_constant():
    assert bootstrap([0.3] * 10) == (0.3, 0.3)
    assert all(math.isnan(x) for x in bootstrap([0.3]))


def test_spearman_and_slope():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert ols_slope([0, 1, 2], [1, 3, 5]) == 2.0


def test_permutation_p():
    assert permutation_p([1.0] * 12) < 0.001
    assert permutation_p([1.0, -1.0] * 6) == 1.0


def test_benjamini_hochberg():
    # cutoffs at q=0.05 for 4 tests: 0.0125, 0.025, 0.0375, 0.05
    assert benjamini_hochberg([0.01, 0.02, 0.04, 0.5]) == [True, True, False, False]
    assert benjamini_hochberg([0.03, 0.035, 0.04, 0.045]) == [True, True, True, True]
    assert benjamini_hochberg([float("nan"), 0.001]) == [False, True]
