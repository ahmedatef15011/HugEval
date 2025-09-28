from acmecli.scoring import clamp01


def test_clamp_basic_bounds():
    assert clamp01(-1) == 0.0
    assert clamp01(2) == 1.0
    assert clamp01(0.5) == 0.5
