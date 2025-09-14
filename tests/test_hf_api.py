from acmecli.metrics.hf_api import freshness_days_since_update, popularity_downloads_likes


def test_popularity_extremes():
    s0, _ = popularity_downloads_likes(0, 0)
    s1, _ = popularity_downloads_likes(1_000_000, 100_000)
    assert 0 <= s0 <= 1 and 0 <= s1 <= 1
    assert s1 >= s0


def test_freshness_days_since_update():
    fresh, _ = freshness_days_since_update(0)
    stale, _ = freshness_days_since_update(365)
    assert fresh == 1.0
    assert 0 <= stale <= 1
    assert fresh >= stale
