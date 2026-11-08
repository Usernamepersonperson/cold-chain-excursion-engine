import pytest

from ccee.profiles import Band, ProductProfile, get_profile


def test_builtin_profiles_have_non_overlapping_bands():
    for name in ("2-8C", "15-25C", "frozen"):
        assert get_profile(name).lo < get_profile(name).hi


def test_unknown_profile():
    with pytest.raises(KeyError):
        get_profile("nope")


def test_overlapping_bands_rejected():
    with pytest.raises(ValueError):
        ProductProfile("x", 2, 8, bands=(Band(8, 15, 10), Band(14, 20, 10)))


def test_band_overlapping_range_rejected():
    with pytest.raises(ValueError):
        ProductProfile("x", 2, 8, bands=(Band(7, 15, 10),))


def test_inverted_range_rejected():
    with pytest.raises(ValueError):
        ProductProfile("x", 8, 2)


def test_direction_and_band_lookup(p28):
    assert p28.direction(8.1) == "high" and p28.direction(1.9) == "low" and p28.direction(5) is None
    assert p28.in_range(2.0) and p28.in_range(8.0)
    assert p28.band_index(10.0) is not None
    assert p28.band_index(70.0) is None
