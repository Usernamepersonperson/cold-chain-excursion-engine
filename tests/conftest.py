import numpy as np
import pandas as pd
import pytest

from ccee.profiles import get_profile


def frame(temps, start="2025-07-01T00:00:00Z", step_min=5):
    ts = pd.Timestamp(start) + pd.to_timedelta(np.arange(len(temps)) * step_min, unit="min")
    return pd.DataFrame({"ts": ts, "temp_c": np.asarray(temps, float)})


@pytest.fixture
def p28():
    return get_profile("2-8C")
