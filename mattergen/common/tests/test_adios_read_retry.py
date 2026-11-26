import numpy as np
import pytest

adios_dataset = pytest.importorskip("mattergen.common.data.adiosDataset")


class _FlakyReader:
    """Minimal FileReader stand-in whose first `fail_times` reads raise an ADIOS2 error."""

    calls = 0

    def __init__(self, fail_times, message="[ADIOS2 EXCEPTION] couldn't read from file"):
        self.fail_times = fail_times
        self.message = message

    def read(self, variable, start, count):
        if _FlakyReader.calls < self.fail_times:
            _FlakyReader.calls += 1
            raise RuntimeError(self.message)
        if variable.endswith("variable_count"):
            return np.array([2])
        if variable.endswith("variable_offset"):
            return np.array([0])
        if variable.endswith("atomic_numbers"):
            return np.array([[1], [8]])
        if variable.endswith("cell"):
            return np.eye(3)
        return np.zeros((count[0], 3))

    def close(self):
        pass


def _make_dataset(monkeypatch, fail_times, message=None):
    _FlakyReader.calls = 0
    kwargs = {} if message is None else {"message": message}
    monkeypatch.setattr(
        adios_dataset.adios2, "FileReader", lambda _: _FlakyReader(fail_times, **kwargs)
    )
    monkeypatch.setattr(adios_dataset, "_READ_RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(adios_dataset, "_READ_RETRIES", 3)
    ds = object.__new__(adios_dataset.HydraGNNAdiosCrystalDataset)
    ds.filename = "fake.bp"
    ds.label = "trainset"
    ds.property_names = set()
    ds.f = _FlakyReader(fail_times, **kwargs)
    ds.variables = {
        f"trainset/{k}": {"Shape": "4, 3"} for k in ("atomic_numbers", "pos", "cell")
    }
    return ds


def test_transient_adios_error_is_retried(monkeypatch):
    ds = _make_dataset(monkeypatch, fail_times=2)
    atomic_numbers, pos, cell, forces = ds._read_sample_with_retry(0)
    assert atomic_numbers.tolist() == [1, 8]
    assert pos.shape == (2, 3)
    assert forces is None


def test_persistent_adios_error_is_raised(monkeypatch):
    ds = _make_dataset(monkeypatch, fail_times=100)
    with pytest.raises(RuntimeError, match="ADIOS2"):
        ds._read_sample_with_retry(0)


def test_non_adios_error_is_not_retried(monkeypatch):
    ds = _make_dataset(monkeypatch, fail_times=1, message="shape mismatch")
    with pytest.raises(RuntimeError, match="shape mismatch"):
        ds._read_sample_with_retry(0)
    assert _FlakyReader.calls == 1
