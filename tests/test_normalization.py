import json

import numpy as np
import pytest

from cos1.analysis import statistics
from cos1.normalization import (
    Parameters,
    apply,
    fit_group_z,
    fit_individual,
    fit_reference,
    load_parameters,
    mean_square_spectrum,
    save_parameters,
)


@pytest.mark.parametrize(
    "method,expected",
    [
        ("none", [1, 2, 3]),
        ("individual_z", [-np.sqrt(1.5), 0, np.sqrt(1.5)]),
        ("max_abs", [1 / 3, 2 / 3, 1]),
        ("rms", np.array([1, 2, 3]) / np.sqrt(14 / 3)),
    ],
)
def test_individual_formula_and_no_mutation(method, expected):
    x = np.array([1.0, 2.0, 3.0])
    old = x.copy()
    y = apply(x, fit_individual(x, method))
    np.testing.assert_allclose(y, expected, atol=1e-15)
    np.testing.assert_array_equal(x, old)
    assert not np.shares_memory(x, y)


def test_group_mixture_weights_between_means():
    # Equal record weights: short [0,2] and long [4,4,4,4], mean=2.5.
    p = fit_group_z({"short": np.array([0.0, 2.0]), "long": np.full(4, 4.0)})
    assert p.offset == 2.5
    assert p.scale == pytest.approx(np.sqrt(2.75))
    assert p.offset != pytest.approx(3.0)  # sample-weighted mean


def test_reference_median_rms():
    p = fit_reference({"a": np.array([3.0, -3.0]), "b": np.full(7, 5.0)})
    assert p.scale == 4
    np.testing.assert_allclose(apply([4, 8], p), [1, 2])


@pytest.mark.parametrize("method", ["individual_z", "max_abs", "rms"])
def test_zero_rejected_without_epsilon(method):
    with pytest.raises(ValueError):
        fit_individual(np.zeros(8), method)


def test_constant_policy():
    x = np.full(8, -4.0)
    with pytest.raises(ValueError):
        fit_individual(x, "individual_z")
    with pytest.raises(ValueError):
        fit_group_z({"a": x})
    with pytest.raises(ValueError):
        fit_reference({"zero": np.zeros(8)})
    for method in ("max_abs", "rms"):
        np.testing.assert_array_equal(apply(x, fit_individual(x, method)), -np.ones(8))
    np.testing.assert_array_equal(
        apply(np.zeros(8), fit_individual(np.zeros(8), "none")), np.zeros(8)
    )
    assert statistics(x)["excess"] is None


@pytest.mark.parametrize("bad", [[], [np.nan], [np.inf], [[1, 2]], [1 + 1j]])
def test_invalid_input(bad):
    with pytest.raises(ValueError):
        fit_individual(bad, "none")


@pytest.mark.parametrize("scale", [0, -1, np.inf, np.nan])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):
        Parameters("rms", 0, scale)


def test_save_reuse_new_data_without_refit(tmp_path):
    p = fit_group_z({"train_a": np.array([1, 2, 3])})
    path = tmp_path / "parameters.json"
    save_parameters(path, p)
    q = load_parameters(path)
    assert p == q
    np.testing.assert_allclose(
        apply([10, 20], q), (np.array([10, 20]) - 2) / np.sqrt(2 / 3)
    )
    assert apply([10, 20], q).mean() != 0
    json.loads(path.read_text())


def test_six_x_z_equivalence():
    x = np.array([-4.0, 2.0, 1.0, 8.0, 9.0])
    np.testing.assert_allclose(
        apply(x, fit_individual(x, "individual_z")),
        apply(6 * x, fit_individual(6 * x, "individual_z")),
        atol=1e-15,
    )


@pytest.mark.parametrize("offset,scale", [(0, 2), (4, 3), (-10, 0.2)])
def test_affine_statistics(offset, scale):
    x = np.array([-4.0, 2.0, 1.0, 8.0, 9.0])
    p = Parameters("group_z", offset, scale)
    a = statistics(x)
    b = statistics(apply(x, p))
    assert b["mean"] == pytest.approx((a["mean"] - offset) / scale)
    assert b["std_centered"] == pytest.approx(a["std_centered"] / scale)
    assert b["rms_raw"] == pytest.approx(
        np.hypot(a["std_centered"], a["mean"] - offset) / scale
    )
    assert b["excess"] == pytest.approx(a["excess"])
    assert b["crest_centered"] == pytest.approx(a["crest_centered"])
    if offset == 0:
        assert b["crest_raw"] == pytest.approx(a["crest_raw"])


@pytest.mark.parametrize("n", [1, 2, 9, 10])
@pytest.mark.parametrize("center", [True, False])
def test_parseval_even_odd_dc_nyquist(n, center):
    x = np.arange(n, dtype=float) + 2
    f, p = mean_square_spectrum(x, 51200, center)
    expected = np.mean((x - x.mean()) ** 2) if center else np.mean(x * x)
    assert p.sum() == pytest.approx(expected, abs=1e-13)
    assert len(f) == len(p) == n // 2 + 1


def test_real_cohort_and_theory():
    from cos1.normalization_study import load_study, fit_common, evaluate

    config, meta, arrays = load_study()
    fitted, membership = fit_common(config, meta, arrays)
    _, _, metrics, checks = evaluate(config, meta, arrays, fitted, membership)
    assert len(metrics) == 102
    assert metrics.status.eq("ok").sum() == 89
    assert metrics.status.eq("not_applicable").sum() == 13
    assert checks.passed.all()
    assert len(checks) == 89
    assert fitted["group_z:HUST_X"]["parameters"].fit_ids == (
        "hustbearing_medium_ball_65_full_X",
    )
    assert (
        membership["hustbearing_medium_ball_65", "group_z"]
        == membership["hustbearing_medium_ball_65_full_X", "group_z"]
    )


def _update_fixture(tmp_path):
    import hashlib
    import runpy
    from cos1.data import ROOT

    source, target = tmp_path / "update", tmp_path / "project"
    source.mkdir()
    target.mkdir()
    (source / "new.txt").write_text("new")
    (target / "base.txt").write_text("base")
    manifest = {
        "required_project_files": {"base.txt": hashlib.sha256(b"base").hexdigest()},
        "payload": {"new.txt": hashlib.sha256(b"new").hexdigest()},
    }
    (source / "STAGE3_UPDATE_MANIFEST.json").write_text(json.dumps(manifest))
    install = runpy.run_path(str(ROOT / "scripts/apply_stage3_update.py"))["install"]
    return source, target, install


def test_update_dry_run_apply_idempotent(tmp_path):
    source, target, install = _update_fixture(tmp_path)
    assert install(source, target) == 1
    assert not (target / "new.txt").exists()
    assert install(source, target, apply=True) == 1
    assert install(source, target, apply=True) == 0
    assert (target / "base.txt").read_text() == "base"


def test_update_local_conflict_preserved(tmp_path):
    source, target, install = _update_fixture(tmp_path)
    (target / "new.txt").write_text("user work")
    with pytest.raises(ValueError, match="LOCAL CONFLICT"):
        install(source, target, apply=True)
    assert (target / "new.txt").read_text() == "user work"


def test_update_changed_base_stops_before_copy(tmp_path):
    source, target, install = _update_fixture(tmp_path)
    (target / "base.txt").write_text("changed")
    with pytest.raises(ValueError, match="BASE MISMATCH"):
        install(source, target, apply=True)
    assert not (target / "new.txt").exists()


def test_update_corrupt_payload_stops(tmp_path):
    source, target, install = _update_fixture(tmp_path)
    (source / "new.txt").write_text("corrupt")
    with pytest.raises(ValueError, match="UPDATE CORRUPT"):
        install(source, target, apply=True)
    assert not (target / "new.txt").exists()
