import socket

import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from modules.datasets import DATASETS, load_dataset


# Golden numeric values from the attributed public CSVs, in source row order.
SALARY_X = [
    1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7,
    3.9, 4.0, 4.0, 4.1, 4.5, 4.9, 5.1, 5.3, 5.9, 6.0,
    6.8, 7.1, 7.9, 8.2, 8.7, 9.0, 9.5, 9.6, 10.3, 10.5
]
SALARY_Y = [
    39343, 46205, 37731, 43525, 39891, 56642, 60150, 54445, 64445, 57189,
    63218, 55794, 56957, 57081, 61111, 67938, 66029, 83088, 81363, 93940,
    91738, 98273, 101302, 113812, 109431, 105582, 116969, 112635, 122391, 121872
]
STUDY_X = [
    2.5, 5.1, 3.2, 8.5, 3.5, 1.5, 9.2, 5.5, 8.3, 2.7, 7.7, 5.9, 4.5,
    3.3, 1.1, 8.9, 2.5, 1.9, 6.1, 7.4, 2.7, 4.8, 3.8, 6.9, 7.8
]
STUDY_Y = [
    21, 47, 27, 75, 30, 20, 88, 60, 81, 25, 85, 62, 41,
    42, 17, 95, 30, 24, 67, 69, 30, 54, 35, 76, 86
]


@pytest.fixture
def offline_elsewhere(monkeypatch, tmp_path):
    def reject_network(*args, **kwargs):
        raise AssertionError("Dataset loading must work offline")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize(
    "name, expected_x, expected_y",
    [
        ("Salary vs Years of Experience", SALARY_X, SALARY_Y),
        ("Study Hours vs Exam Score", STUDY_X, STUDY_Y)
    ]
)
def test_csv_rows_are_preserved_offline(name, expected_x, expected_y, offline_elsewhere):
    X, y = load_dataset(name)

    assert X.shape == (len(expected_x), 1)
    assert y.shape == (len(expected_y),)
    np.testing.assert_array_equal(X[:, 0], expected_x)
    np.testing.assert_array_equal(y, expected_y)
    assert np.isfinite(X).all()
    assert np.isfinite(y).all()


def test_diabetes_is_all_442_rows_with_only_standardized_bmi(offline_elsewhere):
    expected = load_diabetes()
    X, y = load_dataset("Diabetes BMI vs Disease Progression")

    assert X.shape == (442, 1)
    assert y.shape == (442,)
    bmi_index = expected.feature_names.index("bmi")
    np.testing.assert_array_equal(X[:, 0], expected.data[:, bmi_index])
    np.testing.assert_array_equal(y, expected.target)
    assert X.mean() == pytest.approx(0, abs=1e-12)
    assert np.sum(X ** 2) == pytest.approx(1)


@pytest.mark.parametrize("name", ["Synthetic", "Unknown dataset"])
def test_curated_loader_rejects_non_curated_choices(name):
    with pytest.raises(ValueError, match="Not a curated dataset"):
        load_dataset(name)


def test_catalog_preserves_default_and_documents_units_and_sources():
    assert list(DATASETS) == [
        "Synthetic",
        "Salary vs Years of Experience",
        "Study Hours vs Exam Score",
        "Diabetes BMI vs Disease Progression"
    ]
    for name, info in DATASETS.items():
        assert info["description"]
        assert info["x_label"]
        assert info["y_label"]
        if name != "Synthetic":
            assert info["source_name"]
            assert info["source_url"].startswith("https://")
