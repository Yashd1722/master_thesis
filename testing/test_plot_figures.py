"""Self-checks for the non-trivial bits of plot_figures.py.

Run: python testing/test_plot_figures.py
"""
import numpy as np
from pathlib import Path
import tempfile

from testing.plot_figures import _favored_class_freq, plot_confusion_matrix


def test_favored_class_freq():
    # window 0 -> fold, window 1 -> null, window 2 -> null
    rec = {
        "p_fold":          [0.7, 0.1, 0.1],
        "p_hopf":          [0.1, 0.2, 0.2],
        "p_transcritical": [0.1, 0.2, 0.2],
        "p_null":          [0.1, 0.5, 0.5],
    }
    freq = _favored_class_freq(rec)
    assert np.isclose(freq.sum(), 1.0)
    assert np.allclose(freq, [1 / 3, 0, 0, 2 / 3])          # [fold, hopf, trans, null]
    assert _favored_class_freq({"p_fold": []}) is None       # missing tracks


def test_confusion_matrix_row_normalised():
    # imbalanced rows: the plot must normalise per row, not by global max
    data = {
        "model": "x", "accuracy": 0.5,
        "confusion_matrix": [[90, 10], [1, 1]],
    }
    with tempfile.TemporaryDirectory() as d:
        plot_confusion_matrix(data, Path(d), ["a", "b"], "cm.png")
        assert (Path(d) / "cm.png").exists()
    cm = np.array(data["confusion_matrix"], float)
    norm = cm / cm.sum(axis=1, keepdims=True)
    assert np.allclose(norm.sum(axis=1), 1.0)
    assert np.allclose(norm[1], [0.5, 0.5])


if __name__ == "__main__":
    test_favored_class_freq()
    test_confusion_matrix_row_normalised()
    print("ok")
