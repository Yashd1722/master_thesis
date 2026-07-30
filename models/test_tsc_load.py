"""Old checkpoints (pre-use_custom_pipeline) must still load and predict."""
import sys, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from models.tsc import TSCModel

CKPT = pathlib.Path(__file__).resolve().parents[1] / "checkpoints"


def test_load_and_predict():
    pkls = sorted(CKPT.glob("*_best.pkl"))
    assert pkls, "no checkpoints to check"
    for p in pkls:
        m = TSCModel.load(p)
        assert isinstance(m.use_custom_pipeline, bool)
        # legacy pickles route to ._clf, new ones to the transformer pipeline
        assert m.use_custom_pipeline == hasattr(m, "_transformer")
        ts_len = 500 if "ts_500" in p.name else 1500
        # eval feeds (N, 4, L) when inference.use_4channel; _prep drops to
        # channel 0 for the univariate models
        probs = m.predict_proba(np.random.randn(4, 4, ts_len))
        assert probs.shape == (4, m.num_classes), (p.name, probs.shape)


if __name__ == "__main__":
    test_load_and_predict()
    print("ok")
