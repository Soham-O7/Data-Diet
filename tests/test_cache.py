import numpy as np

from dataprune.embedders import CachedEmbedder


class _Counting:
    name = "counting"

    def __init__(self):
        self.calls = 0

    def transform(self, imgs):
        self.calls += len(imgs)
        return np.asarray(imgs, dtype=np.float32).reshape(len(imgs), -1)[:, :4]


def test_cached_embedder_embeds_each_row_once():
    base = _Counting()
    c = CachedEmbedder(base)
    X = np.random.default_rng(0).integers(0, 255, (50, 8, 8), dtype=np.uint8)
    Z1 = c.fit_transform(X)
    Z2 = c.fit_transform(X)
    Z3 = c.transform(X[10:30][::-1])
    assert base.calls == 50 and c.n_embedded == 50
    assert np.array_equal(Z1, Z2) and np.array_equal(Z3, Z1[10:30][::-1])
    X2 = np.concatenate([X[:5], X[:5] + 1])           # 5 seen rows + 5 new rows
    c.transform(X2)
    assert base.calls == 55