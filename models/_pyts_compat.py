"""pyts 0.13.0 (last released 2023) calls the deprecated scipy sparse `.A`
accessor, removed in current scipy. This patches it back as an alias for
`.toarray()` so pyts's BagOfPatterns/SAXVSM run unmodified — it doesn't touch
pyts's actual algorithm, just an attribute name scipy renamed after pyts's
last release. Import this before importing anything from pyts.
"""
import scipy.sparse

for _cls in (scipy.sparse.csr_matrix, scipy.sparse.csc_matrix):
    if not hasattr(_cls, "A"):
        _cls.A = property(lambda self: self.toarray())
