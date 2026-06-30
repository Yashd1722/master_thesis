"""
src/constants.py
Single source of truth for class ordering and key indices.

Bury et al. (2021) DOI: 10.1073/pnas.2106140118 defines four bifurcation types.
The ordering below is the canonical mapping used throughout this codebase.
"""

# Class index → bifurcation type, matching Bury's label CSV exactly.
CLASS_NAMES = ["fold", "hopf", "transcritical", "null"]

# Index of the "null" (no-transition) class — used to compute p(transition).
# p(transition) = 1 - probs[:, NULL_IDX]
NULL_IDX = 3
