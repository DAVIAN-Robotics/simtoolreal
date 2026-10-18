"""The 32-number BPS shape code per cap16 object: PCA of the BPS over the 3,200 objects, first 32 scores, each divided
by its spread over the pool so no value reaches the env's +-10 observation clamp.
Usage: make_bps32.py <shape_codes.npz> <out npz>
"""
import sys
import numpy as np
z = np.load(sys.argv[1])
X, names = z["bps"].astype(np.float64), z["names"]
mean = X.mean(0)
_, S, Vt = np.linalg.svd(X - mean, full_matrices=False)
score = (X - mean) @ Vt[:32].T
sigma = score.std(0)
code = score / sigma
print("explained", (S[:32] ** 2).sum() / (S ** 2).sum())
print("objects with a raw score past 10:", int((np.abs(score) > 10).any(1).sum()))
print("whitened abs max per dim", np.abs(code).max(0).round(2))
assert np.abs(code).max() < 10
np.savez(sys.argv[2], names=names, code=code.astype(np.float32), sigma=sigma.astype(np.float32),
         mean=mean.astype(np.float32), components=Vt[:32].astype(np.float32), basis=z["basis"])
