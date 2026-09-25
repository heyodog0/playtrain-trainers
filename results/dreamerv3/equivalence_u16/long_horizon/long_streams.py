"""Identical input streams for U16e, regenerated from per-step seeds on both sides."""
import numpy as np
SHAPES = {'k': (256, 128), 'b': (128,), 's': (128,), 'blk': (8, 64, 96)}
def init_params():
    r = np.random.default_rng(12345)
    return {'k': (r.standard_normal(SHAPES['k']) * 0.05).astype(np.float32), 'b': np.zeros(128, np.float32),
            's': np.ones(128, np.float32), 'blk': (r.standard_normal(SHAPES['blk']) * 0.02).astype(np.float32)}
def grads(t):
    """Nonstationary: a log-scale random walk per tensor, heavy-tailed noise, spikes that
    trigger AGC, and occasional all-zero gradients."""
    r = np.random.default_rng([7, t]); out = {}
    for i, (k, sh) in enumerate(SHAPES.items()):
        scale = 10 ** (2 * np.sin(t / (400 + 97 * i)) - 1)
        g = r.standard_t(3, sh) * scale
        if r.random() < 0.01: g = g * 300
        if r.random() < 0.005: g = g * 0
        out[k] = g.astype(np.float32)
    return out
def returns(t):
    r = np.random.default_rng([11, t])
    drift = 5 * np.sin(t / 700) + t / 2000; scale = 0.2 + 3 * (1 + np.sin(t / 333)) ** 2
    return (drift + scale * r.standard_normal((1024, 15)) + (r.random((1024, 15)) < 0.02) * 40).astype(np.float32)
def source(t):
    r = np.random.default_rng([13, t])
    return {'kernel': (np.sin(t / 250) + 0.3 * r.standard_normal((8, 16))).astype(np.float32),
            'bias': (np.cos(t / 90) + 0.1 * r.standard_normal(16)).astype(np.float32)}
