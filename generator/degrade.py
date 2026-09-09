"""Independently controllable image degradation.

Each axis takes a numeric parameter so it can be swept one at a time while the
others stay clean. Compounding them is a separate, deliberate experiment.
"""


def degrade(img, rotation=0, blur=0, jpeg_q=100, scale=1.0,
            perspective=0.0, shadow=0.0, noise=0.0):
    raise NotImplementedError
