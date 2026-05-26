"""
Module 4 - Uncertainty Quantification.

Propagates measurement, calibration, sampling, and modelling
uncertainty through to all reported batch-level outputs of
Sub-system 5. Consumes Module 2 batches and Module 3 descriptors,
produces a BatchUncertainty object containing standard uncertainties,
95% confidence intervals, and a per-source contribution table for
every headline descriptor.

Methodology follows JCGM 100:2008 (GUM) for analytical per-fibre
propagation and JCGM 101:2008 (GUM Supplement 1) for the Monte Carlo
batch-level layer.
"""