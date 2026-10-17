"""dacomp: dependence-aware inference for comparing predictive models.

The package implements the procedures studied in the paper
"Are Machine Learning Improvements Statistically Real? Dependence-Aware
Inference for Reliable Model Comparison on Spatiotemporal Data".
"""

from .inference import (
    TestResult,
    block_bootstrap,
    corrected_resampled_t,
    da_hac,
    dm_hac,
    fold_means,
    hac_variance,
    iid_bootstrap,
    mixed_model,
    paired_t_folds,
)
from .splitting import blocked_folds, make_blocks, random_folds, training_mask
from .variogram import empirical_variogram, fit_exponential_variogram, dependence_ranges

__all__ = [
    "TestResult",
    "block_bootstrap",
    "blocked_folds",
    "corrected_resampled_t",
    "da_hac",
    "dependence_ranges",
    "dm_hac",
    "empirical_variogram",
    "fit_exponential_variogram",
    "fold_means",
    "hac_variance",
    "iid_bootstrap",
    "make_blocks",
    "mixed_model",
    "paired_t_folds",
    "random_folds",
    "training_mask",
]
