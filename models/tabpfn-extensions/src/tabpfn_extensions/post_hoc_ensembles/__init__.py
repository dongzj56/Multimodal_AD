try:
    from . import abstract_validation_utils, greedy_weighted_ensemble, pfn_phe
    from .greedy_weighted_ensemble import (
        GreedyWeightedEnsemble,
        GreedyWeightedEnsembleClassifier,
        GreedyWeightedEnsembleRegressor,
    )
    from .sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor
except ImportError:
    raise ImportError(
        "Please install tabpfn-extensions with the 'post_hoc_ensembles' extra: pip install 'tabpfn-extensions[post_hoc_ensembles]'",
    )

__all__ = [
    "AutoTabPFNClassifier",
    "AutoTabPFNRegressor",
    "GreedyWeightedEnsemble",
    "GreedyWeightedEnsembleClassifier",
    "GreedyWeightedEnsembleRegressor",
    "abstract_validation_utils",
    "greedy_weighted_ensemble",
    "pfn_phe",
]
