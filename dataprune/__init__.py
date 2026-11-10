"""dataprune - energy-aware dataset pruning (hash -> SemDeDup -> confident learning -> difficulty -> CCS/D2 -> quotas)."""
from .config import PruneConfig
from .experiment import run_experiment
from .impact import ImpactTracker, compare_training
from .insights import class_warnings, health_score, recommend_keep_ratio, removal_export
from .pipeline import PruneResult, prune

__all__ = ["PruneConfig", "prune", "PruneResult", "run_experiment", "removal_export", "class_warnings",
           "health_score", "recommend_keep_ratio", "ImpactTracker", "compare_training"]
__version__ = "0.1.0"
