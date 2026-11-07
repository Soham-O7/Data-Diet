"""dataprune - energy-aware dataset pruning (hash -> SemDeDup -> confident learning -> difficulty -> CCS/D2 -> quotas)."""
from .config import PruneConfig
from .experiment import run_experiment
from .pipeline import PruneResult, prune

__all__ = ["PruneConfig", "prune", "PruneResult", "run_experiment"]
__version__ = "0.1.0"
