# feedback module
# handles user feedback for improving recommendations

from .config import EVENT_SIGNALS, DEFAULT_WEIGHTS
from .simulator import generate_synthetic_feedback
from .aggregator import aggregate_feedback
from .optimizer import WeightOptimizer
