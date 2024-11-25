from .schema import (
    IntentPrediction,
    TrainingResponse,
    # ... add other specific classes/functions you need
)

# If you want to specify what gets imported when someone does `from schema import *`
__all__ = [
    'IntentPrediction',
    'TrainingResponse',
    # ... add other items you want to expose
]