"""
Configuration parameters for Module 6 (Predictive Modelling).
 
Currently holds:
 
    FeatureConfig    : feature-extraction configuration
    TrainingConfig   : ridge-regression training configuration
    ComparatorConfig : random-forest comparator configuration
    PredictionConfig : Monte Carlo prediction-interval configuration
 
All are re-exported here so all Module 6 configuration lives in one
place. Additional config dataclasses will be added here as later
components are built.
"""
 
from .features import FeatureConfig
from .training import TrainingConfig
from .comparator import ComparatorConfig
from .prediction import PredictionConfig
 
__all__ = ["FeatureConfig", "TrainingConfig", "ComparatorConfig",
           "PredictionConfig"]