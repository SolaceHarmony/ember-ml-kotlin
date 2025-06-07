"""
Features module providing stateful components and dynamic stateless operations.

Stateful components (PCA, Standardize, Normalize) are accessed via factory
functions defined here. Stateless operations (one_hot) are dynamically aliased
from the active backend or common implementations.
"""

import importlib
import sys
import os
from typing import List, Optional, Callable, Any, Dict

# Import backend control functions
from ember_ml.backend import get_backend, get_backend_module

# --- Stateful Components ---

# Import the stateful classes directly from their implementation files
from ember_ml.nn.features.pca_features import PCA
from ember_ml.nn.features.standardize_features import Standardize
from ember_ml.nn.features.normalize_features import Normalize
from ember_ml.nn.features.temporal_processor import TemporalStrideProcessor
from ember_ml.nn.features.feature_engineer import GenericFeatureEngineer
from ember_ml.nn.features.generic_csv_loader import GenericCSVLoader
from ember_ml.nn.features.generic_type_detector import GenericTypeDetector
from ember_ml.nn.features.test_feature_extraction import test_feature_extraction
from ember_ml.nn.features.column_feature_extraction import ColumnFeatureExtractor, ColumnPCAFeatureExtractor, TemporalColumnFeatureExtractor
from ember_ml.nn.features.terabyte_feature_extractor import TerabyteFeatureExtractor, TerabyteTemporalStrideProcessor
# from ember_ml.nn.features.bigquery.feature_processing import BigQueryFeatureExtractor # Commented out as it's not found
from ember_ml.nn.features.animated_feature_processor import AnimatedFeatureProcessor
from ember_ml.nn.features.speedtest_event_processor import SpeedtestEventProcessor
from ember_ml.nn.features.enhanced_type_detector import EnhancedTypeDetector
# Import one_hot directly from tensor_features to avoid circular imports
from ember_ml.nn.features.tensor_features import one_hot

# Define Factory Functions locally
def pca():
    """Factory function to create a PCA instance."""
    return PCA()

def standardize():
    """Factory function to create a Standardize instance."""
    try:
        # Instantiation might fail if class definition has issues
        return Standardize()
    except NameError:
        print("Warning: Standardize class not found.")
        return None
    except Exception as e:
        print(f"Error instantiating Standardize: {e}")
        return None


def normalize():
    """Factory function to create a Normalize instance."""
    try:
        # Instantiation might fail if class definition has issues
        return Normalize()
    except NameError:
        print("Warning: Normalize class not found.")
        return None
    except Exception as e:
        print(f"Error instantiating Normalize: {e}")
        return None

# --- Dynamic Aliasing for Stateless Operations ---

# List of stateless feature operations to alias
_FEATURES_STATELESS_OPS_LIST: List[str] = [
    # 'one_hot' is now directly imported above
]

# Placeholder initialization
for _op_name in _FEATURES_STATELESS_OPS_LIST:
    if _op_name not in globals():
        globals()[_op_name] = None

# Keep track if aliases have been set for the current backend
_aliased_backend_features: Optional[str] = None

def get_features_ops_module():
    """Loads the module containing feature operations."""
    backend_name = get_backend()
    try:
        # Try loading backend-specific first
        # Assume backend feature ops live directly under backend module now
        module_name = f"ember_ml.backend.{backend_name}.features"
        module = importlib.import_module(module_name)
        # print(f"DEBUG: Loaded backend features op module: {module_name}")
        return module
    except (ImportError, ModuleNotFoundError):
        # Fallback to common implementation file
        try:
            # Common implementation is now directly in nn/features
            module_name = "ember_ml.nn.features.tensor_features"
            module = importlib.import_module(module_name)
            # print(f"DEBUG: Loaded common features op module: {module_name}")
            return module
        except (ImportError, ModuleNotFoundError):
            print(f"Error: Could not load feature ops from backend '{backend_name}' or common 'tensor_features'.")
            return None

def _update_features_aliases():
    """Dynamically updates this module's namespace with backend/common feature ops."""
    global _aliased_backend_features
    backend_name = get_backend()

    if backend_name == _aliased_backend_features:
        return

    features_ops_module = get_features_ops_module()
    if features_ops_module is None:
        print("Warning: Features ops module could not be loaded. Aliases not updated.")
        # Clear potentially stale aliases
        current_module = sys.modules[__name__]
        for op_name in _FEATURES_STATELESS_OPS_LIST:
             setattr(current_module, op_name, None)
             globals()[op_name] = None
        return

    current_module = sys.modules[__name__]
    missing_ops = []

    for op_name in _FEATURES_STATELESS_OPS_LIST:
        try:
            # Try getting from the primary (backend or fallback) module first
            op_function = getattr(features_ops_module, op_name)
            setattr(current_module, op_name, op_function)
            globals()[op_name] = op_function
        except AttributeError:
            # If not found in the primary module, try the common module explicitly
            # This handles cases where a backend module exists but lacks the function
            try:
                common_module_name = "ember_ml.nn.features.tensor_features"
                common_features_ops_module = importlib.import_module(common_module_name)
                op_function = getattr(common_features_ops_module, op_name)
                setattr(current_module, op_name, op_function)
                globals()[op_name] = op_function
                # print(f"DEBUG: Aliased '{op_name}' from common module.") # Optional debug
            except (ImportError, ModuleNotFoundError, AttributeError):
                # If not found in common module either, set to None
                setattr(current_module, op_name, None)
                globals()[op_name] = None
                missing_ops.append(op_name)

    if missing_ops:
        # print(f"Warning: Features module '{features_ops_module.__name__}' does not provide: {', '.join(missing_ops)}")
        pass # Suppress for now
    _aliased_backend_features = backend_name

# --- Initial alias setup ---
_update_features_aliases()


# --- Define __all__ ---
# Export factory functions, aliased stateless functions, and stateful classes
__all__ = [
    # Factory Functions
    'pca',
    'standardize',
    'normalize',
    # Aliased Stateless Functions
    'one_hot',
    # Stateful Classes (for type hinting/direct use)
    'PCA',
    'Standardize',
    'Normalize',
    'TemporalStrideProcessor',
    'GenericFeatureEngineer',
    'GenericCSVLoader',
    'GenericTypeDetector',
    'test_feature_extraction',
    'ColumnFeatureExtractor',
    'ColumnPCAFeatureExtractor',
    'TemporalColumnFeatureExtractor',
    'TerabyteFeatureExtractor',
    'TerabyteTemporalStrideProcessor',
    # 'BigQueryFeatureExtractor', # Commented out as it's not found
    'AnimatedFeatureProcessor',
    'SpeedtestEventProcessor',
    'EnhancedTypeDetector',
]