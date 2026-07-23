"""Single-frame HOT3D MANO-to-real-hand restoration MVP.

The package deliberately wraps the repository's established HOT3D C1 camera
conversion and MANO rendering code instead of maintaining a second geometry
pipeline.
"""

from .conditions import ConditionBuilder, ConditionConfig
from .hot3d_dataset import Hot3DSingleFrameDataset

__all__ = ["ConditionBuilder", "ConditionConfig", "Hot3DSingleFrameDataset"]
