"""Explicit MLX port of MONAI SegResNet inference.

Public names load on first access, so importing the package (or the medmlx
declaration in ``segresnet_mlx.medmlx``) does not import MLX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from segresnet_mlx.inference import forward_patch as forward_patch
    from segresnet_mlx.network import SegResNet as SegResNet
    from segresnet_mlx.predictor import BUNDLE_CONFIGS as BUNDLE_CONFIGS
    from segresnet_mlx.predictor import BundlePredictor as BundlePredictor
    from segresnet_mlx.predictor import SlidingWindowConfig as SlidingWindowConfig
    from segresnet_mlx.predictor import predict_logits as predict_logits
    from segresnet_mlx.telemetry import WindowTelemetry as WindowTelemetry
    from segresnet_mlx.topology import SegResNetSpec as SegResNetSpec
    from segresnet_mlx.weights import load_model as load_model
    from segresnet_mlx.weights import required_shapes as required_shapes

__version__ = "0.3.2"

_EXPORTS = {
    "BUNDLE_CONFIGS": "segresnet_mlx.predictor",
    "BundlePredictor": "segresnet_mlx.predictor",
    "SegResNet": "segresnet_mlx.network",
    "SegResNetSpec": "segresnet_mlx.topology",
    "SlidingWindowConfig": "segresnet_mlx.predictor",
    "WindowTelemetry": "segresnet_mlx.telemetry",
    "forward_patch": "segresnet_mlx.inference",
    "load_model": "segresnet_mlx.weights",
    "predict_logits": "segresnet_mlx.predictor",
    "required_shapes": "segresnet_mlx.weights",
}

__all__ = [
    "BUNDLE_CONFIGS",
    "BundlePredictor",
    "SegResNet",
    "SegResNetSpec",
    "SlidingWindowConfig",
    "WindowTelemetry",
    "forward_patch",
    "load_model",
    "predict_logits",
    "required_shapes",
]


def __getattr__(name: str) -> object:
    """Load a public export on first access and cache it in the package."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'segresnet_mlx' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy exports in interactive package discovery."""
    return sorted({*globals(), *_EXPORTS})
