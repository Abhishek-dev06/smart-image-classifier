"""Shared configuration and helper functions for Smart Image Classifier."""

import random
from pathlib import Path

import numpy as np

# Reproducibility
SEED = 42

# Paths (relative to project root, so it works on any computer)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DIR = DATASET_DIR / "raw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "validation"
TEST_DIR = DATASET_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

# Image and training settings
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Split ratios: 70% train, 15% validation, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def get_class_names(directory: Path) -> list[str]:
    """Return sorted class names (sub-folder names) of a dataset directory."""
    if not directory.exists():
        raise FileNotFoundError(f"Folder not found: {directory}")
    names = sorted(p.name for p in directory.iterdir() if p.is_dir())
    if not names:
        raise ValueError(f"No class folders found inside: {directory}")
    return names