"""
Utility functions to load pre-split Pokemon card datasets.

Use these functions in your notebooks to easily load train/val/test splits.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional


def load_split_data(split_name: str = "train") -> pd.DataFrame:
    """
    Load a specific split of the Pokemon card dataset.

    Args:
        split_name: Which split to load - "train", "validation", or "test"

    Returns:
        DataFrame with the requested split

    Raises:
        FileNotFoundError: If split files don't exist
    """
    split_dir = Path("data/splits")

    if not split_dir.exists():
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}\n"
            "Run: uv run python data/create_splits.py"
        )

    file_map = {
        "train": "pokemon_train.csv",
        "validation": "pokemon_validation.csv",
        "val": "pokemon_validation.csv",  # Alias
        "test": "pokemon_test.csv"
    }

    if split_name not in file_map:
        raise ValueError(
            f"Invalid split_name: {split_name}\n"
            f"Valid options: {list(file_map.keys())}"
        )

    file_path = split_dir / file_map[split_name]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}\n"
            "Run: uv run python data/create_splits.py"
        )

    return pd.read_csv(file_path)


def load_all_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all three splits at once.

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    train_df = load_split_data("train")
    val_df = load_split_data("validation")
    test_df = load_split_data("test")

    return train_df, val_df, test_df


def load_xy_split(
    split_name: str = "train",
    target_col: str = "type",
    feature_cols: Optional[list] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load a split with features (X) and target (y) separated.

    Args:
        split_name: Which split to load - "train", "validation", or "test"
        target_col: Name of the target column (default: "type")
        feature_cols: List of feature columns to use (if None, uses all except target)

    Returns:
        Tuple of (X, y) where X is features and y is target
    """
    df = load_split_data(split_name)

    if feature_cols is None:
        feature_cols = [col for col in df.columns if col != target_col]

    X = df[feature_cols]
    y = df[target_col]

    return X, y


def get_split_info() -> dict:
    """
    Get information about the dataset splits.

    Returns:
        Dictionary with split metadata
    """
    import json

    metadata_path = Path("data/splits/split_metadata.json")

    if not metadata_path.exists():
        return {
            "error": "Split metadata not found. Run: uv run python data/create_splits.py"
        }

    with open(metadata_path, 'r') as f:
        return json.load(f)


# Example usage functions for notebooks
def example_basic_usage():
    """Example: Load training data."""
    train_df = load_split_data("train")
    print(f"Loaded {len(train_df)} training cards")
    return train_df


def example_all_splits():
    """Example: Load all three splits."""
    train_df, val_df, test_df = load_all_splits()
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    return train_df, val_df, test_df


def example_xy_format():
    """Example: Load with X/y separation."""
    X_train, y_train = load_xy_split("train", target_col="type")
    X_val, y_val = load_xy_split("validation", target_col="type")

    print(f"Training: X={X_train.shape}, y={len(y_train)}")
    print(f"Validation: X={X_val.shape}, y={len(y_val)}")

    return X_train, y_train, X_val, y_val


def example_with_specific_features():
    """Example: Load with specific feature columns."""
    feature_cols = [
        'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed',
        'generation', 'is_legendary'
    ]

    X_train, y_train = load_xy_split(
        "train",
        target_col="type",
        feature_cols=feature_cols
    )

    print(f"Features: {list(X_train.columns)}")
    print(f"Shape: {X_train.shape}")

    return X_train, y_train


if __name__ == "__main__":
    print("="*70)
    print("POKEMON CARD SPLIT LOADER - EXAMPLES")
    print("="*70)

    # Example 1: Load single split
    print("\n1. Load training data:")
    try:
        train_df = example_basic_usage()
        print(f"   Columns: {list(train_df.columns)}")
    except FileNotFoundError as e:
        print(f"   Error: {e}")

    # Example 2: Load all splits
    print("\n2. Load all splits:")
    try:
        train_df, val_df, test_df = example_all_splits()
    except FileNotFoundError as e:
        print(f"   Error: {e}")

    # Example 3: Load with X/y format
    print("\n3. Load with X/y separation:")
    try:
        X_train, y_train, X_val, y_val = example_xy_format()
    except FileNotFoundError as e:
        print(f"   Error: {e}")

    # Example 4: Get split info
    print("\n4. Get split metadata:")
    info = get_split_info()
    if "error" not in info:
        print(f"   Total cards: {info['dataset_info']['total_cards']}")
        print(f"   Split ratio: {info['dataset_info']['split_ratio']}")
        print(f"   Num classes: {info['num_classes']}")
    else:
        print(f"   Error: {info['error']}")

    print("\n" + "="*70)
