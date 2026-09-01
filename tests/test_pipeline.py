from pathlib import Path

from src.pipeline.build_features import build_split, extract_features


def test_extract_features_shape():
    sample = next(Path("data/raw/train").glob("*/*.jpeg"))
    features = extract_features(sample)
    assert features.shape == (1764,)


def test_build_split_matches_folder_structure():
    X, y, paths = build_split(Path("data/raw/test"))
    assert X.shape[0] == len(y) == len(paths)
    assert set(y) == {
        "2S1", "BMP2", "BRDM_2", "BTR60", "BTR70",
        "D7", "T62", "T72", "ZIL131", "ZSU_23_4",
    }
