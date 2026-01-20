from pathlib import Path
from src.preprocessing import create_dataset_splits #src/ is just a folder, not importable 

def test_dataset_split_structure(tmp_path):
    """
    Verify train/val/test folders are created correctly
    """

    raw = tmp_path / "raw"
    processed = tmp_path / "processed"

    # Create fake dataset
    for cls in ["mask", "no_mask"]:
        cls_dir = raw / cls
        cls_dir.mkdir(parents=True)
        for i in range(10):
            (cls_dir / f"{i}.jpg").touch()

    create_dataset_splits(
        raw_data_dir=str(raw),
        output_dir=str(processed)
    )

    for split in ["train", "val", "test"]:
        for cls in ["mask", "no_mask"]:
            assert (processed / split / cls).exists()
