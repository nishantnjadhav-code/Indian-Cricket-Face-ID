import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset" / "processed"

with open(ROOT / "models" / "class_names.json", encoding="utf-8") as f:
    model_classes = json.load(f)

for split in ("train", "val", "test"):
    folder = DATA_DIR / split
    counts = {
        p.name: sum(
            1 for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in
            {".jpg", ".jpeg", ".png", ".webp"}
        )
        for p in folder.iterdir()
        if p.is_dir()
    }

    print(f"\n--- {split.upper()} ---")
    print("Player folders:", len(counts))
    print("Images:", sum(counts.values()))
    print("Matches model class ordering:",
          sorted(counts) == model_classes)
    print("Players with fewer than 5 images:",
          sum(n < 5 for n in counts.values()))
    print("Players with 5–9 images:",
          sum(5 <= n < 10 for n in counts.values()))
    print("Minimum images per player:",
          min(counts.values(), default=0))
    print("Maximum images per player:",
          max(counts.values(), default=0))
