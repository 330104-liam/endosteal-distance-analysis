from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_endosteal_batch import distance_histogram, save_distance_histogram

observed = np.array([2.0, 4.0, 11.0, 18.0, 24.0])
eligible = np.array([1.0, 2.0, 4.0, 5.0, 11.0, 12.0, 18.0, 20.0, 24.0, 27.0])
rows = distance_histogram(observed, eligible, 10.0)
assert [r["observed_count"] for r in rows] == [2, 2, 1], rows
assert np.allclose(
    [r["expected_random_count"] for r in rows], [2.0, 1.5, 1.5]
), rows
assert sum(r["observed_count"] for r in rows) == 5
assert sum(r["expected_random_count"] for r in rows) == 5
assert all(r["random_count_95pct_low"] <= r["expected_random_count"]
           <= r["random_count_95pct_high"] for r in rows)
with TemporaryDirectory() as temp:
    image = Path(temp) / "histogram.png"
    save_distance_histogram(image, rows, "synthetic", len(observed))
    assert image.exists() and image.stat().st_size > 10000
print("PASS: histogram counts, matched random expectations, interval and PNG")

