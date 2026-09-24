"""Synthetic checks for automatic boundary, overlap detection and null test."""
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_endosteal_batch import (
    automatic_shaft_roi, detect_overlap_objects, masks_at_analysis_scale,
    monte_carlo, orient_vertical,
)


def main():
    rng = np.random.default_rng(4)
    h, w = 1800, 1000
    dapi = rng.normal(15, 2, (h, w)).astype(np.float32)
    left = (220 + 8 * np.sin(np.arange(h) / 120)).astype(int)
    right = (780 + 8 * np.sin(np.arange(h) / 140)).astype(int)
    for y in range(h):
        dapi[y, left[y]:right[y]] += 95
    dapi = ndi.gaussian_filter(dapi, 2)
    auto = automatic_shaft_roi(dapi, 1.0, 1000, 600)
    assert auto["metrics"]["qc_status"] == "PASS", auto["metrics"]
    cyx = np.stack([dapi, np.zeros_like(dapi), np.zeros_like(dapi)])
    oriented, transposed = orient_vertical(cyx)
    assert not transposed and oriented.shape == cyx.shape
    marrow, boundary, _ = masks_at_analysis_scale(auto, (h, w), 1)

    red = np.zeros((h, w), dtype=np.float32)
    cd31 = np.zeros_like(red)
    start, stop = np.where(marrow)[0].min(), np.where(marrow)[0].max()
    for y in np.linspace(start + 40, stop - 40, 20).astype(int):
        x = int(left[y] + 20)
        red[y-4:y+5, x-4:x+5] = 200
        cd31[y-5:y+6, x-5:x+6] = 180
    detected = detect_overlap_objects(
        red, cd31, marrow, 1.0, 50, 50, 2.0, 20.0)
    assert len(detected["centres_yx"]) == 20
    distance = ndi.distance_transform_edt(~boundary)
    rr = np.rint(detected["centres_yx"][:, 0]).astype(int)
    cc = np.rint(detected["centres_yx"][:, 1]).astype(int)
    observed = distance[rr, cc]
    null, p = monte_carlo(
        observed, distance[marrow & ~boundary], 2000,
        np.random.default_rng(8))
    assert np.median(observed) < np.median(null)
    assert p < 0.01

    horizontal, transposed = orient_vertical(cyx.transpose(0, 2, 1))
    assert transposed and horizontal.shape == cyx.shape
    print("PASS: automatic ROI, endosteal traces, overlap objects, null test, orientation")


if __name__ == "__main__":
    main()