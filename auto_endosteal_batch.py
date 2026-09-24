#!/usr/bin/env python3
"""Automatic, reproducible shaft endosteal-distance analysis.

Designed for whole-bone fluorescence TIFFs with DAPI, red lineage signal, and
CD31. The shaft ROI and the two marrow-facing cortical boundaries are selected
from DAPI only. Red/CD31 overlap is measured only after ROI selection.
"""

from __future__ import annotations

import argparse
import csv

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from scipy import ndimage as ndi
from scipy.stats import hypergeom, t as student_t, ttest_1samp, ttest_ind
from skimage.filters import threshold_otsu
from skimage.measure import regionprops


VERSION = "1.4.0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path, help="A TIFF or a folder containing TIFFs")
    p.add_argument("--output", type=Path,
                   help="Default: INPUT/auto_endosteal_results")
    p.add_argument("--recursive", action="store_true",
                   help="Search subfolders recursively")
    p.add_argument("--dapi-channel", type=int, default=0)
    p.add_argument("--red-channel", type=int, default=1)
    p.add_argument("--cd31-channel", type=int, default=2)
    p.add_argument("--um-per-pixel", type=float,
                   help="Override TIFF XY calibration")
    p.add_argument("--roi-length-um", type=float, default=1500.0)
    p.add_argument("--analysis-pixel-um", type=float, default=1.8,
                   help="Target analysis pixel size after integer downsampling")
    p.add_argument("--boundary-target-width-px", type=int, default=1100)
    p.add_argument("--red-threshold", type=float,
                   help="Fixed raw red threshold; default: Otsu within each ROI")
    p.add_argument("--cd31-threshold", type=float,
                   help="Fixed raw CD31 threshold; default: Otsu within each ROI")
    p.add_argument("--cd31-expand-um", type=float, default=3.6)
    p.add_argument("--min-object-um2", type=float, default=66.0)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--distance-bin-um", type=float, default=25.0,
                   help="Distance histogram bin width in micrometers")
    p.add_argument("--seed", type=int, default=20260923)
    p.add_argument("--manifest", type=Path,
                   help="Optional CSV with filename, animal_id and optional group columns")
    p.add_argument("--allow-qc-fail", action="store_true",
                   help="Analyze even when automatic boundary QC fails")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    positive = ("roi_length_um", "analysis_pixel_um", "boundary_target_width_px",
                "cd31_expand_um", "min_object_um2", "iterations",
                "distance_bin_um")
    for name in positive:
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    channels = [args.dapi_channel, args.red_channel, args.cd31_channel]
    if min(channels) < 0 or len(set(channels)) != 3:
        p.error("DAPI, red, and CD31 channel indices must be distinct and nonnegative")
    if args.um_per_pixel is not None and args.um_per_pixel <= 0:
        p.error("--um-per-pixel must be positive")
    return args


def discover_tiffs(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    globber = path.rglob if recursive else path.glob
    files = sorted(
        f for f in globber("*")
        if f.is_file()
        and f.suffix.lower() in {".tif", ".tiff"}
        and "(RGB)" not in f.name.upper()
        and "auto_endosteal_results" not in str(f)
    )
    if not files:
        raise ValueError(f"No non-RGB TIFF files found in {path}")
    return files


def rational_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(value[0]) / float(value[1])


def infer_um_per_pixel(tif: tifffile.TiffFile) -> tuple[float, dict]:
    page = tif.pages[0]
    meta = tif.imagej_metadata or {}
    unit = str(meta.get("unit", "")).strip().lower().replace("µ", "u")
    resolution_unit = int(page.tags["ResolutionUnit"].value) if "ResolutionUnit" in page.tags else 1
    xres = rational_float(page.tags["XResolution"].value) if "XResolution" in page.tags else math.nan
    yres = rational_float(page.tags["YResolution"].value) if "YResolution" in page.tags else math.nan

    def convert(resolution: float) -> float:
        if not np.isfinite(resolution) or resolution <= 0:
            return math.nan
        if unit in {"micron", "microns", "um", "micrometer", "micrometers"}:
            return 1.0 / resolution
        if resolution_unit == 3:
            return 10000.0 / resolution
        if resolution_unit == 2:
            return 25400.0 / resolution
        return math.nan

    x_um, y_um = convert(xres), convert(yres)
    if not np.isfinite(x_um) or not np.isfinite(y_um):
        raise ValueError("No usable XY calibration; supply --um-per-pixel")
    anisotropy = abs(x_um - y_um) / np.mean([x_um, y_um])
    if anisotropy > 0.02:
        raise ValueError(f"XY pixel calibration differs by {anisotropy:.1%}")
    value = float(np.mean([x_um, y_um]))
    return value, {
        "imagej_unit": unit,
        "x_resolution": xres,
        "y_resolution": yres,
        "resolution_unit": resolution_unit,
        "x_um_per_pixel": x_um,
        "y_um_per_pixel": y_um,
    }


def load_tiff(path: Path, channel_indices: list[int],
              calibration_override: float | None) -> tuple[np.ndarray, float, dict]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        axes = series.axes.upper()
        original_axes = series.axes
        original_shape = list(series.shape)
        array = series.asarray()
        inferred, calibration_meta = (calibration_override, {"override": True})
        if calibration_override is None:
            inferred, calibration_meta = infer_um_per_pixel(tif)
        imagej_meta = tif.imagej_metadata or {}
        lut_ends = None
        if "LUTs" in imagej_meta:
            try:
                lut_ends = [list(map(int, lut[:, -1])) for lut in imagej_meta["LUTs"]]
            except Exception:
                lut_ends = None

    if len(axes) != array.ndim:
        raise ValueError(f"Axes {axes} do not match shape {array.shape}")
    if "T" in axes:
        axis = axes.index("T")
        array = np.take(array, 0, axis=axis)
        axes = axes.replace("T", "")
    if "Z" in axes:
        axis = axes.index("Z")
        array = np.max(array, axis=axis)
        axes = axes.replace("Z", "")
    if "C" not in axes and "S" not in axes and "Q" in axes:
        q = axes.index("Q")
        if 3 <= array.shape[q] <= 6:
            axes = axes.replace("Q", "C")
    for axis_name in list(axes):
        if axis_name not in "CYXS":
            axis = axes.index(axis_name)
            if array.shape[axis] != 1:
                raise ValueError(f"Unsupported axis {axis_name} with size {array.shape[axis]}")
            array = np.take(array, 0, axis=axis)
            axes = axes.replace(axis_name, "")
    channel_axis = "C" if "C" in axes else "S" if "S" in axes else None
    if channel_axis is None:
        raise ValueError(f"No channel axis found (axes={axes})")
    array = np.moveaxis(array, axes.index(channel_axis), 0)
    spatial_axes = axes.replace(channel_axis, "")
    if spatial_axes == "XY":
        array = array.swapaxes(1, 2)
    elif spatial_axes != "YX":
        raise ValueError(f"Unsupported spatial axes {spatial_axes}")
    if max(channel_indices) >= array.shape[0]:
        raise ValueError(f"Only {array.shape[0]} channels; requested {channel_indices}")
    selected = np.asarray(array[channel_indices])
    return selected, float(inferred), {
        "source_axes": original_axes,
        "source_shape": original_shape,
        "selected_shape_cyx": list(selected.shape),
        "calibration": calibration_meta,
        "lut_end_rgb": lut_ends,
    }


def orient_vertical(cyx: np.ndarray) -> tuple[np.ndarray, bool]:
    height, width = cyx.shape[1:]
    if height >= 1.25 * width:
        return cyx, False
    if width >= 1.25 * height:
        return cyx.transpose(0, 2, 1), True
    raise ValueError(
        f"Image aspect ratio {height}x{width} is not a whole-bone long-axis view"
    )


def downsample(cyx: np.ndarray, factor: int) -> np.ndarray:
    return cyx[:, ::factor, ::factor]


def scale01(image: np.ndarray) -> np.ndarray:
    values = image[np.isfinite(image)]
    if values.size == 0:
        return np.zeros_like(image, dtype=float)
    lo, hi = np.percentile(values, (1, 99.7))
    return np.clip((np.nan_to_num(image, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)


def rgb_composite(cyx: np.ndarray) -> np.ndarray:
    dapi, red, cd31 = cyx
    rgb = np.zeros((*dapi.shape, 3), dtype=float)
    rgb[..., 0] = scale01(red)
    rgb[..., 1] = scale01(cd31)
    rgb[..., 2] = scale01(dapi)
    return rgb


def trace_candidate(smooth: np.ndarray, gradient: np.ndarray, pixel_um: float,
                    center: int, roi_length_um: float) -> dict:
    height, width = smooth.shape
    left_start = max(0, int(center - 0.30 * width))
    left_end = max(left_start + 3, int(center - 0.06 * width))
    right_start = min(width - 3, int(center + 0.06 * width))
    right_end = min(width, int(center + 0.30 * width))
    if left_end <= left_start or right_end <= right_start:
        raise ValueError("Boundary search bands collapsed")

    left = np.argmax(gradient[:, left_start:left_end], axis=1) + left_start
    right = np.argmin(gradient[:, right_start:right_end], axis=1) + right_start
    median_size = max(3, int(round(110.0 / pixel_um)))
    if median_size % 2 == 0:
        median_size += 1
    left = ndi.median_filter(left, size=median_size)
    right = ndi.median_filter(right, size=median_size)
    sigma_y = max(1.0, 40.0 / pixel_um)
    left = ndi.gaussian_filter1d(left.astype(float), sigma_y)
    right = ndi.gaussian_filter1d(right.astype(float), sigma_y)

    yi = np.arange(height)
    li = np.clip(np.rint(left).astype(int), 0, width - 1)
    ri = np.clip(np.rint(right).astype(int), 0, width - 1)
    strength = gradient[yi, li] - gradient[yi, ri]
    contrast = np.full(height, -np.inf, dtype=float)
    pad = max(2, round(30.0 / pixel_um))
    band = max(3, round(80.0 / pixel_um))
    for y, (lft, rgt) in enumerate(zip(li, ri)):
        if rgt - lft <= 2 * pad:
            continue
        inside = smooth[y, lft + pad:rgt - pad]
        outside = np.r_[
            smooth[y, max(0, lft - band):max(0, lft - pad)],
            smooth[y, min(width, rgt + pad):min(width, rgt + band)],
        ]
        if inside.size and outside.size:
            contrast[y] = np.median(inside) - np.median(outside)

    widths_um = (right - left) * pixel_um
    window = max(20, int(round(roi_length_um / pixel_um)))
    if window >= 0.60 * height:
        raise ValueError("Requested ROI is too long for this image")
    step = max(1, window // 25)
    dynamic = max(
        float(np.percentile(smooth, 95) - np.percentile(smooth, 5)), 1e-9
    )
    candidates = []
    for start in range(int(0.18 * height), int(0.92 * height) - window, step):
        stop = start + window
        sl = slice(start, stop)
        if not np.all(np.isfinite(contrast[sl])):
            continue
        median_width = float(np.median(widths_um[sl]))
        width_cv = float(np.std(widths_um[sl]) / max(np.mean(widths_um[sl]), 1))
        roughness = float(
            np.median(np.abs(np.diff(left[sl])))
            + np.median(np.abs(np.diff(right[sl])))
        )
        median_contrast = float(np.median(contrast[sl]))
        median_strength = float(np.median(strength[sl]))
        score = (
            median_contrast
            + 2.0 * median_strength
            - 30.0 * width_cv
            - 5.0 * roughness
        )
        if median_width < 500 or median_width > 3500:
            score = -np.inf
        candidates.append({
            "score": score, "start": start, "stop": stop,
            "median_width_um": median_width, "width_cv": width_cv,
            "roughness_ds_px": roughness,
            "median_contrast": median_contrast,
            "median_contrast_fraction": median_contrast / dynamic,
            "median_boundary_strength": median_strength,
        })
    finite = [item for item in candidates if np.isfinite(item["score"])]
    if not finite:
        raise ValueError("No valid shaft window found")
    best = max(finite, key=lambda item: item["score"])
    best.update({
        "center_x_ds": center,
        "left_x_ds": left,
        "right_x_ds": right,
        "search_bands_ds": [left_start, left_end, right_start, right_end],
        "dynamic_range": dynamic,
    })
    return best


def automatic_shaft_roi(dapi: np.ndarray, um_per_pixel: float,
                        target_width: int, roi_length_um: float) -> dict:
    detect_ds = max(1, int(round(dapi.shape[1] / target_width)))
    dapi_ds = dapi[::detect_ds, ::detect_ds].astype(np.float32)
    detect_pixel_um = um_per_pixel * detect_ds
    smooth = ndi.gaussian_filter(
        dapi_ds,
        sigma=(max(1, 40.0 / detect_pixel_um),
               max(1, 22.0 / detect_pixel_um)),
    )
    gradient = np.gradient(smooth, axis=1)
    width = smooth.shape[1]
    centers = np.unique(np.rint(
        np.linspace(0.44 * width, 0.56 * width, 7)
    ).astype(int))
    traces = []
    for center in centers:
        try:
            traces.append(trace_candidate(
                smooth, gradient, detect_pixel_um, int(center), roi_length_um
            ))
        except ValueError:
            pass
    if not traces:
        raise ValueError("No valid centre/boundary solution")
    best = max(traces, key=lambda item: item["score"])
    metrics = {
        key: best[key] for key in (
            "score", "median_width_um", "width_cv", "roughness_ds_px",
            "median_contrast", "median_contrast_fraction",
            "median_boundary_strength", "center_x_ds", "search_bands_ds",
            "dynamic_range", "start", "stop",
        )
    }
    issues = []
    if best["median_contrast_fraction"] < 0.15:
        issues.append("low marrow-to-cortex DAPI contrast")
    if best["width_cv"] > 0.20:
        issues.append("unstable marrow width")
    if best["median_width_um"] < 500 or best["median_width_um"] > 3500:
        issues.append("implausible marrow width")
    left_roi = best["left_x_ds"][best["start"]:best["stop"]]
    right_roi = best["right_x_ds"][best["start"]:best["stop"]]
    if (np.min(left_roi) < 0.03 * width
            or np.max(right_roi) > 0.97 * width):
        issues.append("boundary touches image side")
    metrics["qc_status"] = "FAIL" if issues else "PASS"
    metrics["qc_issues"] = issues
    return {
        "detect_downsample": detect_ds,
        "detect_pixel_um": detect_pixel_um,
        "left_x_ds": best["left_x_ds"],
        "right_x_ds": best["right_x_ds"],
        "roi_start_y_ds": best["start"],
        "roi_stop_y_ds": best["stop"],
        "metrics": metrics,
    }


def masks_at_analysis_scale(auto: dict, shape: tuple[int, int],
                            analysis_ds: int) -> tuple[np.ndarray, np.ndarray, dict]:
    height, width = shape
    detect_ds = auto["detect_downsample"]
    detect_y = np.arange(len(auto["left_x_ds"])) * detect_ds / analysis_ds
    analysis_rows = np.arange(height)
    left = np.interp(
        analysis_rows, detect_y,
        auto["left_x_ds"] * detect_ds / analysis_ds,
    )
    right = np.interp(
        analysis_rows, detect_y,
        auto["right_x_ds"] * detect_ds / analysis_ds,
    )
    start = max(0, int(round(auto["roi_start_y_ds"] * detect_ds / analysis_ds)))
    stop = min(height, int(round(auto["roi_stop_y_ds"] * detect_ds / analysis_ds)))
    marrow = np.zeros(shape, dtype=bool)
    boundary = np.zeros(shape, dtype=bool)
    for y in range(start, stop):
        lft = int(np.clip(round(left[y]), 0, width - 1))
        rgt = int(np.clip(round(right[y]), 0, width - 1))
        if rgt <= lft:
            continue
        marrow[y, lft:rgt + 1] = True
        boundary[y, lft] = True
        boundary[y, rgt] = True
    if not marrow.any() or not boundary.any():
        raise ValueError("Automatic ROI became empty at analysis resolution")
    return marrow, boundary, {
        "roi_start_y": start, "roi_stop_y": stop,
        "left_x": left[start:stop].tolist(),
        "right_x": right[start:stop].tolist(),
    }


def otsu_or_fixed(image: np.ndarray, mask: np.ndarray,
                  fixed: float | None) -> float:
    if fixed is not None:
        return float(fixed)
    values = image[mask & np.isfinite(image)]
    if values.size < 100 or np.max(values) <= np.min(values):
        raise ValueError("Insufficient signal variation for Otsu threshold")
    return float(threshold_otsu(values))


def detect_overlap_objects(red: np.ndarray, cd31: np.ndarray, marrow: np.ndarray,
                           pixel_um: float, red_fixed: float | None,
                           cd31_fixed: float | None, expand_um: float,
                           min_area_um2: float) -> dict:
    red_threshold = otsu_or_fixed(red, marrow, red_fixed)
    cd31_threshold = otsu_or_fixed(cd31, marrow, cd31_fixed)
    cd31_binary = (cd31 > cd31_threshold) & marrow
    expand_px = max(0, int(round(expand_um / pixel_um)))
    if expand_px:
        cd31_near = ndi.binary_dilation(cd31_binary, iterations=expand_px)
    else:
        cd31_near = cd31_binary
    overlap = (red > red_threshold) & cd31_near & marrow
    labels, _ = ndi.label(overlap, structure=np.ones((3, 3), dtype=int))
    min_area_px = max(1, int(math.ceil(min_area_um2 / pixel_um**2)))
    objects = [obj for obj in regionprops(labels) if obj.area >= min_area_px]
    centres = np.asarray(
        [[obj.centroid[0], obj.centroid[1]] for obj in objects], dtype=float
    ).reshape((-1, 2))
    areas_um2 = np.asarray([obj.area * pixel_um**2 for obj in objects], dtype=float)
    return {
        "red_threshold": red_threshold,
        "cd31_threshold": cd31_threshold,
        "cd31_binary": cd31_binary,
        "overlap": overlap,
        "centres_yx": centres,
        "areas_um2": areas_um2,
        "min_area_px": min_area_px,
        "expand_px": expand_px,
    }


def monte_carlo(observed: np.ndarray, eligible: np.ndarray, iterations: int,
                rng: np.random.Generator) -> tuple[np.ndarray, float]:
    if observed.size == 0:
        raise ValueError("No red/CD31 objects detected")
    if eligible.size < observed.size:
        raise ValueError("Random domain smaller than observed object count")
    medians = np.empty(iterations, dtype=float)
    for i in range(iterations):
        medians[i] = np.median(
            rng.choice(eligible, size=observed.size, replace=False)
        )
    obs = float(np.median(observed))
    p = (1 + int(np.count_nonzero(medians <= obs))) / (iterations + 1)
    return medians, p


def stable_rng(seed: int, path: Path) -> np.random.Generator:
    # Start every image from the recorded seed so moving/renaming the source
    # does not change the Monte Carlo result.
    return np.random.default_rng(seed)


def original_xy(row: float, col: float, analysis_ds: int,
                transposed: bool) -> tuple[float, float]:
    if transposed:
        return row * analysis_ds, col * analysis_ds
    return col * analysis_ds, row * analysis_ds


def save_overlay(path: Path, cyx: np.ndarray, marrow: np.ndarray,
                 boundary: np.ndarray, centres: np.ndarray, title: str) -> None:
    ys, xs = np.where(boundary)
    my, mx = np.where(marrow)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(rgb_composite(cyx))
    ax.scatter(xs, ys, s=0.2, c="#33ff66", linewidths=0,
               label="automatic endosteum")
    if centres.size:
        ax.scatter(centres[:, 1], centres[:, 0], s=9, facecolors="none",
                   edgecolors="white", linewidths=0.6,
                   label="red/CD31 overlap object")
    if my.size:
        ax.plot([mx.min(), mx.max(), mx.max(), mx.min(), mx.min()],
                [my.min(), my.min(), my.max(), my.max(), my.min()],
                color="#00ffff", lw=0.8, label="selected shaft extent")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_null_plot(path: Path, null_medians: np.ndarray,
                   observed_median: float, image_name: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(null_medians, bins=45, color="#7890a8", edgecolor="white")
    ax.axvline(observed_median, color="#d33b3b", lw=2,
               label="Observed red/CD31 median")
    ax.set_title(image_name)
    ax.set_xlabel("Median distance to automatic endosteum (um)")
    ax.set_ylabel("Monte Carlo draws")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_distance_boxplot(path: Path, observed: np.ndarray,
                          random_distances: np.ndarray, p_value: float,
                          image_name: str, iterations: int) -> None:
    """Show every distance; p is the independent spatial Monte Carlo result."""
    if observed.size != random_distances.size or observed.size == 0:
        raise ValueError("Boxplot groups must have the same nonzero size")
    fig, ax = plt.subplots(figsize=(5.8, 6.0))
    values = [np.asarray(observed, dtype=float),
              np.asarray(random_distances, dtype=float)]
    box = ax.boxplot(values, positions=[0, 1], widths=0.38,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "#20242b", "linewidth": 1.6},
                     whiskerprops={"color": "#5a606a"},
                     capprops={"color": "#5a606a"})
    colors = ["#b64c4b", "#557a9d"]
    for patch, color in zip(box["boxes"], colors):
        patch.set(facecolor=color, alpha=0.18, edgecolor=color, linewidth=1.4)
    jitter_rng = np.random.default_rng(5107)
    for x, sample, color in zip((0, 1), values, colors):
        jitter = jitter_rng.uniform(-0.15, 0.15, size=sample.size)
        ax.scatter(x + jitter, sample, s=13, alpha=0.72, c=color,
                   linewidths=0, zorder=3)
    ymax = float(max(np.max(values[0]), np.max(values[1])))
    margin = max(15.0, 0.08 * ymax)
    bracket_y = ymax + margin
    ax.plot([0, 0, 1, 1],
            [bracket_y - 0.18 * margin, bracket_y, bracket_y,
             bracket_y - 0.18 * margin], color="#333333", lw=1.1)
    ax.text(0.5, bracket_y + 0.16 * margin,
            f"Spatial Monte Carlo p = {p_value:.3g}",
            ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, bracket_y + 1.8 * margin)
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0, 1], ["Actual", "Random"])
    ax.set_ylabel("Distance to endosteum (um)")
    ax.set_title(Path(image_name).stem, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.018,
             f"n={observed.size} per column; one random sample shown. "
             f"p uses {iterations:,} random medians.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def save_animal_ratio_boxplot(path: Path, animal_rows: list[dict],
                              test_rows: list[dict]) -> None:
    """One dot per animal; t test is on each animal's log distance ratio."""
    groups = sorted({row["group"] for row in animal_rows})
    if groups == [""]:
        groups = ["all"]
    samples = []
    for group in groups:
        key = "" if group == "all" else group
        samples.append(np.asarray(
            [row["exp_median_log_ratio"] for row in animal_rows
             if row["group"] == key], dtype=float
        ))
    fig, ax = plt.subplots(figsize=(max(5.5, 1.5 * len(groups) + 2.5), 5.6))
    positions = np.arange(len(groups), dtype=float)
    box = ax.boxplot(samples, positions=positions, widths=0.38,
                     patch_artist=True, showfliers=False,
                     medianprops={"color": "#20242b", "linewidth": 1.6})
    for patch in box["boxes"]:
        patch.set(facecolor="#b64c4b", alpha=0.18,
                  edgecolor="#b64c4b", linewidth=1.4)
    rng = np.random.default_rng(5107)
    for x, sample in zip(positions, samples):
        jitter = rng.uniform(-0.12, 0.12, sample.size)
        ax.scatter(x + jitter, sample, s=34, color="#b64c4b",
                   alpha=0.85, linewidths=0, zorder=3)
    ax.axhline(1, color="#677784", lw=1, ls="--")
    maximum = float(max(1.0, *(float(np.max(v)) for v in samples)))
    margin = max(0.12, 0.12 * maximum)
    for x, group, sample in zip(positions, groups, samples):
        record = next(
            (r for r in test_rows
             if r["test"] == "one_sample_student_t_vs_random"
             and r["group_a"] == group), None
        )
        label = (f"t p={record['p_two_sided']:.3g}"
                 if record and record["status"] == "computed"
                 else "t p=NA")
        ax.text(x, float(np.max(sample)) + 0.28 * margin, label,
                ha="center", fontsize=9)
    between = next(
        (r for r in test_rows
         if r["test"] == "two_sample_student_t_equal_variance"), None
    )
    if between:
        label = (f"between groups p={between['p_two_sided']:.3g}"
                 if between["status"] == "computed" else "between groups p=NA")
        ax.text(float(np.mean(positions)), maximum + 1.25 * margin,
                label, ha="center", fontsize=9)
    ax.set_ylim(0, maximum + (2.1 if between else 1.3) * margin)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_xticks(positions, groups)
    ax.set_ylabel("Animal observed / uniform random distance")
    ax.set_title("Animal-level distance ratio")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.5, 0.02,
             "Each point is one animal. Student t tests use animal log ratios.",
             ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, dpi=600)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def distance_histogram(observed: np.ndarray, eligible: np.ndarray,
                       bin_um: float) -> list[dict]:
    """Observed structures versus uniform random marrow locations, matched in n."""
    if observed.size == 0 or eligible.size == 0 or bin_um <= 0:
        raise ValueError("Histogram needs distances and a positive bin width")
    maximum = float(max(np.max(observed), np.max(eligible)))
    n_bins = max(1, int(math.ceil(maximum / bin_um)))
    edges = np.arange(n_bins + 1, dtype=float) * bin_um
    observed_counts, _ = np.histogram(observed, bins=edges)
    domain_counts, _ = np.histogram(eligible, bins=edges)
    population = int(eligible.size)
    sample_size = int(observed.size)
    rows = []
    for i, (actual, available) in enumerate(zip(observed_counts, domain_counts)):
        expected = sample_size * int(available) / population
        low = int(hypergeom.ppf(0.025, population, int(available), sample_size))
        high = int(hypergeom.ppf(0.975, population, int(available), sample_size))
        rows.append({
            "bin_start_um": float(edges[i]),
            "bin_end_um": float(edges[i + 1]),
            "observed_count": int(actual),
            "expected_random_count": float(expected),
            "random_count_95pct_low": low,
            "random_count_95pct_high": high,
            "observed_percent": float(100 * actual / sample_size),
            "expected_random_percent": float(100 * available / population),
            "observed_over_expected": (
                float(actual / expected) if expected > 0 else ""
            ),
        })
    return rows


def save_distance_histogram(path: Path, rows: list[dict],
                            image_name: str, n_objects: int) -> None:
    starts = np.asarray([row["bin_start_um"] for row in rows], dtype=float)
    ends = np.asarray([row["bin_end_um"] for row in rows], dtype=float)
    centers = (starts + ends) / 2
    actual = np.asarray([row["observed_count"] for row in rows], dtype=float)
    expected = np.asarray([row["expected_random_count"] for row in rows], dtype=float)
    low = np.asarray([row["random_count_95pct_low"] for row in rows], dtype=float)
    high = np.asarray([row["random_count_95pct_high"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, actual, width=0.88 * (ends - starts),
           color="#d85d58", alpha=0.85, label="Observed red/CD31 structures")
    ax.plot(centers, expected, color="#263b53", marker="o", markersize=3,
            lw=1.7, label="Uniform random expected count")
    ax.fill_between(centers, low, high, color="#263b53", alpha=0.16,
                    label="Random 95% interval per bin")
    ax.set_xlim(0, float(ends[-1]))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Distance to automatic endosteum (um)")
    ax.set_ylabel("Number of structures per distance bin")
    ax.set_title(f"{image_name} | n={n_objects} red/CD31 overlap structures",
                 fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"filename", "animal_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Manifest needs filename and animal_id columns")
    mapping = {}
    for row in rows:
        filename = (row["filename"] or "").strip()
        animal_id = (row["animal_id"] or "").strip()
        group = (row.get("group") or "").strip()
        if not filename or not animal_id:
            raise ValueError("Each manifest row needs filename and animal_id")
        if filename in mapping:
            raise ValueError(f"Duplicate manifest filename: {filename}")
        mapping[filename] = {"animal_id": animal_id, "group": group}
    return mapping


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def analyze_one(path: Path, prefix: Path, args: argparse.Namespace,
                animal_id: str | None) -> tuple[dict, list[dict], np.ndarray]:
    channels, um_per_pixel, source_meta = load_tiff(
        path,
        [args.dapi_channel, args.red_channel, args.cd31_channel],
        args.um_per_pixel,
    )
    oriented, transposed = orient_vertical(channels)
    dapi_full = oriented[0]
    auto = automatic_shaft_roi(
        dapi_full, um_per_pixel, args.boundary_target_width_px,
        args.roi_length_um,
    )
    if auto["metrics"]["qc_status"] == "FAIL" and not args.allow_qc_fail:
        raise ValueError(
            "Automatic boundary QC failed: "
            + "; ".join(auto["metrics"]["qc_issues"])
        )

    analysis_ds = max(1, int(round(args.analysis_pixel_um / um_per_pixel)))
    analysis = downsample(oriented, analysis_ds).astype(np.float32)
    pixel_um = um_per_pixel * analysis_ds
    marrow, boundary, boundary_coords = masks_at_analysis_scale(
        auto, analysis.shape[1:], analysis_ds
    )
    detection = detect_overlap_objects(
        analysis[1], analysis[2], marrow, pixel_um,
        args.red_threshold, args.cd31_threshold,
        args.cd31_expand_um, args.min_object_um2,
    )
    centres = detection["centres_yx"]
    if len(centres) < 5:
        raise ValueError(f"Only {len(centres)} red/CD31 objects detected")

    distance = ndi.distance_transform_edt(~boundary, sampling=pixel_um)
    rr = np.clip(np.rint(centres[:, 0]).astype(int), 0, distance.shape[0] - 1)
    cc = np.clip(np.rint(centres[:, 1]).astype(int), 0, distance.shape[1] - 1)
    observed = distance[rr, cc]
    rng = stable_rng(args.seed, path)
    eligible_uniform = distance[marrow & ~boundary]
    uniform_null, p_uniform = monte_carlo(
        observed, eligible_uniform, args.iterations, rng
    )
    display_random = np.random.default_rng(args.seed + 1).choice(
        eligible_uniform, size=observed.size, replace=False
    )
    histogram_rows = distance_histogram(
        observed, eligible_uniform, args.distance_bin_um
    )
    cd31_domain = distance[detection["cd31_binary"] & marrow]
    cd31_null, p_cd31 = monte_carlo(
        observed, cd31_domain, args.iterations, rng
    )
    obs_median = float(np.median(observed))
    uniform_median = float(np.median(uniform_null))
    cd31_median = float(np.median(cd31_null))
    u_lo, u_hi = np.quantile(uniform_null, (0.025, 0.975))
    c_lo, c_hi = np.quantile(cd31_null, (0.025, 0.975))

    save_overlay(
        prefix.with_name(prefix.name + "_AUTO_QC.png"),
        analysis, marrow, boundary, centres,
        f"{path.name}\nQC={auto['metrics']['qc_status']}; "
        f"objects={len(centres)}; pixel={pixel_um:.3f} um",
    )
    save_null_plot(
        prefix.with_name(prefix.name + "_uniform_null.png"),
        uniform_null, obs_median, path.name,
    )
    save_distance_histogram(
        prefix.with_name(prefix.name + "_distance_histogram.png"),
        histogram_rows, path.name, len(centres),
    )
    save_distance_boxplot(
        prefix.with_name(prefix.name + "_distance_boxplot.png"),
        observed, display_random, p_uniform, path.name, args.iterations,
    )
    write_csv(
        prefix.with_name(prefix.name + "_distance_boxplot_points.csv"),
        ([{"series": "actual", "point_id": i + 1, "distance_um": float(value)}
          for i, value in enumerate(observed)]
         + [{"series": "random", "point_id": i + 1, "distance_um": float(value)}
            for i, value in enumerate(display_random)]),
        ["series", "point_id", "distance_um"],
    )
    write_csv(
        prefix.with_name(prefix.name + "_distance_histogram.csv"),
        histogram_rows,
        ["bin_start_um", "bin_end_um", "observed_count",
         "expected_random_count", "random_count_95pct_low",
         "random_count_95pct_high", "observed_percent",
         "expected_random_percent", "observed_over_expected"],
    )

    annotation = {
        "version": VERSION,
        "source": file_identity(path),
        "source_metadata": source_meta,
        "um_per_pixel_original": um_per_pixel,
        "transposed_for_analysis": transposed,
        "analysis_downsample": analysis_ds,
        "analysis_pixel_um": pixel_um,
        "distance_histogram": {
            "bin_width_um": args.distance_bin_um,
            "random_domain": "marrow pixels excluding boundary",
            "random_count_interval": "pointwise 95% hypergeometric",
            "random_sample_size": len(centres),
        },
        "distance_boxplot": {
            "display_random_seed": args.seed + 1,
            "display_random_sample_size": len(centres),
            "display_random_sample_is_used_for_p_value": False,
            "p_value": p_uniform,
            "p_value_definition": "one-sided spatial Monte Carlo test of median distance",
        },
        "automatic_boundary": {
            "detect_downsample": auto["detect_downsample"],
            "detect_pixel_um": auto["detect_pixel_um"],
            "metrics": auto["metrics"],
            "analysis_coordinates": boundary_coords,
        },
        "signal_detection": {
            "red_threshold": detection["red_threshold"],
            "cd31_threshold": detection["cd31_threshold"],
            "cd31_expand_px": detection["expand_px"],
            "cd31_expand_um_requested": args.cd31_expand_um,
            "minimum_area_px": detection["min_area_px"],
            "minimum_area_um2_requested": args.min_object_um2,
            "object_definition": "red/CD31 overlap connected component",
        },
    }
    prefix.with_name(prefix.name + "_auto_annotations.json").write_text(
        json.dumps(annotation, indent=2), encoding="utf-8"
    )

    object_rows = []
    for object_id, ((row, col), area, dist) in enumerate(
        zip(centres, detection["areas_um2"], observed), start=1
    ):
        x_orig, y_orig = original_xy(row, col, analysis_ds, transposed)
        object_rows.append({
            "image": path.name,
            "animal_id": animal_id or "",
            "object_id": object_id,
            "x_original_px": x_orig,
            "y_original_px": y_orig,
            "overlap_area_um2": float(area),
            "distance_um": float(dist),
        })
    write_csv(
        prefix.with_name(prefix.name + "_objects.csv"),
        object_rows,
        ["image", "animal_id", "object_id", "x_original_px", "y_original_px",
         "overlap_area_um2", "distance_um"],
    )
    write_csv(
        prefix.with_name(prefix.name + "_uniform_null_medians.csv"),
        [{"iteration": i + 1, "median_um": float(value)}
         for i, value in enumerate(uniform_null)],
        ["iteration", "median_um"],
    )

    summary = {
        "image": path.name,
        "animal_id": animal_id or "",
        "qc_status": auto["metrics"]["qc_status"],
        "qc_issues": "; ".join(auto["metrics"]["qc_issues"]),
        "um_per_pixel_original": um_per_pixel,
        "analysis_pixel_um": pixel_um,
        "roi_length_um_requested": args.roi_length_um,
        "roi_length_um_effective": (
            (auto["roi_stop_y_ds"] - auto["roi_start_y_ds"])
            * auto["detect_pixel_um"]
        ),
        "distance_bin_um": args.distance_bin_um,
        "n_overlap_objects": len(centres),
        "roi_area_um2": float(np.count_nonzero(marrow) * pixel_um**2),
        "median_marrow_width_um": auto["metrics"]["median_width_um"],
        "boundary_contrast_fraction": auto["metrics"]["median_contrast_fraction"],
        "boundary_width_cv": auto["metrics"]["width_cv"],
        "red_threshold": detection["red_threshold"],
        "cd31_threshold": detection["cd31_threshold"],
        "observed_median_um": obs_median,
        "uniform_random_median_um": uniform_median,
        "uniform_random_95pct_low_um": float(u_lo),
        "uniform_random_95pct_high_um": float(u_hi),
        "observed_over_uniform_random": obs_median / uniform_median,
        "p_closer_than_uniform_one_sided": p_uniform,
        "cd31_pixel_random_median_um": cd31_median,
        "cd31_pixel_random_95pct_low_um": float(c_lo),
        "cd31_pixel_random_95pct_high_um": float(c_hi),
        "observed_over_cd31_pixel_random": obs_median / cd31_median,
        "p_closer_than_cd31_pixels_one_sided": p_cd31,
    }
    return summary, object_rows, uniform_null


def animal_level(rows: list[dict], output: Path) -> None:
    """One value per animal; t tests use animal log(observed/random) ratios."""
    by_animal: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("animal_id"):
            by_animal.setdefault(row["animal_id"], []).append(row)
    if not by_animal:
        return

    animal_rows = []
    for animal, fields in sorted(by_animal.items()):
        groups = {str(x.get("group", "")).strip() for x in fields}
        if len(groups) != 1:
            raise ValueError(f"Animal {animal} has conflicting groups: {groups}")
        ratios = np.asarray(
            [float(x["observed_over_uniform_random"]) for x in fields],
            dtype=float,
        )
        if np.any(~np.isfinite(ratios)) or np.any(ratios <= 0):
            raise ValueError(f"Animal {animal} has invalid observed/random ratios")
        log_ratio = float(np.median(np.log(ratios)))
        animal_rows.append({
            "animal_id": animal,
            "group": groups.pop(),
            "n_fields": len(fields),
            "median_observed_over_uniform_random": float(np.median(ratios)),
            "median_log_ratio": log_ratio,
            "exp_median_log_ratio": float(np.exp(log_ratio)),
        })
    write_csv(
        output / "animal_summary.csv", animal_rows,
        ["animal_id", "group", "n_fields",
         "median_observed_over_uniform_random", "median_log_ratio",
         "exp_median_log_ratio"],
    )

    groups = sorted({row["group"] for row in animal_rows})
    if len(groups) > 1 and "" in groups:
        raise ValueError("Some animals have a group and others do not")
    test_rows = []
    if groups == [""]:
        subsets = [("all", animal_rows)]
    else:
        subsets = [
            (group, [row for row in animal_rows if row["group"] == group])
            for group in groups
        ]
    for label, subset in subsets:
        values = np.asarray([row["median_log_ratio"] for row in subset])
        n = len(values)
        record = {
            "test": "one_sample_student_t_vs_random",
            "group_a": label, "group_b": "random_ratio_1",
            "n_a": n, "n_b": "",
            "mean_log_ratio_a": float(np.mean(values)),
            "mean_log_ratio_b": 0.0,
        }
        if n < 3:
            record["status"] = "insufficient_animals_minimum_3"
        elif np.std(values, ddof=1) == 0:
            record["status"] = "zero_between_animal_variance"
        else:
            result = ttest_1samp(values, popmean=0.0)
            df = n - 1
            mean = float(np.mean(values))
            margin = float(student_t.ppf(0.975, df) * np.std(values, ddof=1) / np.sqrt(n))
            record.update({
                "estimate_log_ratio": mean,
                "ci95_low_log_ratio": mean - margin,
                "ci95_high_log_ratio": mean + margin,
                "ratio_estimate": float(np.exp(mean)),
                "ratio_ci95_low": float(np.exp(mean - margin)),
                "ratio_ci95_high": float(np.exp(mean + margin)),
                "t_statistic": float(result.statistic),
                "df": df,
                "p_two_sided": float(result.pvalue),
                "p_closer_one_sided": float(student_t.cdf(result.statistic, df)),
                "status": "computed",
            })
        test_rows.append(record)

    if len(groups) == 2:
        first, second = (
            np.asarray([row["median_log_ratio"] for row in subset])
            for _, subset in subsets
        )
        na, nb = len(first), len(second)
        record = {
            "test": "two_sample_student_t_equal_variance",
            "group_a": groups[0], "group_b": groups[1],
            "n_a": na, "n_b": nb,
            "mean_log_ratio_a": float(np.mean(first)),
            "mean_log_ratio_b": float(np.mean(second)),
        }
        if min(na, nb) < 3:
            record["status"] = "insufficient_animals_minimum_3_per_group"
        else:
            df = na + nb - 2
            pooled_variance = (
                (na - 1) * np.var(first, ddof=1)
                + (nb - 1) * np.var(second, ddof=1)
            ) / df
            if pooled_variance == 0:
                record["status"] = "zero_between_animal_variance"
            else:
                result = ttest_ind(first, second, equal_var=True)
                diff = float(np.mean(first) - np.mean(second))
                margin = float(
                    student_t.ppf(0.975, df)
                    * np.sqrt(pooled_variance * (1 / na + 1 / nb))
                )
                record.update({
                    "estimate_log_ratio": diff,
                    "ci95_low_log_ratio": diff - margin,
                    "ci95_high_log_ratio": diff + margin,
                    "ratio_estimate": float(np.exp(diff)),
                    "ratio_ci95_low": float(np.exp(diff - margin)),
                    "ratio_ci95_high": float(np.exp(diff + margin)),
                    "t_statistic": float(result.statistic),
                    "df": df,
                    "p_two_sided": float(result.pvalue),
                    "status": "computed",
                })
        test_rows.append(record)

    columns = [
        "test", "group_a", "group_b", "n_a", "n_b",
        "mean_log_ratio_a", "mean_log_ratio_b",
        "estimate_log_ratio", "ci95_low_log_ratio", "ci95_high_log_ratio",
        "ratio_estimate", "ratio_ci95_low", "ratio_ci95_high",
        "t_statistic", "df", "p_two_sided", "p_closer_one_sided", "status",
    ]
    write_csv(output / "animal_test.csv", test_rows, columns)
    save_animal_ratio_boxplot(
        output / "animal_ratio_boxplot.png", animal_rows, test_rows
    )
    message = [
        "Animal is the independent unit. Each animal contributes the median of "
        "its field log(observed/random) ratios.",
        "Ratio < 1 means closer to endosteum than uniform random marrow points.",
        "One-sample Student t: animal log ratios vs 0. Two-sided p is primary; "
        "one-sided p tests log ratio < 0.",
        "Two-sample Student t (when exactly two groups): equal-variance, "
        "two-sided test of group_a minus group_b animal log ratios.",
        "Check animal-level distribution, outliers, and variance assumptions "
        "before interpreting t tests; image-level Monte Carlo p values are separate.",
        "",
    ]
    for record in test_rows:
        label = f"{record['test']}: {record['group_a']} vs {record['group_b']}"
        if record["status"] == "computed":
            message.append(
                f"{label}: n={record['n_a']}"
                + (f",{record['n_b']}" if record["n_b"] != "" else "")
                + f", t({record['df']})={record['t_statistic']:.5g}, "
                f"two-sided p={record['p_two_sided']:.6g}, "
                f"ratio estimate={record['ratio_estimate']:.5g} "
                f"(95% CI {record['ratio_ci95_low']:.5g} to "
                f"{record['ratio_ci95_high']:.5g})"
                + (f", closer one-sided p={record['p_closer_one_sided']:.6g}"
                   if "p_closer_one_sided" in record else "")
            )
        else:
            message.append(f"{label}: {record['status']}")
    (output / "animal_test.txt").write_text(
        "\n".join(message) + "\n", encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    files = discover_tiffs(args.input, args.recursive)
    output = args.output or (
        (args.input if args.input.is_dir() else args.input.parent)
        / "auto_endosteal_results"
    )
    output.mkdir(parents=True, exist_ok=True)
    existing = list(output.glob("*_auto_annotations.json"))
    if existing and not args.overwrite:
        raise ValueError(
            f"Output folder already contains {len(existing)} analyzed image(s); "
            "use --overwrite or choose a new --output folder"
        )
    manifest = load_manifest(args.manifest)
    settings = {
        "version": VERSION,
        **{
            key: (str(value.resolve()) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "files": [str(f.resolve()) for f in files],
        "selection_rule": (
            "DAPI-only automatic shaft window maximizing marrow/cortex contrast "
            "and boundary stability; red signal is not used to choose the ROI"
        ),
    }
    (output / "run_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8"
    )

    summaries, all_objects, errors = [], [], []
    for index, path in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {path}", flush=True)
        prefix = output / f"{index:03d}_{path.stem}"

        metadata = manifest.get(path.name, {})
        animal_id = metadata.get("animal_id")
        try:
            summary, objects, _ = analyze_one(path, prefix, args, animal_id)
            summary["group"] = metadata.get("group", "")
            for obj in objects:
                obj["group"] = summary["group"]
            summaries.append(summary)
            all_objects.extend(objects)
            print(
                f"  {summary['qc_status']} | n={summary['n_overlap_objects']} | "
                f"observed={summary['observed_median_um']:.2f} um | "
                f"random={summary['uniform_random_median_um']:.2f} um | "
                f"p={summary['p_closer_than_uniform_one_sided']:.6f}",
                flush=True,
            )
        except Exception as exc:
            errors.append({
                "image": path.name, "status": "FAILED",
                "message": str(exc),
            })
            print(f"  FAILED: {exc}", file=sys.stderr, flush=True)

    summary_fields = [
        "image", "animal_id", "group", "qc_status", "qc_issues",
        "um_per_pixel_original", "analysis_pixel_um",
        "roi_length_um_requested", "roi_length_um_effective",
        "distance_bin_um", "n_overlap_objects",
        "roi_area_um2", "median_marrow_width_um",
        "boundary_contrast_fraction", "boundary_width_cv",
        "red_threshold", "cd31_threshold", "observed_median_um",
        "uniform_random_median_um", "uniform_random_95pct_low_um",
        "uniform_random_95pct_high_um", "observed_over_uniform_random",
        "p_closer_than_uniform_one_sided", "cd31_pixel_random_median_um",
        "cd31_pixel_random_95pct_low_um",
        "cd31_pixel_random_95pct_high_um",
        "observed_over_cd31_pixel_random",
        "p_closer_than_cd31_pixels_one_sided",
    ]
    write_csv(output / "image_summary.csv", summaries, summary_fields)
    write_csv(
        output / "all_overlap_objects.csv", all_objects,
        ["image", "animal_id", "group", "object_id", "x_original_px",
         "y_original_px", "overlap_area_um2", "distance_um"],
    )
    write_csv(
        output / "processing_log.csv", errors,
        ["image", "status", "message"],
    )
    animal_level(summaries, output)

    report = [
        f"Automatic endosteal batch analysis v{VERSION}",
        f"Input images discovered: {len(files)}",
        f"Successfully analyzed: {len(summaries)}",
        f"Failed or skipped: {len(errors)}",
        f"Requested shaft ROI length: {args.roi_length_um:.1f} um",
        "",
        "Interpret n_overlap_objects as red/CD31 overlap connected structures, "
        "not independent endothelial cells.",
        "Per-image Monte Carlo p values are conditional spatial tests for each "
        "field. They are not biological-replicate p values.",
        "Inspect every *_AUTO_QC.png before accepting its numerical result.",
    ]
    if summaries:
        ratios = np.asarray(
            [float(x["observed_over_uniform_random"]) for x in summaries]
        )
        report.extend([
            "",
            f"Median field ratio (observed/random): {np.median(ratios):.6f}",
            f"Fields with ratio < 1: {int(np.count_nonzero(ratios < 1))}/"
            f"{len(ratios)}",
        ])
    (output / "REPORT.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(report), flush=True)
    return 0 if summaries else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
