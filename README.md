# Endosteal distance analysis

[中文说明](README.zh-CN.md)

A reproducible 2D analysis of whether Tomato-positive, CD31-overlapping structures lie closer to the endosteum than uniformly sampled points from the same marrow region. The shaft window and the two marrow-facing boundaries are selected from DAPI without using the red signal. This repository contains code and synthetic tests only; no microscopy images or experimental results are included.

## Requirements

Python 3.10 or newer. Tested with Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, scikit-image 0.24.0, tifffile 2023.4.12, and Matplotlib 3.9.2.

On Windows, create a local environment in this directory:

~~~powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
~~~

The BAT launcher uses .venv automatically. It can also use the Python launcher or a Python executable selected with the ENDOSTEAL_PYTHON environment variable.

## Input and run

Use an original multichannel TIFF of a long-bone section, with channels DAPI=0, Tomato=1, CD31=2. The TIFF needs XY calibration in micrometers per pixel, or provide --um-per-pixel. The analysis is 2D even if the TIFF contains a Z stack.

On Windows, drag one TIFF or a folder of TIFFs onto run_auto_endosteal.bat. The launcher prompts for shaft ROI length in micrometers; Enter accepts 3000. It writes each run to a separate directory under runs/. For a repeatable folder default, place roi_length_um.txt in the TIFF folder with one positive integer on the first line.

Direct command-line example:

~~~powershell
.\.venv\Scripts\python.exe .\auto_endosteal_batch.py "C:\path\to\images" --output "C:\path\to\results" --roi-length-um 3000
~~~

For animal-level tests, place manifest.csv in the TIFF folder. It needs filename and animal_id columns; group is optional for two-group comparisons. See manifest_example.csv. Use one animal ID for all images from that animal.

## Outputs and interpretation

- Each AUTO_QC.png shows the selected shaft region, endosteal traces, and detected red/CD31 structures. Inspect every image before using its measurements.
- image_summary.csv reports observed and random median distances, the observed/random ratio, ROI area and length, and one-sided per-image spatial Monte Carlo p values.
- Each distance_boxplot.png and PDF shows all observed structure distances and one equal-sized random sample. The p label comes from the full Monte Carlo distribution of random medians, not from treating plotted structures as independent biological replicates.
- Each distance_histogram.png and CSV compares observed counts with expected random counts across distance bins.
- With a manifest, animal_summary.csv and animal_test.csv summarize independent animals. The one-sample Student t test is on each animal's median field log(observed/random) ratio versus zero. When exactly two groups are present, an equal-variance two-sample Student t test compares animal log ratios. The animal_ratio_boxplot.png shows one point per animal.

A red/CD31 connected component is a signal structure, not necessarily one endothelial cell. Automatic tracing can fail near metaphyses or unusual sections. The analysis skips QC failures by default. Use the same ROI length and acquisition conditions within an experiment, and interpret image-level spatial p values separately from animal-level inference.

## Tests

~~~powershell
.\.venv\Scripts\python.exe .\tests\test_auto_endosteal_batch.py
.\.venv\Scripts\python.exe .\tests\test_distance_histogram.py
.\.venv\Scripts\python.exe .\tests\test_student_t.py
~~~

The tests use synthetic data and do not need experimental TIFFs.