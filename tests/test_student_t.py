from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import numpy as np
from scipy.stats import ttest_1samp, ttest_ind
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_endosteal_batch import animal_level, load_manifest

with TemporaryDirectory() as temp:
    output = Path(temp)
    manifest = output / "manifest.csv"
    manifest.write_text(
        "filename,animal_id,group\n"
        "a1.tif,A1,Control\n"
        "a2.tif,A1,Control\n"
        "b1.tif,B1,Treated\n", encoding="utf-8"
    )
    mapping = load_manifest(manifest)
    assert mapping["a1.tif"] == {"animal_id": "A1", "group": "Control"}
    rows = []
    control = [0.70, 0.75, 0.80]
    treated = [1.05, 1.10, 1.15]
    for group, ratios in [("Control", control), ("Treated", treated)]:
        for i, ratio in enumerate(ratios):
            animal = group + str(i)
            for field_ratio in (ratio * 0.95, ratio * 1.05):
                rows.append({
                    "animal_id": animal, "group": group,
                    "observed_over_uniform_random": field_ratio,
                })
    animal_level(rows, output)
    assert (output / "animal_ratio_boxplot.png").stat().st_size > 10000
    assert (output / "animal_ratio_boxplot.pdf").stat().st_size > 1000
    with (output / "animal_test.csv").open(newline="", encoding="utf-8-sig") as handle:
        tests = list(csv.DictReader(handle))
    assert len(tests) == 3, tests
    means = {
        group: np.array([np.median(np.log([ratio * 0.95, ratio * 1.05]))
                         for ratio in ratios])
        for group, ratios in [("Control", control), ("Treated", treated)]
    }
    expected_control = ttest_1samp(means["Control"], 0)
    expected_between = ttest_ind(
        means["Control"], means["Treated"], equal_var=True
    )
    assert np.isclose(float(tests[0]["t_statistic"]), expected_control.statistic)
    assert np.isclose(float(tests[0]["p_two_sided"]), expected_control.pvalue)
    assert np.isclose(float(tests[2]["t_statistic"]), expected_between.statistic)
    assert np.isclose(float(tests[2]["p_two_sided"]), expected_between.pvalue)
    assert tests[0]["p_closer_one_sided"] and not tests[2]["p_closer_one_sided"]
    assert "\n" in (output / "animal_test.txt").read_text(encoding="utf-8")
    print("PASS: manifest, one-sample Student t, two-sample Student t, output format")

