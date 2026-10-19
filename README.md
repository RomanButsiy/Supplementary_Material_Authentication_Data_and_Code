# Supplementary data and code

**Session-disjoint ECG authentication using rhythm-adaptive zone-cyclic morphology**

Roman Butsii, Serhii Lupenko and Imad Ireifidzh

## Contents

- `Supplementary_Material.pdf`: methods, tables S1-S7 and figures S1-S2.
- `data/cycles.npz`: 36 arrays (`NNN_S1`, `NNN_S2`), containing 13,551 aligned and mean-centered cardiac cycles from 18 subjects; each row has 360 float64 coordinates.
- `code/run_analysis.py`: model fitting, primary evaluation, sensitivity experiments and identity holdout.
- `code/summarize.py`: numerical summaries, conditional bootstrap intervals and paired comparisons.
- `code/make_figures.py`: main figures 1-6 and supplementary figures S1-S2.
- `code/run.py`: entry point that runs these scripts and collects their results.
- `results/`: numerical CSV outputs, model scores in one NPZ file, and figures in PDF.

## Run

Use Python 3.13 with the pinned dependencies (reference interpreter: 3.13.5). From this directory:

```bash
python -m pip install -r requirements.txt

# Regenerate numerical summaries and figures from the supplied scores
python code/run.py --summaries --output recalculated

# Refit all models and reproduce all experiments, summaries and figures
python code/run.py --output rerun --jobs 1
```

Both commands preserve the supplied results and write to a new output directory. Full fitting includes Gaussian processes and can take substantially longer. Temporary per-job files are removed automatically after the command finishes. Dependencies must be installed before running; the calculation itself requires no network access.

## Read results

The three `*_all.csv` files contain individual evaluation results. `summary_*.csv` contains macro-means and conditional 95% claimant-bootstrap limits. `counts_and_bounds.csv` contains comparison counts; `sample_counts.csv` contains partition sizes. `security_operating_points.csv` contains calibration-selected thresholds and achieved test errors.

All CSV rates are fractions; figures and supplementary tables display percentages. Genuine=1, impostor=0; acceptance is `score >= threshold`. Read CSV thresholds with `pandas.read_csv(..., float_precision='round_trip')`.

The combined `results/scores.npz` uses keys such as `002_S1_K5__y`, `002_S1_K5__yv`, `002_S1_K5__s_Linear_SVM` and `002_S1_K5__v_Linear_SVM`. The prefix identifies claimant, source block and K. `y`/`yv` are test/calibration labels; `s_`/`v_` are the corresponding model scores. Each prefix also has SIC `mu`, `sd`, `sic_kappa` and `sic_eta`. The opposite block is used for testing. `score_items()` in `run_analysis.py` reads these entries.

## Data source and scope

Descriptors are derived from García-González M A and Argelagós-Palau A, *Combined measurement of ECG, Breathing and Seismocardiograms (CEBSDB)*, v1.0.0, PhysioNet (2014), https://doi.org/10.13026/C2KW23, under the Open Data Commons Attribution License v1.0. Preserve this source attribution when redistributing the data.

The package reproduces descriptor-level experiments. It does not contain raw acquisition samples or timestamps. SIC settings were chosen during exploratory evaluation of the same cohort, including test results; they require independent validation. The supplementary material specifies the complete protocol and interpretation of the conditional intervals.
