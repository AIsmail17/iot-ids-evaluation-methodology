# Lightweight ML-Based Intrusion Detection for IoT Networks

Code for an internship research project on the CIC-IoT2023 dataset. Six
classes: Benign, DDoS, DoS, Reconnaissance, Spoofing, Brute Force.

Every model is evaluated twice:

- **Protocol A (random split):** the standard method used in the literature.
- **Protocol B (grouped split):** each capture file goes entirely into
  training or entirely into testing, so the model is tested on recording
  sessions it has never seen.

Everything else is identical between the two, so the difference in scores
measures how much of the reported performance comes from the evaluation
method rather than the model.

## Main results (pooled 5-fold, macro-recall, six classes)

| Model | Protocol A | Protocol B | Change |
|---|---|---|---|
| Majority-class baseline | 0.1667 | 0.1667 | 0.0000 |
| Logistic Regression | 0.6501 | 0.5298 | −0.1203 |
| Decision Tree | 0.6969 | 0.5704 | −0.1265 |
| Random Forest | 0.7463 | 0.6196 | −0.1267 |
| LightGBM | 0.7951 | 0.6217 | −0.1734 |
| XGBoost | 0.7981 | 0.6244 | −0.1737 |

The best model, XGBoost, is also among the smallest (about 1.7 MB, against
about 2 GB for Random Forest).

## Repository layout

| Path | Contents |
|---|---|
| `src/load.py` | Reads the 309 raw CSV files into one Parquet file, adding `label` and `source_file` columns |
| `src/sample.py` | Class-capped subsample (46.8M → 4.46M rows) |
| `src/split.py` | Protocol A and Protocol B train/test splits |
| `src/preprocess.py` | Scaling fitted on the training partition only; label encoding |
| `src/features.py` | Feature ranking, fitted on the training partition only |
| `src/train.py` | Model training and evaluation; writes results rows |
| `src/kfold.py` | Pooled 5-fold evaluation used for the main results |
| `src/bench.py` | Model size, latency and throughput on a single core |
| `src/dataset_stats.py`, `src/duplicates_full.py` | Dataset statistics and duplicate analysis |
| `src/report_tables.py`, `src/confusion_figures.py`, `src/methodology_figure.py`, `src/analyse.py` | Generate the tables and figures from the results files |
| `notebooks/01_preprocessing_and_eda.ipynb` | Preprocessing and exploratory analysis, with outputs |
| `configs/` | Experiment settings (YAML) |
| `results/tables/`, `results/figures/` | Generated tables and figures used in the report |

## Running

The dataset is not included. Download the CSV release of CIC-IoT2023 from the
Canadian Institute for Cybersecurity and set `RAW_ROOT` in `src/load.py` to
its `CSV` folder.

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python.exe src\load.py
.venv\Scripts\python.exe src\sample.py
.venv\Scripts\python.exe src\kfold.py
```
