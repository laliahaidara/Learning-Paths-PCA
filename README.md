# Learning Paths: 0/1 Knapsack with Dynamic Programming and Branch and Bound

Final Project for Design and Complexity of Algorithms (UnB).

Choosing which study modules to take within a time budget is the 0/1 knapsack problem. This
project implements and compares the two best algorithms for it: **dynamic programming (DP)**
and **branch and bound (B&B)**.

## Structure
```
algorithms/               # ALGORITHM files (one module per algorithm)
  model.py                # LearningModule, Result, validation
  sorting.py              # hand-written merge sort (used by B&B)
  dp.py                   # DP: table (with reconstruction), vector (Θ(T) space) and memoized
  branch_and_bound.py     # B&B with a fractional (Dantzig) bound
common.py                 # shared helpers: save_csv, measure (median time), run_dp
instances.py              # synthetic generators (uncorrelated / weakly / strongly correlated)
real_instances.py         # real, anonymized instances built from real_data/*.csv
real_data/                # anonymized module/topic data from an adaptive-learning platform
experiment.py             # EXPERIMENT file: test instance + 5 experiment blocks
experiment_validation.py  # checks our algorithms against OR-Tools, SciPy and NumPy
experiment_real.py        # experiments on the real (anonymized) platform instances
tests/                    # unit tests that pin the report's worked example (6 modules, T = 10)
requirements.txt          # matplotlib (+ ortools, scipy, numpy only for the validation)
results/                  # CSVs, plots (PNG) and the full output of one run
final-report-en.pdf       # THE SUBMITTED REPORT (English, ABNT format)
```

## How to run

### 1. Requirements
- Python 3.10 or newer (developed with 3.12)
- `experiment.py` only needs **matplotlib** (for the plots)
- `experiment_validation.py` also needs **ortools**, **scipy** and **numpy**
- `experiment_real.py` only needs what `experiment.py` needs

### 2. Set up a virtual environment (recommended)
Run these from the project root (the folder containing this README):
```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
If you only want the main experiments, `pip install matplotlib` is enough.

### 3. Main experiments — `experiment.py`
```bash
python experiment.py                            # all blocks (~10 s)
python experiment.py --quick                    # smaller sizes, faster
python experiment.py --only demo correctness    # just the listed blocks
```

| Block         | What it does |
|---------------|--------------|
| `demo`        | Runs DP (three variants) and B&B on the test instance and prints the DP table |
| `correctness` | Compares both algorithms with a brute-force reference on 1,200 random instances |
| `scale`       | DP for large n: confirms that the cost is exactly n(T+1) |
| `pseudo`      | Same modules with times (and T) scaled up: shows DP is pseudo-polynomial |
| `families`    | Cost of B&B and DP per instance family and n |

Output goes to `results/` (CSV files and `fig_*.png` plots); progress is printed to the terminal.

### 3b. Unit tests — the worked example
```bash
python -m unittest discover -s tests -v
```
They check the DP table, the chosen modules (Python Basics + Data Structures, 91 points), the
memoized DP (32 states) and B&B. If a number changes, the "Test instance" part of the report
has to change too.

### 4. Validation against external libraries — `experiment_validation.py`
```bash
python experiment_validation.py                  # all blocks (~10 min)
python experiment_validation.py --only optimum   # one block
```

| Block        | What it does |
|--------------|--------------|
| `optimum`    | DP and B&B vs. OR-Tools and SciPy/HiGHS as optimum oracles |
| `sorting`    | Our merge sort vs. Python's `sorted()` with exact fractions |
| `pearson`    | Time–score correlation: `statistics` (used in the report) vs. NumPy vs. SciPy |
| `runtime`    | Pure-Python DP runtime vs. OR-Tools (C++) and SciPy/HiGHS |
| `bnb`        | Does our B&B finish on large strongly correlated instances? (20 s limit per run) |

Output: `results/validation_*.csv`.

### 5. Real data — `experiment_real.py`
```bash
python experiment_real.py         # real GuIA instances (--only profile / algorithms / pseudopoly / sweep)
```
Output: `results/real_*.csv`. The three scripts share `common.py`.

> **Note:** running the experiments overwrites the files in `results/`. Timing numbers will
> differ from machine to machine; operation counts do not.

## Team and authorship
Lalia Sadio, Adriana Souza, Leandro Bergmann and Dalila Rezende. Lalia proposed the topic.
Lalia and Leandro built the algorithms and the experiments; Adriana and Dalila wrote the
theoretical background (Dalila used the Sebrae learning platform as her basis); every other
task was done by all four and reviewed together, and each member can explain every file.
The same division is described in the Members section of the report.

## Use of AI tools
Claude (Anthropic) was used as an assistant: English review of the report, review of the
code (comments/docstrings, formatting, the unit test of the worked example) and drafts of the
presentation material. The group reviewed and validated every change by re-running the
experiments. The use of AI is also disclosed in the report.
