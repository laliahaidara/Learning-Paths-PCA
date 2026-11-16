# Learning Paths: 0/1 Knapsack (DP) vs. the Paper's DFS vs. Greedy

Final Project for Design and Complexity of Algorithms (UnB).

## Structure
```
algorithms/               # ALGORITHM files (one module per algorithm)
  model.py                # LearningModule, Result, validation
  sorting.py              # hand-written merge sort (used by greedy and B&B)
  dfs.py                  # exhaustive DFS with pruning + backtracking (the paper's algorithm, knapsack version)
  graph_dfs.py            # DFS over the prerequisite graph (RUTICO style, extra)
  dp.py                   # DP: table (with reconstruction), vector (Θ(T) space) and memoized
  greedy.py               # greedy by score/time + max(greedy, best single module)
  branch_and_bound.py     # extra comparison: B&B with a fractional bound
  meet_in_the_middle.py   # extra comparison: Horowitz–Sahni, Θ(2^(n/2))
  fptas.py                # extra comparison: (1-ε)-approximation
  grasp.py                # bonus: GRASP metaheuristic
instances.py              # synthetic generators (uncorrelated / weakly / strongly correlated)
real_instances.py         # real, anonymized instances built from real_data/*.csv
real_data/                # anonymized module/topic data from an adaptive-learning platform
experiment.py             # main experiments: test instance + 9 experiment blocks
experiment_validation.py  # checks our algorithms against OR-Tools, SciPy and NumPy
tests/                    # unit tests that pin the report's worked example (6 modules, T = 10)
requirements.txt          # matplotlib (+ ortools, scipy, numpy only for the validation)
results/                  # CSVs, plots (PNG) and the full output of one run
report/                   # the paper (English, IEEE-style): report.pdf (final), report.tex + sections/ (LaTeX source),
                          # relatorio-final-atualizado.docx/.tex = Portuguese reading version of the report,
                          # figures/, make_tables.py (regenerates the tables from results/*.csv)
```

## How to run

### 1. Requirements
- Python 3.10 or newer (developed with 3.12)
- `experiment.py` only needs **matplotlib** (for the plots)
- `experiment_validation.py` also needs **ortools**, **scipy** and **numpy**

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
python experiment.py                            # all blocks (~1 min)
python experiment.py --quick                    # smaller sizes, faster
python experiment.py --only demo correctness    # just the listed blocks
python experiment.py --quick --only scale       # flags can be combined
```

| Block         | What it does |
|---------------|--------------|
| `demo`        | Runs every algorithm on the test instance and prints the DP table |
| `correctness` | Compares every algorithm with the exhaustive DFS on random instances |
| `time`        | DFS vs. B&B vs. meet-in-the-middle vs. DP for small n (worst and average case) |
| `scale`       | DP and greedy for large n (DFS can't run here): confirms Θ(nT) and Θ(n log n) |
| `quality`     | Greedy gap from the optimum by n, instance family and budget ratio |
| `pseudo`      | Same modules with times (and T) scaled up: shows DP is pseudo-polynomial |
| `families`    | Effort of the exact algorithms per instance family and n |
| `fptas`       | FPTAS quality and cost vs. the exact DP for several ε |
| `graph`       | The paper's DFS on a complete DAG: confirms 2^(n-1) path growth |

Output goes to `results/` (CSV files and `fig_*.png` plots); progress is printed to the terminal.
To keep a log like `results/full_output.txt`: `python experiment.py | tee results/full_output.txt`.

### 3b. Unit tests — the worked example
```bash
python -m unittest discover -s tests -v
```
They check the DP table, the chosen modules (Python Basics + Data Structures, 91 points), the
greedy result (77 points, a 15.4% gap) and the adversarial instance. If a number changes, the
"Test instance" section of the report has to change too.

### 4. Validation against external libraries — `experiment_validation.py`
```bash
python experiment_validation.py                  # all blocks (~10 min)
python experiment_validation.py --only optimum   # one block
python experiment_validation.py --only sorting pearson
```

| Block        | What it does |
|--------------|--------------|
| `optimum`    | Every exact algorithm vs. OR-Tools and SciPy/HiGHS as optimum oracles |
| `guarantees` | Approximation guarantees of greedy / FPTAS and GRASP quality vs. the OR-Tools optimum |
| `sorting`    | Our merge sort vs. Python's `sorted()` with exact fractions |
| `pearson`    | Time–score correlation: `statistics` (used in the report) vs. NumPy vs. SciPy |
| `runtime`    | Pure-Python DP runtime vs. OR-Tools (C++) and SciPy/HiGHS |
| `bnb`        | Does our B&B finish on large strongly correlated instances? (20 s limit per run) |

Output: `results/validation_*.csv`.

### 5. Update the report
After re-running the experiments, refresh the LaTeX tables and rebuild the PDF:
```bash
python report/make_tables.py      # rewrites report/sections/tab_*.tex from results/*.csv
cd report
pdflatex report.tex
pdflatex report.tex               # run twice so references and the table of contents are correct
```
New plots are written to `results/`; copy any you want into `report/figures/` to use them in the paper.
Member names are on the cover page and in the Members section of `report/report.tex` / `report/sections/*.tex`.

> **Note:** running the experiments overwrites the files in `results/`, and `make_tables.py`
> overwrites the tables in `report/sections/`. Commit or back up anything you want to keep first.
> Timing numbers will differ from machine to machine.

## Team and authorship
Lalia Sadio, Adriana Souza, Leandro Bergmann and Dalila Rezende. Lalia proposed the topic.
Lalia and Leandro built the algorithms and the experiments; Adriana and Dalila wrote the
theoretical background (Dalila used the Sebrae learning platform as her basis); every other
task was done by all four and reviewed together, and each member can explain every file.
The same division is described in the Members section of the report.

## Use of AI tools
Claude (Anthropic) was used as an assistant: English review of the report, review of the
code (an edge-case fix in `grasp.py` for `alpha = 0`, comments/docstrings, formatting, the unit
test of the worked example) and drafts of the presentation material. The group reviewed and
validated every change by re-running the experiments (identical values and operation counts
before and after). The use of AI is also disclosed in the report.
