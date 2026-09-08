# CR-Pipeline Examples

Example scripts that demonstrate the **current** public API, verified end-to-end
against a live checkout (each one actually runs — see "Verified" below). They are
kept small on purpose: they show *how* to call the pipeline from your own code,
not how to train well. For real training use `crp train` or the desktop app.

## Quick Start

```bash
# 1. Train a tiny population (3 generations x 16 agents) -> runs/example_basic/
python examples/01_basic_training.py

# 2. Evaluate that saved population in a Swiss tournament + ELO
python examples/02_tournament_evaluation.py

# 3. Hyperparameter search over GA settings (each trial = one short real run;
#    the default budget is ~9 trials, so expect several minutes)
python examples/03_hyperparameter_optimization.py

# 4. Reports + programmatic run comparison (no browser needed).
#    Uncomment launch_dashboard() inside to open the Streamlit dashboard instead.
python examples/04_visualization.py
```

## Example Descriptions

### `01_basic_training.py` — basic evolutionary training
- Builds a `TrainingConfig` (population, elites, GA rates, tournament mode) and
  runs it through `EvolutionTrainer.train()` as a context manager.
- Prints generations completed, best fitness (ELO in tournament mode) and the
  best agent record.

### `02_tournament_evaluation.py` — tournament evaluation + ELO
- Loads every genome from the newest `gen_XXXX/population.pt` under
  `runs/example_basic/` (the same checkpoint format training writes).
- Runs a **Swiss** tournament via `FitnessEvaluator.run_tournament` (the path
  `crp tournament` uses) and prints rankings, win/draw/loss records, final ELO
  and one head-to-head record.

### `03_hyperparameter_optimization.py` — hyperparameter optimization
- Defines an objective that runs one short real training run per candidate and
  scores it by best fitness reached (the same design as `crp hpo`).
- Shows all three optimizers behind the shared interface: Bayesian (default),
  random search, grid search.

### `04_visualization.py` — visualization & reporting
- `ReportGenerator.generate_training_report(...)` for HTML reports with charts.
- `RunManager.compare_runs([...], metric="best")` to compare fitness curves of
  the two best runs in `runs/`.
- `run_advanced_dashboard(runs_dir=...)` launches the 8-tab Streamlit dashboard
  (long-running server, so it is not called by default).

## Verified

All four were executed against a live checkout on 2026-09-08:

| Example | Result of the verification run |
|---------|-------------------------------|
| `01_basic_training.py` | 3 generations completed, best fitness (ELO) printed, checkpoint written to `runs/example_basic/gen_0003/`. |
| `02_tournament_evaluation.py` | Loaded 16 agents from that checkpoint; full Swiss ranking + ELO + H2H record printed. |
| `03_hyperparameter_optimization.py` | Single-trial objective scored a real mini-run (ELO ≈ 1531); optimizer construction and result access verified. |
| `04_visualization.py` | Generated `reports/training_report.html`; compared two existing runs by best fitness. |

## Notes & gotchas

- **Windows + multiprocessing:** run examples as files (`python examples/...py`).
  Piping code into the interpreter with `-c`/stdin breaks the worker pool on
  Windows (spawned workers re-import `__main__`, which has no file).
- **Checkpoint shape:** genomes must match the current default policy
  (20,071 parameters). Checkpoints from earlier builds of this project are
  rejected with a clear message — see the main README's "Note on checkpoints".
- Example run directories (`runs/example_*`, `reports/`) are gitignored; they
  accumulate locally and can be deleted at any time.
