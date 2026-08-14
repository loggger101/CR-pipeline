# CR-Pipeline Examples

This directory contains example scripts demonstrating CR-Pipeline usage.

## Quick Start

```bash
# Run basic training
python examples/01_basic_training.py

# Run tournament evaluation
python examples/02_tournament_evaluation.py

# Run hyperparameter optimization
python examples/03_hyperparameter_optimization.py

# Launch visualization
python examples/04_visualization.py
```

## Example Descriptions

### 01_basic_training.py
Demonstrates basic evolutionary training with:
- Configuration and initialization
- Training loop execution
- Checkpoint saving
- Results evaluation

### 02_tournament_evaluation.py
Demonstrates tournament-style evaluation with:
- Round-robin tournament setup
- ELO rating computation
- Head-to-head analysis
- Bracket visualization

### 03_hyperparameter_optimization.py
Demonstrates hyperparameter optimization with:
- Bayesian optimization
- Grid search
- Random search
- Configuration comparison

### 04_visualization.py
Demonstrates visualization and reporting with:
- Streamlit dashboard launch
- Report generation
- Run comparison
- Tournament result analysis

## Advanced Usage

For advanced usage, see the main README.md for:
- Full CLI reference
- Configuration schema documentation
- Architecture options
- Training phases
- Pipeline orchestration
