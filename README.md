# PAC-Bayes Bounds for Domain Adaptation

This repository runs a Study-3 PAC-Bayes experiment comparing transfer
learning (`TL`) and direct learning (`DL`) for theory-specific option-pricing
neural networks.

The true DGP is a Bates-mimic surface with features `x1..x5`. BSM learners use
only `x1..x3`, HSV learners use `x1..x4`, and each theory gets an independent
PAC-Bayes bound optimization. The main experiment varies Bates jump strength
`a_J` and optionally repeats the full experiment over multiple random seeds.

## Project Structure

```text
.
├── main.py        # entry point
├── runner.py      # Runs single-seed or multi-seed workflows from parsed args 
├── experiment.py  # Core experiment loops for one a_J grid and many seeds
├── data.py        # True/theory data-generating processes
├── model.py       # Toy MLP and parameter utilities
├── training.py    # Source pretraining and Stage-1 empirical training
├── pacbayes.py    # PAC-Bayes posterior optimization and bound computation
├── metrics.py     # Multi-seed summaries and rank/monotonicity diagnostics
├── plotting.py    # Plot generation and training-history collection helpers
└── utils.py       # Shared device, seed, and model-state helpers
```

## Run

From the project directory:

```bash
python3 main.py
```
