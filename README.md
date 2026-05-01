# Missingness-Aware Pedestrian Trajectory Prediction

Code repository for **Missingness-Aware Pedestrian Trajectory Prediction from Visual Observations**.

This project studies short-horizon pedestrian trajectory prediction on ETH/UCY data when the observed visual track is incomplete. Each sample uses 8 observed 2D positions and predicts the next 12 positions. The predicted trajectory is then mapped to a simple navigation action: `stop`, `go`, `turn_left`, or `turn_right`.

## Overview

The project compares four predictors:

- **Constant Velocity**: extrapolates from the two most recent valid observations.
- **Vanilla LSTM**: encoder-decoder LSTM that predicts future coordinates directly from filled coordinate histories.
- **Missingness-Aware LSTM**: uses motion features, observation masks, and missing-gap features, then predicts a residual correction over a constant-velocity forecast.
- **Missingness-Aware Transformer**: uses the same missingness-aware motion features with Transformer encoder layers, a residual head, and a learned residual gate.

Missing observations are simulated only in the 8-step input history. The evaluated settings are `complete`, `random_0.1`, `random_0.3`, `random_0.5`, `contiguous`, and `partial`.

## Repository Layout

- `data/`: ETH/UCY data, trajectory dataset loader, and missingness utilities.
- `baseline_model/`: constant-velocity and vanilla LSTM baselines.
- `project_model/`: missingness-aware LSTM and Transformer models.
- `navigation/`: rule-based navigation decision logic.
- `utils/`: metrics, plotting, and trajectory helpers.
- `results/`: checkpoints, experiment CSV, and generated plots.
- `final_report/`: CVPR-format report, generated figures, and figure-generation script.
- `notebooks/`: Colab notebook for training and experiment runs.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the smoke test:

```bash
.venv/bin/python smoke_test.py
```

## Experiments

Run one model on one split:

```bash
.venv/bin/python experiment.py --model constant_velocity --split zara1 --missing-mode complete
```

Train and evaluate a missingness-aware model:

```bash
.venv/bin/python experiment.py \
  --model missing_transformer \
  --split zara1 \
  --missing-mode random \
  --drop-rate 0.3 \
  --mixed-missingness \
  --epochs 20 \
  --hidden-dim 128 \
  --dropout 0.05 \
  --lr 5e-4 \
  --teacher-forcing-decay \
  --loss huber \
  --residual-weight 0.01 \
  --feature-mode motion \
  --plot
```

Run the full evaluation table:

```bash
.venv/bin/python run_full_experiments.py \
  --splits zara1 zara2 eth hotel univ \
  --epochs 10 \
  --hidden-dim 128 \
  --dropout 0.05 \
  --lr 5e-4 \
  --teacher-forcing-decay \
  --loss huber \
  --feature-mode motion
```

The full run writes:

- `results/full_experiment_results.csv`
- representative navigation plots under `results/plots/`
- metrics including ADE, FDE, action accuracy, stop precision, and stop recall

## Report and Figures

Generate report-ready figures from the saved CSV:

```bash
.venv/bin/python final_report/generate_report_figures.py
```

Compile the CVPR-format report:

```bash
pdflatex -interaction=nonstopmode -jobname=final_report_cvpr_full -output-directory=final_report final_report/final_report_cvpr_full.tex
pdflatex -interaction=nonstopmode -jobname=final_report_cvpr_full -output-directory=final_report final_report/final_report_cvpr_full.tex
```

Main report output:

- `final_report/final_report_cvpr_full.pdf`

## Demos

### Virtual Road Demo

The most stable visual demo is the virtual road-driving interface. It uses scripted pedestrian observations, runs the trained trajectory model, and converts the predicted path into high-level navigation behavior.

```bash
.venv/bin/python demo_virtual_navigation.py
```

### Grid Navigation Demo

The grid demo replays a real ETH/UCY trajectory window, shows observed and missing points, draws the predicted and true futures, and displays the selected navigation action.

```bash
.venv/bin/python demo_grid_navigation.py --split zara1 --missing-mode random --drop-rate 0.3
```

For a headless saved image:

```bash
.venv/bin/python demo_grid_navigation.py \
  --split zara1 \
  --sample-index 0 \
  --missing-mode random \
  --drop-rate 0.3 \
  --headless \
  --save-path results/plots/grid_demo.png
```

### Webcam Demo

The webcam demo is a qualitative bridge from live image observations to trajectory-based navigation actions. It tracks one moving person or object, builds a short 2D history, predicts future points, and displays the selected action.

```bash
.venv/bin/python demo_webcam_navigation.py --mirror
```

Use a specific checkpoint:

```bash
.venv/bin/python demo_webcam_navigation.py \
  --checkpoint results/colab_zara1_missing_transformer_best.pt \
  --model missing_transformer \
  --mirror
```

Press `q` to quit the camera window.

## Notes

- The full experiment uses five scene splits: `zara1`, `zara2`, `eth`, `hotel`, and `univ`.
- The missingness-aware models are trained with mixed missingness: `random_0.1`, `random_0.3`, `random_0.5`, and `partial`.
- The vanilla LSTM does not receive masks, gap features, velocity, or acceleration.
- The demos are intended for qualitative navigation visualization, not calibrated physical-robot benchmarking.
