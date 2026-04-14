# Missingness-Aware Trajectory Prediction

This project uses preprocessed ETH/UCY-style trajectory coordinates to compare
baseline trajectory predictors with missingness-aware proposed models.

## Folder Layout

- `data/`: raw/preprocessed datasets plus dataset loading and missingness utilities.
- `baseline_model/`: Baseline A constant velocity and Baseline B vanilla LSTM encoder-decoder.
- `project_model/`: proposed missingness-aware LSTM and Transformer models.
- `navigation/`: rule-based stop/go/turn decision demo.
- `utils/`: shared metrics and plotting helpers.
- `results/`: generated checkpoints and plots.
- `notebooks/`: Colab-ready notebook.

## Local Run

```bash
.venv/bin/python smoke_test.py
.venv/bin/python experiment.py --model constant_velocity --split zara1 --missing-mode complete
.venv/bin/python experiment.py --model vanilla_lstm --split zara1 --epochs 5 --teacher-forcing-ratio 0.5
.venv/bin/python experiment.py --model vanilla_lstm --split zara1 --missing-mode random --drop-rate 0.3 --epochs 5
.venv/bin/python experiment.py --model missing_lstm --split zara1 --missing-mode random --drop-rate 0.3 --mixed-missingness --epochs 20 --hidden-dim 128 --dropout 0.05 --lr 5e-4 --teacher-forcing-decay --loss huber --residual-weight 0.01 --feature-mode motion --plot
.venv/bin/python experiment.py --model missing_transformer --split zara1 --missing-mode random --drop-rate 0.3 --mixed-missingness --epochs 20 --hidden-dim 128 --dropout 0.05 --lr 5e-4 --teacher-forcing-decay --loss huber --residual-weight 0.01 --feature-mode motion --plot
.venv/bin/python run_full_experiments.py --splits zara1 zara2 --epochs 5 --teacher-forcing-decay
```

Baseline B is a project-specific vanilla LSTM encoder-decoder adapted from
[lkulowski/LSTM_encoder_decoder](https://github.com/lkulowski/LSTM_encoder_decoder):
an encoder consumes the observed sequence, a decoder starts from the final
observed coordinate, training can use teacher forcing, and evaluation predicts
recursively.

The proposed models are residual-over-constant-velocity models: they first
build a constant-velocity forecast from the observed points and missingness
mask, then learn corrections to that forecast. The Transformer variant uses a
learned gate on the residual correction so it can stay close to constant
velocity when the short-horizon motion is nearly linear. The optimized
`--feature-mode motion` setting feeds relative position, velocity,
acceleration, observation mask, and time-gap features to the proposed models
while still using absolute coordinates for the constant-velocity base.

To adjust training length, change `--epochs`. In Colab, change the `epochs=`
line in the config cell.

`run_full_experiments.py` writes `results/full_experiment_results.csv` and
representative navigation plots under `results/plots/`. For the full report,
run all splits with `--splits zara1 zara2 eth hotel univ`. The CSV includes
ADE/FDE plus navigation decision metrics such as `action_accuracy`,
`stop_precision`, and `stop_recall`.

## Laptop Webcam Demo

The webcam demo is a qualitative bridge from the coordinate model to a robot
navigation-style interface. It tracks one moving person/object in the laptop
camera, converts the center point into a short 2D trajectory, predicts the next
trajectory points, and displays a `stop`, `go`, `turn_left`, or `turn_right`
action.

```bash
.venv/bin/python demo_webcam_navigation.py --mirror
```

By default, the demo loads `results/colab_zara1_missing_transformer_best.pt`. If you
train a stronger checkpoint, pass it explicitly:

```bash
.venv/bin/python demo_webcam_navigation.py --checkpoint results/colab_zara1_missing_transformer_best.pt --model missing_transformer --mirror
```

If the tracker is too sensitive or not sensitive enough, adjust
`--min-contour-area`. Press `q` to quit the camera window. This demo uses image
coordinates for visualization, so treat it as a classroom navigation demo rather
than a calibrated physical robotics benchmark.

The same demo also includes a simple RGB-only obstacle-zone layer for immediate
obstacles in the lower camera view. To test only obstacle zones:

```bash
.venv/bin/python demo_webcam_navigation.py --mirror --disable-trajectory
```

To tune the obstacle detector, adjust the lower-image ROI and sensitivity:

```bash
.venv/bin/python demo_webcam_navigation.py --mirror --obstacle-roi-start 0.55 --obstacle-threshold 0.025 --obstacle-min-area 1200
```

## Chessboard Grid Demo

For a more reliable classroom demo, use the top-down chessboard visualization.
It replays a real ETH/UCY trajectory window, shows the robot/camera as a dot,
marks missing observations, draws the predicted and true futures, and displays
the navigation action.

```bash
.venv/bin/python demo_grid_navigation.py --split zara1 --missing-mode random --drop-rate 0.3
```

For a headless smoke test or report image:

```bash
.venv/bin/python demo_grid_navigation.py --split zara1 --sample-index 0 --missing-mode random --drop-rate 0.3 --headless --save-path results/plots/grid_demo.png
```

To use the webcam as the live sensor while showing the clean chessboard view:

```bash
.venv/bin/python demo_grid_navigation.py --live-webcam --camera-index 0 --mirror
```

Use `--no-show-camera-inset` if you want only the chessboard on screen.

If you need to recreate the local environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
