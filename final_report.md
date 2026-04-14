# Missingness-Aware Pedestrian Trajectory Prediction for Robot Navigation

## Abstract

Reliable robot navigation around pedestrians requires anticipating future human motion from imperfect visual observations. In real camera pipelines, pedestrian detections may be missing because of occlusion, blur, crowding, or detector failure. This project studies pedestrian trajectory prediction under incomplete observations and connects the predicted trajectories to an interpretable robot navigation policy. We formulate the task as observing 8 past two-dimensional pedestrian positions and predicting 12 future positions on ETH/UCY-style scenes. We compare a constant-velocity baseline, a vanilla LSTM encoder-decoder, a missingness-aware LSTM, and a missingness-aware Transformer under complete, random, contiguous, and partial observation dropout. The missingness-aware models explicitly encode observation masks, missing-gap features, and motion features, allowing them to remain robust when visual evidence is sparse. Across five scene splits, missingness-aware models improve robustness in harder missingness settings, with the missingness-aware Transformer achieving the best average ADE under 50 percent random dropout and strong results on Zara scenes. We further demonstrate how trajectory prediction can drive high-level navigation actions in live, dataset replay, and virtual road-driving demos.

## 1. Introduction

Pedestrian-aware navigation is a core problem in mobile robotics and intelligent transportation. A robot cannot safely move through a shared space by reacting only to the current pedestrian location; it must reason about where the pedestrian is likely to be in the near future. This makes trajectory prediction a natural bridge between computer vision perception and downstream navigation.

In a real computer vision system, however, the input trajectory is rarely complete. Pedestrian detectors may fail for several frames when a person is occluded, partially outside the frame, or moving through a crowded region. A model trained only on complete tracks can treat these missing observations as ordinary coordinates, leading to unstable forecasts and unsafe navigation commands.

This project focuses on missingness-aware trajectory prediction. Given a sequence of observed positions, we predict the future pedestrian path and map that predicted path to a simple robot action: `stop`, `go`, `turn_left`, or `turn_right`. The central question is:

> How can a trajectory model remain useful for robot navigation when the visual observation history is incomplete?

Our contributions are:

1. We build a fixed-window trajectory preprocessing pipeline from raw ETH/UCY-style pedestrian annotations.
2. We simulate realistic observation missingness patterns and evaluate robustness under each condition.
3. We implement missingness-aware LSTM and Transformer predictors that consume masks, gap features, and motion features.
4. We connect predicted pedestrian futures to rule-based robot navigation decisions.
5. We provide a stable virtual road-driving demo where scripted pedestrian observations are fed into the trained model and the red predicted trajectory controls a vehicle-like robot.

## 2. Data Preprocessing

We converted raw scene-level pedestrian trajectory annotations into fixed-length learning samples ourselves. Each raw scene contains pedestrian identities, frame or time indices, and two-dimensional world coordinates. Our preprocessing converts these variable-length tracks into uniform tensors suitable for neural prediction.

The preprocessing pipeline is:

1. **Parse raw trajectory annotations.** We read the pedestrian records for each scene and extract pedestrian ID, frame/time, and 2D position.
2. **Group by pedestrian identity.** All detections belonging to the same pedestrian are collected into a single track.
3. **Sort each track temporally.** Within each pedestrian track, positions are ordered by frame/time so that motion history is physically meaningful.
4. **Filter short tracks.** Tracks shorter than the required window length are skipped. In this project the window length is `obs_len + pred_len = 8 + 12 = 20`.
5. **Build sliding windows.** For each sufficiently long track, we slide a 20-step window through the sequence. The first 8 points form the observed trajectory and the next 12 points form the future target.
6. **Split by scene and phase.** Windows are assigned to train, validation, or test phases using scene-level splits. We evaluate on `zara1`, `zara2`, `eth`, `hotel`, and `univ`.
7. **Represent observations as tensors.** Each observation window has shape `[8, 2]`; each future target has shape `[12, 2]`.
8. **Generate missingness masks.** For each evaluation condition, we create a binary observation mask where 1 means observed and 0 means missing.
9. **Carry forward valid observations.** Missing coordinates are filled by carrying forward the most recent valid observation. This preserves tensor shape while preventing missing frames from becoming arbitrary zeros.
10. **Compute missingness-aware motion features.** We augment the filled coordinates with relative displacement, velocity, acceleration, the observation mask, and a normalized missing-gap feature that measures time since the last valid observation.

After preprocessing, the fixed-window samples are stored locally for fast loading during training and evaluation. The important point is that the learning examples and missingness conditions are constructed by our pipeline, rather than treating the dataset as already prepared for this robustness study.

## 3. Problem Formulation

Let the observed pedestrian trajectory be

```text
X_obs = [x_1, x_2, ..., x_8],     x_t in R^2
```

and the future target be

```text
Y = [y_1, y_2, ..., y_12],        y_t in R^2.
```

The model predicts

```text
Y_hat = f(X_obs, M),
```

where `M` is an observation mask. In complete settings, all entries in `M` are 1. Under missingness, some observed positions are hidden before prediction. We evaluate the predicted future with average displacement error (ADE) and final displacement error (FDE):

```text
ADE = mean_t ||Y_hat_t - Y_t||_2
FDE = ||Y_hat_12 - Y_12||_2.
```

For navigation, each predicted trajectory is passed to a rule-based decision function. If predicted points enter the front safety zone, the robot stops. If the predicted path passes close to the left or right, the robot turns away. Otherwise it continues forward.

## 4. Missingness Conditions

We evaluate six observation settings:

| Condition | Description |
|---|---|
| `complete` | All 8 observed positions are available. |
| `random_0.1` | Each non-initial observation is randomly dropped with probability 0.1. |
| `random_0.3` | Random observation dropout with probability 0.3. |
| `random_0.5` | Random observation dropout with probability 0.5. |
| `contiguous` | A contiguous block of observations is removed. |
| `partial` | The last part of the observation window is hidden, simulating a target that disappears before prediction. |

The first observation is kept in random missingness settings to avoid completely unanchored trajectories.

## 5. Method

### 5.1 Constant-Velocity Baseline

The constant-velocity model estimates velocity from the last two observed positions and extrapolates it for 12 future steps. When observations are missing, it uses the last valid positions according to the mask. This baseline is strong on many ETH/UCY trajectories because pedestrian motion is often locally smooth.

### 5.2 Vanilla LSTM Encoder-Decoder

The vanilla LSTM reads the 8 observed coordinates and recursively decodes 12 future coordinates. It does not explicitly receive missingness masks or gap information. This makes it a useful test of whether a standard sequence model can handle incomplete observations without missingness-specific features.

### 5.3 Missingness-Aware LSTM

The missingness-aware LSTM extends the encoder input with missingness features. It predicts a residual on top of a constant-velocity forecast:

```text
Y_hat = Y_cv + residual_LSTM(features).
```

This structure keeps the strong local-motion prior of constant velocity while allowing the neural model to correct it when the observed history indicates curvature, stopping, or interaction.

### 5.4 Missingness-Aware Transformer

The missingness-aware Transformer uses a compact Transformer encoder over the 8-step observation window. Its input features include motion and missingness information. The model predicts a residual and a learned gate:

```text
Y_hat = Y_cv + gate * residual_Transformer(features).
```

The gate is initialized conservatively, encouraging the model to stay close to the constant-velocity prior early in training and learn corrections when useful.

### 5.5 Navigation Rule

Predicted future points are interpreted in robot-local coordinates. The navigation rule checks whether the prediction enters a rectangular front safety zone. If so, the robot action is `stop`. If the predicted path is near but to one side, the action is `turn_left` or `turn_right`; otherwise the action is `go`.

This rule is intentionally simple and interpretable. The project focus is not end-to-end control, but whether predicted pedestrian futures can support readable navigation decisions.

## 6. Experiments

### 6.1 Dataset Splits

We evaluate on five scene splits:

```text
zara1, zara2, eth, hotel, univ
```

For each split, we train and evaluate the four model families under the six missingness settings. The final result table contains 120 rows:

```text
5 splits x 4 models x 6 missingness conditions = 120 evaluations.
```

### 6.2 Training Details

Neural models are trained with teacher forcing and a decaying teacher-forcing ratio. Missingness-aware models are trained with mixed missingness conditions, so they learn to use masks and gap features rather than assuming complete observation histories. Evaluation reports ADE, FDE, navigation action accuracy, stop precision, and stop recall.

### 6.3 Metrics

We report:

| Metric | Meaning |
|---|---|
| ADE | Mean displacement error over all 12 predicted future steps. Lower is better. |
| FDE | Displacement error at the final predicted step. Lower is better. |
| Action accuracy | Agreement between the action from the predicted trajectory and the action from the true future trajectory. |
| Stop precision | Fraction of predicted `stop` actions that match true `stop` cases. |
| Stop recall | Fraction of true `stop` cases recovered by the prediction-driven rule. |

## 7. Results

The following table reports average ADE/FDE across all five splits. Each cell is `ADE / FDE`.

| Missingness | Constant Velocity | Vanilla LSTM | Missing LSTM | Missing Transformer |
|---|---:|---:|---:|---:|
| complete | 0.534 / 1.148 | 1.846 / 2.499 | 0.548 / 1.147 | 0.554 / 1.186 |
| random_0.1 | 0.547 / 1.150 | 1.879 / 2.538 | 0.560 / 1.158 | 0.561 / 1.191 |
| random_0.3 | 0.602 / 1.181 | 1.994 / 2.682 | 0.591 / 1.192 | 0.598 / 1.225 |
| random_0.5 | 0.733 / 1.297 | 2.181 / 2.924 | 0.703 / 1.314 | 0.698 / 1.343 |
| contiguous | 0.534 / 1.148 | 1.846 / 2.499 | 0.548 / 1.147 | 0.554 / 1.186 |
| partial | 0.873 / 1.350 | 2.551 / 3.489 | 0.735 / 1.345 | 0.749 / 1.400 |

The constant-velocity baseline is very competitive on complete and lightly missing data, which is expected because many pedestrian trajectories are locally linear over short horizons. However, as missingness becomes more severe, the missingness-aware models become more attractive. Under `random_0.5`, the missingness-aware Transformer obtains the best average ADE. Under `partial`, the missingness-aware LSTM produces the best average ADE and nearly matches constant velocity in FDE while substantially improving the trajectory average.

The vanilla LSTM performs poorly across all conditions. This suggests that a standard recurrent model without a strong motion prior or missingness encoding struggles to generalize on this relatively small and scene-dependent trajectory dataset.

### 7.1 Navigation Action Accuracy

The next table reports average action accuracy across splits.

| Missingness | Constant Velocity | Vanilla LSTM | Missing LSTM | Missing Transformer |
|---|---:|---:|---:|---:|
| complete | 0.936 | 0.767 | 0.936 | 0.931 |
| random_0.1 | 0.937 | 0.766 | 0.937 | 0.930 |
| random_0.3 | 0.935 | 0.763 | 0.934 | 0.930 |
| random_0.5 | 0.931 | 0.757 | 0.925 | 0.925 |
| contiguous | 0.936 | 0.767 | 0.936 | 0.931 |
| partial | 0.921 | 0.752 | 0.923 | 0.920 |

Action accuracy remains high for the constant-velocity and missingness-aware models because the navigation rule is coarse: many trajectories map to the same high-level action. This is useful for robot safety because small metric differences do not always change the control decision. The missingness-aware models are especially competitive in the partial condition, where the last observations disappear before prediction.

### 7.2 Split-Level Observations

Performance varies by scene. On `zara1` and `zara2`, the missingness-aware Transformer is consistently strong and gives the best ADE in complete, heavy random, and partial settings. On `hotel` and `univ`, constant velocity remains difficult to beat, likely because the local motion is smoother and less interactive. On `eth`, missingness-aware models are competitive, with the Transformer performing best in complete and partial settings.

This pattern suggests that the best predictor depends on the scene dynamics. A learned residual model is most useful when motion deviates from a simple linear extrapolation, while constant velocity remains a strong prior for smooth short-term motion.

## 8. Qualitative Navigation Demos

We implemented three demonstration modes.

### 8.1 Dataset Replay and Live Chessboard Demo

`demo_grid_navigation.py` displays a top-down grid view. In dataset replay mode, it visualizes observed history, missing frames, predicted future, and the navigation action. In live webcam mode, it can use motion detection to create a top-down tracked trajectory. The live mode is useful for showing the connection to visual perception, but raw camera tracking can be noisy.

### 8.2 Raw Webcam Debug Overlay

`demo_webcam_navigation.py` is a debug tool for raw camera input. It shows the camera view, motion detection, obstacle zones, and trajectory-based action decisions. It is not the main presentation demo because it depends heavily on lighting, camera placement, and background motion.

### 8.3 Virtual Road Navigation Demo

`demo_virtual_navigation.py` is the stable class demonstration. It uses a scripted virtual pedestrian to simulate the output of a vision detector. The blue trajectory is the observed pedestrian motion. The green future path is the scripted ground-truth future. The red future path is the trained model's prediction from the latest 8 observed points.

Thus, the virtual pedestrian motion is synthetic, but the red prediction is real model output. This separation makes the demo repeatable while still demonstrating the learned prediction model and its effect on navigation.

The newest version of the virtual demo includes a multi-lane road and an intersection. The robot follows lanes, changes lanes when needed, turns left or right only at the intersection, stops for predicted pedestrian risk, and resumes its route once the path is safe. This makes the robot behavior more realistic than free-space diagonal steering while keeping the prediction and navigation logic interpretable.

## 9. Discussion

The experiments show three important lessons.

First, simple motion priors are powerful. Constant velocity performs surprisingly well because the prediction horizon is short and many pedestrians move smoothly. This makes it a strong baseline that any learned model must justify.

Second, missingness-aware features are necessary for robust learning. The vanilla LSTM sees only coordinates and has no direct way to distinguish real stationary behavior from carried-forward missing detections. By contrast, missingness-aware models receive masks and gap features, making this distinction explicit.

Third, trajectory metrics and navigation metrics answer different questions. ADE and FDE measure geometric forecast quality, while action accuracy measures whether the prediction leads to the same high-level decision as the ground truth. A model can have small trajectory errors that do not affect the navigation action, or larger errors that matter if they cross the safety zone boundary.

## 10. Limitations

This project has several limitations.

1. The virtual demo uses scripted pedestrians rather than real camera detections. It demonstrates the prediction and navigation logic, but not end-to-end camera perception.
2. The navigation rule is hand-designed and conservative. It is interpretable but not a full motion planner.
3. The model predicts one main pedestrian at a time. Multi-agent social interaction is not explicitly modeled.
4. Constant velocity remains strong on smooth scenes, indicating that the learned models could benefit from more social context, map context, or multi-modal prediction.
5. Stop events are relatively rare in some splits, making stop recall harder to optimize and interpret.

## 11. Conclusion

We presented a missingness-aware pedestrian trajectory prediction system for robot navigation. Starting from raw scene-level pedestrian tracks, we built fixed-length observation and prediction windows, simulated realistic missing observation patterns, and evaluated four prediction models across five ETH/UCY-style scene splits. Missingness-aware LSTM and Transformer models improve robustness in challenging missingness settings by explicitly encoding observation masks, missing-gap timing, and motion features. Finally, we connected the predicted trajectories to interpretable robot navigation actions and built a stable virtual road-driving demo that visualizes observed motion, model predictions, ground-truth future motion, and robot decisions.

The project demonstrates a practical computer vision pipeline: visual pedestrian observations are converted into trajectories, missing detections are handled explicitly, future motion is predicted, and the prediction is used to support safe robot navigation.

## References

1. ETH/UCY pedestrian trajectory benchmark scenes.
2. Constant velocity trajectory forecasting baseline.
3. LSTM encoder-decoder sequence prediction.
4. Transformer encoder sequence modeling.

## Appendix: Reproducibility Notes

Main experiment command:

```bash
python3 run_full_experiments.py
```

Representative virtual demo commands:

```bash
.venv/bin/python demo_virtual_navigation.py --scenario crossing_left
.venv/bin/python demo_virtual_navigation.py --goal-x -5.0 --goal-y 3.0
.venv/bin/python demo_virtual_navigation.py --goal-x 5.0 --goal-y 3.0
```

Result source:

```text
results/full_experiment_results.csv
```

Selected qualitative plots are stored in:

```text
results/plots/
```

