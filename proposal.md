Final Project Proposal

Title: Missingness-Aware Trajectory Prediction for Simple Robot Navigation

1. Introduction and Motivation

Mobile robots often rely on visual observations of surrounding agents, such as pedestrians or moving objects, to make navigation decisions. In real-world environments, however, observations are often incomplete due to occlusion, sensor noise, dropped detections, or limited camera views. This makes trajectory prediction more difficult, and inaccurate predictions can lead to poor navigation behavior.

This project is inspired by two papers. The first paper, Uncovering the Missing Pattern: Unified Framework Towards Trajectory Imputation and Prediction, studies how missing observations affect trajectory prediction and proposes a framework that explicitly models missingness. The second paper, An Interactive Navigation Method with Effect-oriented Affordance, explores how motion understanding can support robot navigation in interactive environments.

Based on these ideas, our project aims to study whether missingness-aware trajectory prediction can improve robot decision-making under partial observation. Instead of building a full autonomous navigation system, we focus on a simpler but meaningful goal: using predicted trajectories to trigger basic robot actions such as stop, go, turn left, and turn right.

2. Problem Statement

The central question of this project is:

Can a missingness-aware trajectory prediction model provide more robust future motion estimates under partial observation, and can those predictions support simple navigation decisions for a mobile robot?

We focus on two related tasks:
	1.	Trajectory prediction under missing observations
	2.	Simple navigation decision support based on predicted motion

3. Proposed Approach

Our project will have two main components.

3.1 Trajectory Prediction Model

We will train a lightweight trajectory prediction model on a public trajectory dataset. Since our computational resources are limited, we plan to use Google Colab and focus on a model that is feasible to train within those constraints.

We will compare:
	•	a baseline trajectory predictor, such as an LSTM-based model that uses observed trajectories only
	•	a missingness-aware trajectory predictor, which incorporates information about which observations are missing, for example through masks or time-gap features

The key idea is that missing observations are not just noise to ignore. Instead, they may contain useful structural information about uncertainty and motion continuity.

3.2 Robot Navigation Demo

We will build a simple Raspberry Pi robot demo that uses predicted future motion to make navigation decisions. The robot will not perform full end-to-end autonomous navigation. Instead, it will receive high-level commands derived from the prediction model, such as:
	•	stop if a predicted trajectory enters a safety zone in front of the robot
	•	turn left or turn right if the predicted motion suggests a future conflict on one side
	•	go if the path is predicted to remain clear

This robot demo will serve as a proof of concept showing how robust motion prediction can support downstream navigation behavior.

4. Dataset and Experimental Setup

We plan to use a public trajectory dataset containing sequences of agent motion, such as pedestrian or multi-agent trajectories. Because we have limited computing resources, we prefer working directly with trajectory coordinates instead of full raw video.

To simulate realistic incomplete observations, we will introduce missingness into the trajectory data in several ways, such as:
	•	random point dropping
	•	contiguous missing segments
	•	partial observation windows that imitate occlusion

We will then evaluate model performance under different missingness levels.

5. Evaluation Plan

We plan to evaluate the project in two ways.

5.1 Prediction Performance

We will measure trajectory prediction quality using standard metrics such as:
	•	ADE (Average Displacement Error)
	•	FDE (Final Displacement Error)

We will compare the baseline model and the missingness-aware model under different missingness conditions.

5.2 Navigation Decision Demonstration

For the robot component, we will demonstrate several predefined scenarios in which predicted future motion leads to navigation actions. For example:
	•	a crossing trajectory causes the robot to stop
	•	a future conflict on the left causes the robot to turn right
	•	a clear predicted path allows the robot to move forward

This part is mainly a qualitative demonstration rather than a full benchmark.

6. Expected Contributions

The expected contributions of this project are:
	1.	A lightweight implementation of trajectory prediction under missing observations
	2.	An empirical comparison between a baseline model and a missingness-aware model
	3.	A simple decision pipeline that maps predicted trajectories to robot navigation actions
	4.	A Raspberry Pi robot demo illustrating the practical value of robust trajectory prediction

7. Scope and Feasibility

Because this is a course project and our computational resources are limited, we will keep the scope manageable.

We do not aim to:
	•	reproduce the full architecture of either paper
	•	perform large-scale training on raw video
	•	build a complete autonomous navigation stack
	•	run all perception and prediction fully onboard the robot

Instead, we aim to build a focused prototype that demonstrates the connection between missingness-aware trajectory prediction and robot navigation support.

This scope is realistic for Google Colab training and a course-level final project.

8. Timeline

A possible project timeline is:

Week 1
	•	finalize dataset
	•	preprocess trajectories
	•	simulate missingness conditions

Week 2
	•	implement and train baseline prediction model
	•	build missingness-aware model

Week 3
	•	evaluate models using ADE/FDE
	•	analyze results under different missingness levels

Week 4
	•	design rule-based navigation decision layer
	•	connect predictions to robot commands

Week 5
	•	build and record Raspberry Pi demo
	•	prepare final report and presentation

9. Conclusion

This project studies how trajectory prediction under incomplete observations can support robot navigation. Inspired by missingness-aware prediction and affordance-based navigation, we propose a lightweight system that combines robust future motion estimation with simple robot decision-making. By comparing a baseline model with a missingness-aware model and demonstrating the result on a Raspberry Pi robot, we hope to show that explicitly modeling missing observations can improve performance in practical navigation-related tasks.