# Real-Time Human-Aware Robot Obstacle Avoidance in Indoor Corridors

This repository contains the implementation of a real-time human-aware robot navigation system for indoor corridor environments. The system integrates 2D LiDAR-based pedestrian detection, multi-target tracking, short-term motion prediction, model-based DWA--VO planning, and learning-based IRL local decision-making.

The goal of this project is to allow a mobile robot to move safely in narrow indoor corridors shared with pedestrians. Unlike static obstacle avoidance, pedestrian-aware navigation must consider human motion, future collision risk, and early avoidance behavior.

## System Overview

The system follows a perception--tracking--prediction--planning pipeline:

1. 2D LiDAR scan segmentation
2. Geometric feature extraction
3. AdaBoost-based pedestrian leg detection
4. Pedestrian center estimation
5. Hungarian data association
6. EKF-based pedestrian tracking and prediction
7. Model-based DWA--VO planning
8. Learning-based IRL local action selection

DWA--VO and IRL are implemented as two separate navigation approaches. Both use pedestrian state information from the same perception and tracking pipeline, but they are evaluated independently.

## Architecture

```text
2D LiDAR
   |
   v
Scan Segmentation
   |
   v
Geometric Feature Extraction
   |
   v
AdaBoost Leg Detection
   |
   v
Pedestrian Center Detection
   |
   v
Hungarian Association
   |
   v
EKF Tracking and Prediction
   |
   v
Pedestrian States
  /                \
 v                  v
DWA--VO             IRL
Model-Based         Learning-Based
Planner             Planner
 |                  |
 v                  v
Velocity Command    Velocity Command
```

## Methods

### LiDAR-Based Pedestrian Detection

Raw 2D LiDAR scans are segmented into candidate clusters. Each segment is represented using handcrafted geometric features, and an AdaBoost classifier is used to detect human leg-like segments.

Detected leg segments are paired to estimate pedestrian centers.

### Pedestrian Tracking and Prediction

The system uses Hungarian data association to match current detections with existing pedestrian tracks. An Extended Kalman Filter is then used to estimate pedestrian position, velocity, heading, and short-term future motion.

### DWA--VO Planner

The DWA--VO branch is a model-based local planner. It evaluates candidate robot velocity commands using:

- relative position
- relative velocity
- safety radius
- future collision risk
- approximate time-to-collision

This allows the robot to react to moving pedestrians more safely than a planner that only considers current obstacle positions.

### IRL Planner

The IRL branch is a learning-based local planner. It learns an action-cost function from expert action labels and selects the lowest-cost local action.

The action set includes:

- `slow_forward`
- `forward`
- `slight_left`
- `slight_right`

Two IRL versions are included:

- `14features_4actions/`: original 14-feature IRL model
- `18fea_4act/`: updated 18-feature IRL model with additional dynamic closest-approach features

The 18-feature model adds features related to dynamic pedestrian interaction, such as closest-approach distance, closing speed, and safety clearance cost.

## Experiment Video

The real-world experiment video is available on YouTube:

[https://youtu.be/zpb_crr0sds](https://youtu.be/zpb_crr0sds)

## Repository Structure

```text
adaboost_leg/
  train data/                 LiDAR training data
  test data/                  LiDAR testing data
  最終模型和時時偵測/          AdaBoost model and real-time detector
  工具/                       Data labeling and conversion tools

EKF+Hungarian/
  code/                       EKF tracking and prediction code
  KF data/                    Tracking experiment data and labels

DWA+VO/
  dwa_vo_success.py           Original DWA--VO planner
  dwa_vo_success_rviz.py      DWA--VO planner with RViz visualization
  dwa_vo_4_success_rviz.py    Updated DWA--VO planner with RViz visualization

irl/
  track_people_state.py       Pedestrian state processing
  14features_4actions/        Original 14-feature IRL model
  18fea_4act/                 Updated 18-feature IRL model and training files
```

## Experimental Results

The experiments compare built-in DWA, DWA--VO, and IRL navigation behavior in stationary and moving pedestrian corridor scenarios.

| Robot | Method | Scenario | Avoidance Start Distance | Minimum Distance | Result |
|---|---|---|---:|---:|---|
| TurtleBot3 | Built-in DWA | Stationary person | 0.77 m | 0.23 m | Success |
| TurtleBot3 | Built-in DWA | Moving person | Collision | Collision | Failed |
| TurtleBot3 | DWA--VO | Stationary person | 1.29 m | 0.67 m | Success |
| TurtleBot3 | DWA--VO | Moving person | 0.72 m | 0.17 m | Avoided, close separation |
| TurtleBot3 | IRL 14-feature | Stationary person | 1.88 m | 0.47 m | Early avoidance |
| TurtleBot3 | IRL 14-feature | Moving person | 1.64 m | 0.25 m | Early avoidance |
| TurtleBot3 | IRL 18-feature ROS2 | Stationary person | 1.23 m | 0.555 m | Preliminary |
| TurtleBot3 | IRL 18-feature ROS2 | Moving person | 1.02 m | 0.15 m | Avoided, small margin |
| MiniBot | DWA--VO | Stationary person | 1.79 m | 0.52 m | Success |
| MiniBot | DWA--VO | Moving person | 1.42 m | 0.35 m | Success |
| MiniBot | IRL 14-feature | Stationary person | 2.52 m | 0.30 m | Early avoidance |
| MiniBot | IRL 14-feature | Moving person | 2.16 m | 0.30 m | Early avoidance |

Overall, DWA--VO tends to provide stronger explicit collision-risk reasoning, while the 14-feature IRL planner tends to start avoidance earlier in the recorded trials. The updated 18-feature IRL model was tested in limited TurtleBot3 ROS 2 experiments and requires further validation.

## Requirements

The code in this repository was developed using a combination of Python, MATLAB, ROS, and ROS 2 tools.

Main requirements include:

- Python 3
- MATLAB
- ROS / ROS 2, depending on the selected module
- RViz for visualization
- 2D LiDAR
- TurtleBot3 or MiniBot platform for robot experiments

## Notes

This repository is organized as an experimental research code collection rather than a fully packaged ROS workspace. Some scripts may require path or topic-name adjustments before running on a different robot platform or ROS setup.

## Limitations

- The LiDAR leg-detection dataset is limited in size.
- Pedestrian detection may become unstable in more complex or crowded scenes.
- The DWA--VO cost function is hand-designed and may require retuning for different robots or corridors.
- The IRL planner depends strongly on expert-label quality.
- The updated 18-feature IRL model was only evaluated in limited TurtleBot3 ROS 2 trials.
- MiniBot evaluation of the updated 18-feature model was limited by hardware issues.
- Human comfort and social navigation behavior are considered indirectly, but not fully modeled.

## Future Work

Possible future improvements include:

- Expanding the LiDAR pedestrian dataset
- Improving multi-person tracking in crowded environments
- Adding camera or RGB-D sensing for longer-range pedestrian perception
- Tuning the 18-feature IRL model with more real-robot trials
- Combining DWA--VO safety reasoning with IRL-based early avoidance behavior
- Adding explicit social-comfort constraints such as personal space and passing distance
