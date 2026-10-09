# Real-Time Human-Aware Robot Obstacle Avoidance in Indoor Corridors

This repository contains the implementation of a real-time human-aware robot navigation system for indoor corridors. The system integrates 2D LiDAR-based pedestrian detection, multi-target tracking, motion prediction, DWA--VO planning, and IRL-based local decision-making.

## System Overview

The system follows a perception--tracking--prediction--planning pipeline:

1. 2D LiDAR scan segmentation
2. AdaBoost-based pedestrian leg detection
3. Pedestrian center estimation
4. Hungarian data association
5. EKF-based pedestrian tracking and prediction
6. Model-based DWA--VO planning
7. Learning-based IRL local action selection

DWA--VO and IRL are implemented as two separate navigation approaches. Both use pedestrian state information, but they are evaluated independently.

## Experiment Video

The real-world experiment video is available on YouTube:

[![Experiment Video](https://img.youtube.com/vi/zpb_crr0sds/hqdefault.jpg)](https://youtu.be/zpb_crr0sds)

Video link: https://youtu.be/zpb_crr0sds

## Repository Structure

```text
adaboost_leg/
  train data/                 LiDAR training data
  test data/                  LiDAR testing data
  最終模型和時時偵測/          AdaBoost model and real-time detector
  工具/                       Data labeling and conversion tools

ekf+匈牙利/
  ekf_line.py                 EKF tracking with Hungarian association
  ellipse_predict.m           Prediction visualization

irl/
  track_people_state.py       Pedestrian state processing
  14features_4actions/        Original 14-feature IRL model
  18fea_4act/                 Updated 18-feature IRL model and training files

dwa_vo_4_success_rviz.py      DWA--VO planner with RViz visualization