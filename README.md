# Real-Time Human-Aware Robot Obstacle Avoidance in Indoor Corridors

This project implements a real-time human-aware obstacle avoidance system for indoor mobile robots operating in corridor environments. The system uses 2D LiDAR data to detect pedestrians, track their motion, predict short-term future positions, and generate safer robot velocity commands using a Dynamic Window Approach enhanced with Velocity Obstacle risk evaluation.

The main goal is to improve robot navigation safety in narrow indoor corridors, where pedestrians should be treated as dynamic agents rather than static obstacles. Compared with a conventional DWA planner that mainly reacts to current obstacle positions, this project considers pedestrian velocity and future collision risk so that the robot can avoid people earlier and more smoothly.

## Project Overview

Indoor service robots are increasingly used in campus buildings, offices, laboratories, hospitals, and public corridors. In these environments, pedestrians may walk toward the robot, cross in front of it, suddenly stop, or appear in narrow spaces with limited avoidance room.

This project integrates:

- LiDAR-based pedestrian leg detection
- AdaBoost classification with handcrafted geometric features
- Leg pairing and pedestrian center estimation
- Hungarian data association for multi-target tracking
- Extended Kalman Filter tracking with a CTRV motion model
- Short-term pedestrian motion prediction
- Dynamic Window Approach local planning
- Velocity Obstacle risk evaluation using relative velocity and Time-to-Collision

## System Pipeline

```text
2D LiDAR Scan
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
Leg Pairing and Pedestrian Center Estimation
      |
      v
Hungarian Data Association
      |
      v
EKF Pedestrian Tracking
      |
      v
Short-Term Motion Prediction
      |
      v
DWA Trajectory Sampling
      |
      v
Velocity Obstacle Risk Evaluation
      |
      v
Safe Robot Velocity Command
```

## Key Methods

### 1. LiDAR-Based Leg Detection

Raw 2D LiDAR scans are segmented into candidate clusters. Each segment is converted into a 10-dimensional handcrafted feature vector, including:

- point count
- standard deviation to centroid
- segment width
- circle fitting radius
- boundary standard deviation
- mean curvature
- mean angular difference
- minimum line fitting error
- maximum line fitting error
- RANSAC inlier ratio

An AdaBoost classifier is trained to classify each segment as either a human leg or a non-leg object.

### 2. Pedestrian Center Estimation

Since a pedestrian usually appears as two leg-like clusters in a 2D LiDAR scan, detected leg segments are paired using distance constraints. The pedestrian center is estimated as the midpoint between the two detected leg centers.

### 3. Multi-Target Tracking

The tracking module uses the Hungarian algorithm to associate current pedestrian detections with existing tracks. This helps maintain stable pedestrian identities across frames, even when detections are noisy or temporarily missing.

### 4. EKF Motion Estimation

Each pedestrian is tracked using an Extended Kalman Filter with a Constant Turn Rate and Velocity motion model. The tracked state is:

```text
[x, y, velocity, heading, turn_rate]
```

The EKF estimates pedestrian position, velocity, heading direction, and short-term future motion.

### 5. DWA-VO Local Planning

The planner extends the Dynamic Window Approach by adding a Velocity Obstacle cost term. For each candidate robot command, the system evaluates whether the relative velocity between the robot and a pedestrian may lead to a future collision.

The VO risk considers:

- relative position
- relative velocity
- combined safety radius
- collision cone
- Time-to-Collision

Commands with high future collision risk are penalized, allowing the robot to avoid moving pedestrians earlier.

## Repository Structure

This repository is organized as an experimental code collection rather than a fully packaged ROS catkin package.

```text
.
├── adaboost_leg/
│   ├── train data/
│   │   ├── train_data_60s_2p.mat
│   │   └── train_data_60s_2p＿label.mat
│   ├── test data/
│   │   ├── test_data_60s_2p.mat
│   │   └── test_data_60s_2p＿label.mat
│   ├── 工具/
│   │   ├── convert_lidar_dat_to_xy_mat.m
│   │   └── label.m
│   └── 最終模型和時時偵測/
│       ├── adaboost_leg.m
│       └── leg_detector.py
│
├── EKF+Hungarian/
│   ├── KF data/
│   │   ├── data/
│   │   │   ├── 10s_2p.dat
│   │   │   ├── 10s_2p_2.dat
│   │   │   ├── 20s_2p.mat
│   │   │   ├── 20s_2p_2.mat
│   │   │   └── kf*_test_*.mat
│   │   └── label/
│   │       └── *_label.mat
│   └── code/
│       ├── KO.py
│       ├── ekf_line.py
│       └── ellipse_predict.m
│
├── DWA+VO/
│   ├── dwa_vo_success.py
│   ├── dwa_vo_success_rviz.py
│   ├── dwa_vo_沒路時會轉頭.py
│   └── dwa_vo_沒路時會轉頭_rviz.py
│
└── README.md
```

## Folder Description

### `adaboost_leg`

Contains the LiDAR leg detection dataset, MATLAB tools, AdaBoost training code, and a real-time ROS leg detector.

Main files:

- `工具/convert_lidar_dat_to_xy_mat.m`: converts LiDAR `.dat` data into MATLAB `.mat` format.
- `工具/label.m`: labeling tool for LiDAR leg data.
- `最終模型和時時偵測/adaboost_leg.m`: MATLAB AdaBoost training and detection code.
- `最終模型和時時偵測/leg_detector.py`: Python ROS node for real-time leg detection and RViz visualization.

### `EKF+Hungarian`

Contains data and code for pedestrian tracking using Hungarian association and EKF prediction.

Main files:

- `code/ekf_line.py`: real-time leg detection, EKF tracking, and RViz visualization.
- `code/KO.py`: AdaBoost, EKF tracking, and collision risk visualization.
- `code/ellipse_predict.m`: MATLAB implementation for EKF prediction, uncertainty ellipse visualization, and collision probability estimation.
- `KF data/data/`: LiDAR and tracking data.
- `KF data/label/`: labeled data.

### `DWA+VO`

Contains the integrated obstacle avoidance planner combining leg detection, EKF tracking, DWA trajectory sampling, and VO-based risk evaluation.

Main files:

- `dwa_vo_success.py`: integrated DWA-VO planner.
- `dwa_vo_success_rviz.py`: DWA-VO planner with RViz visualization.
- `dwa_vo_沒路時會轉頭.py`: version with additional turning behavior when no safe path is available.
- `dwa_vo_沒路時會轉頭_rviz.py`: RViz visualization version of the turning behavior planner.

## Hardware and Software

### Robot Platforms

- TurtleBot3
- Minibot

### Sensors

- 2D LiDAR
- Robot odometry

### Software Environment

- Ubuntu 20.04
- ROS Noetic
- Gazebo
- Python 3
- MATLAB

## Experimental Results

### AdaBoost Leg Detection Dataset

The LiDAR leg detection dataset was collected in a corridor environment with two pedestrians.

| Item | Value |
|---|---:|
| Collected LiDAR duration | Approx. 2 minutes |
| Total frames | 594 |
| Segment samples | 925 |
| Feature dimension | 10 |
| Positive samples | 177 |
| Negative samples | 748 |

### AdaBoost 5-Fold Cross Validation

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Accuracy | 86.28% | 3.15% |
| Precision | 67.97% | 16.91% |
| Recall | 65.45% | 11.59% |
| F1-score | 64.06% | 8.15% |

The classifier can identify human leg-like LiDAR segments, but performance is affected by the small and imbalanced dataset.

### Obstacle Avoidance Results

The proposed DWA-VO planner was compared with the built-in DWA planner in static and moving pedestrian scenarios.

| Method | Robot | Target | Avoidance Start Distance | Minimum Distance | Result |
|---|---|---|---:|---:|---|
| Built-in DWA | TurtleBot3 | Static person | 0.77 m | 0.23 m | Success |
| Built-in DWA | TurtleBot3 | Moving person | Collision | Collision | Failure |
| DWA-VO | TurtleBot3 | Static person | 1.29 m | 0.67 m | Success |
| DWA-VO | TurtleBot3 | Moving person | 0.72 m | 0.17 m | Success |
| DWA-VO | Minibot | Static person | 1.79 m | 0.52 m | Success |
| DWA-VO | Minibot | Moving person | 1.42 m | 0.35 m | Success |

The built-in DWA planner successfully avoided a static pedestrian but failed in the moving pedestrian case. The proposed DWA-VO method avoided both static and moving pedestrians by considering future collision risk from relative velocity and Time-to-Collision.

## Main Contributions

- Built a complete LiDAR-based pedestrian perception pipeline.
- Trained an AdaBoost classifier for human leg detection using handcrafted geometric features.
- Implemented pedestrian tracking with Hungarian data association and EKF.
- Integrated short-term pedestrian motion prediction into local planning.
- Extended DWA with Velocity Obstacle risk evaluation.
- Tested the system on TurtleBot3 and Minibot platforms.
- Demonstrated safer avoidance behavior compared with the built-in DWA planner.

## Limitations

- The LiDAR-based detection range is limited.
- The training dataset is relatively small and imbalanced.
- The current system mainly uses short-term pedestrian prediction.
- The tracking experiments focus on a small number of pedestrians.
- Corridor environments provide limited lateral space for avoidance.
- Human comfort and social navigation behavior are not explicitly modeled yet.

## Future Work

Possible future improvements include:

- Expanding the leg detection dataset.
- Adding camera or RGB-D sensing for longer-range pedestrian perception.
- Improving tracking in crowded multi-person environments.
- Adding social comfort constraints such as personal space and passing distance.
- Improving pedestrian motion prediction with social-force or learning-based models.
- Testing the system in more complex indoor environments beyond corridors.

## Authors

Yung-Ching Ko  
Yu-Ting Tseng  

Department of Mathematics  
National Central University

## Acknowledgement

This project was developed as part of an Introduction to Data Science final project at National Central University. The work combines supervised learning, data association, probabilistic tracking, and robot motion planning to solve a practical human-aware navigation problem.

## References

- K. O. Arras, O. M. Mozos, and W. Burgard, "Using boosted features for the detection of people in 2D range data," ICRA, 2007.
- D. Fox, W. Burgard, and S. Thrun, "The Dynamic Window Approach to Collision Avoidance," IEEE Robotics & Automation Magazine, 1997.
- P. Fiorini and Z. Shiller, "Motion Planning in Dynamic Environments Using Velocity Obstacles," IJRR, 1998.
- H. W. Kuhn, "The Hungarian Method for the Assignment Problem," Naval Research Logistics Quarterly, 1955.
- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," Journal of Basic Engineering, 1960.

## License

This repository currently does not specify an open-source license. Add a `LICENSE` file before public reuse or distribution.
