#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time person position detection + EKF tracking + machine-readable output.

功能：
1. LiDAR segmentation
2. 10 handcrafted features
3. MATLAB exported AdaBoost leg classifier
4. Pair two detected legs into one person
5. EKF tracking / 1-second prediction
6. RViz visualization
7. Publish /people_states as JSON string for IRL data collection

Subscribed:
    /scan

Published:
    /people_markers      visualization_msgs/MarkerArray
    /people_states       std_msgs/String

/people_states JSON format:
{
  "stamp": 123.456,
  "frame_id": "base_scan",
  "predict_time": 1.0,
  "people": [
    {
      "id": 1,
      "x": 2.1,
      "y": 0.3,
      "v": 0.4,
      "theta": 0.1,
      "omega": 0.0,
      "vx": 0.398,
      "vy": 0.039,
      "pred_x_1s": 2.50,
      "pred_y_1s": 0.34,
      "sigma": 0.12,
      "age": 8,
      "miss": 0
    }
  ]
}
"""

import math
import json
import numpy as np
import rospy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# =========================================================
# 1. AdaBoost 模型參數（從 MATLAB 匯出）
# =========================================================
ALPHA_LEG = np.array([
    0.783044, 0.347437, 0.320527, 0.326356, 0.369778,
    0.316612, 0.249111, 0.228206, 0.240536, 0.208360,
    0.241859, 0.207806, 0.165593, 0.167728, 0.152153,
    0.169358, 0.177836, 0.149083, 0.167618, 0.159571,
    0.161077, 0.158761, 0.158964, 0.140098, 0.122884,
    0.141936, 0.124683, 0.118274, 0.119120, 0.109797,
    0.135280, 0.108972, 0.111787, 0.104844, 0.141188,
    0.122933, 0.114286, 0.096909, 0.090066, 0.101552,
    0.118719, 0.104412, 0.108325, 0.098659, 0.093576,
    0.103070, 0.084637, 0.091455, 0.094610, 0.086825
], dtype=np.float64)

# [feature_index, theta, s]
# Python 0-based index
STUMPS_LEG = [
    [5, 28.342846, 1], [1, 0.061183, -1], [2, 0.099685, 1],
    [2, 0.056246, -1], [1, 0.083119, -1], [4, 0.020911, 1],
    [0, 3.500000, 1], [1, 0.083119, -1], [6, 2.920823, 1],
    [3, 0.180842, -1], [8, 0.035349, 1], [4, 0.008846, -1],
    [2, 0.099685, 1], [3, 0.095347, -1], [6, 2.920823, 1],
    [3, 0.319987, -1], [8, 0.035349, 1], [1, 0.104475, -1],
    [2, 0.048098, -1], [2, 0.082441, 1], [2, 0.062081, -1],
    [1, 0.104475, -1], [0, 6.500000, 1], [5, 27.298656, -1],
    [1, 0.020262, -1], [1, 0.065549, -1], [8, 0.007097, 1],
    [4, 0.005266, -1], [1, 0.058460, 1], [1, 0.044350, -1],
    [2, 0.082441, 1], [2, 0.088983, -1], [8, 0.007097, 1],
    [6, 2.920823, 1], [1, 0.083119, -1], [8, 0.035349, 1],
    [1, 0.062675, -1], [6, 2.495484, 1], [7, 0.000846, -1],
    [8, 0.017176, 1], [5, 27.298656, -1], [2, 0.067109, -1],
    [2, 0.082441, 1], [2, 0.088983, -1], [2, 0.087673, 1],
    [1, 0.044350, -1], [2, 0.099685, 1], [2, 0.088983, -1],
    [2, 0.082441, 1], [1, 0.020262, -1]
]

THR_LEG = -0.428649

FEATURE_NAMES = [
    'point_count',
    'std_dev_to_centroid',
    'segment_width',
    'circle_fit_radius',
    'boundary_std_dev',
    'mean_curvature',
    'mean_angular_difference',
    'min_line_fitting_error',
    'max_line_fitting_error',
    'ransac_inlier_ratio'
]


# =========================================================
# 2. Utility functions
# =========================================================
def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# =========================================================
# 3. LiDAR segmentation
# =========================================================
def segment_lidar(xy_points, threshold=0.1):
    if xy_points.shape[0] == 0:
        return []

    clusters = []
    current_cluster = [xy_points[0]]

    for i in range(1, xy_points.shape[0]):
        dist = np.linalg.norm(xy_points[i] - xy_points[i - 1])

        if dist < threshold:
            current_cluster.append(xy_points[i])
        else:
            clusters.append(np.array(current_cluster, dtype=np.float64))
            current_cluster = [xy_points[i]]

    clusters.append(np.array(current_cluster, dtype=np.float64))
    return clusters


# =========================================================
# 4. 10 維 handcrafted features
# =========================================================
def extract_10_features(pts):
    k = pts.shape[0]
    if k < 2:
        return None

    EPS = 1e-12
    x = pts[:, 0]
    y = pts[:, 1]

    point_count = float(k)

    mu = np.mean(pts, axis=0)
    diff_mu = pts - mu
    dist2_mu = np.sum(diff_mu ** 2, axis=1)
    std_dev_to_centroid = math.sqrt(np.sum(dist2_mu) / (k - 1)) if k > 1 else 0.0

    segment_width = float(np.linalg.norm(pts[-1] - pts[0]))

    circle_fit_radius = 0.0
    if k >= 3:
        A = np.column_stack((-2.0 * x, -2.0 * y, np.ones(k)))
        b = -(x ** 2 + y ** 2)

        try:
            theta = np.linalg.pinv(A) @ b
            xc, yc, c3 = theta[0], theta[1], theta[2]
            rc_sq = xc ** 2 + yc ** 2 - c3

            if np.isfinite(rc_sq) and rc_sq > 0:
                circle_fit_radius = float(math.sqrt(rc_sq))
        except np.linalg.LinAlgError:
            pass

    if k >= 2:
        step_vec = np.diff(pts, axis=0)
        step_dist = np.sqrt(np.sum(step_vec ** 2, axis=1))
    else:
        step_dist = np.array([], dtype=np.float64)

    if step_dist.size >= 2:
        boundary_std_dev = float(np.std(step_dist, ddof=1))
    else:
        boundary_std_dev = 0.0

    mean_curvature = 0.0
    if k >= 3:
        curvatures = []

        for t in range(1, k - 1):
            A_pt = pts[t - 1]
            B_pt = pts[t]
            C_pt = pts[t + 1]

            dAB = np.linalg.norm(B_pt - A_pt)
            dBC = np.linalg.norm(C_pt - B_pt)
            dAC = np.linalg.norm(C_pt - A_pt)

            area2 = abs(
                (B_pt[0] - A_pt[0]) * (C_pt[1] - A_pt[1]) -
                (B_pt[1] - A_pt[1]) * (C_pt[0] - A_pt[0])
            )
            area_tri = 0.5 * area2

            denom = dAB * dBC * dAC
            if denom > EPS:
                curvatures.append(4.0 * area_tri / denom)
            else:
                curvatures.append(0.0)

        mean_curvature = float(np.mean(curvatures))

    mean_angular_difference = 0.0
    if k >= 3:
        betas = []

        for t in range(1, k - 1):
            v1 = pts[t - 1] - pts[t]
            v2 = pts[t + 1] - pts[t]

            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 > EPS and n2 > EPS:
                cos_beta = np.dot(v1, v2) / (n1 * n2)
                cos_beta = np.clip(cos_beta, -1.0, 1.0)
                betas.append(math.acos(cos_beta))
            else:
                betas.append(0.0)

        mean_angular_difference = float(np.mean(betas))

    min_line_fitting_error = 0.0
    max_line_fitting_error = 0.0

    if k >= 2:
        centered = pts - mu
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)

        dir_vec = Vt[0]
        normal_vec = np.array([-dir_vec[1], dir_vec[0]], dtype=np.float64)
        normal_norm = np.linalg.norm(normal_vec)

        if normal_norm > EPS:
            normal_vec = normal_vec / normal_norm
            r_line = np.mean(pts @ normal_vec)
            line_err = np.abs(pts @ normal_vec - r_line)

            min_line_fitting_error = float(np.min(line_err))
            max_line_fitting_error = float(np.max(line_err))

    ransac_inlier_ratio = 0.0
    if k >= 2:
        best_inlier = 0
        dist_thr = 0.02
        max_iter = min(30, k * (k - 1) // 2)

        if max_iter > 0:
            for _ in range(max_iter):
                pair = np.random.choice(k, 2, replace=False)
                p1 = pts[pair[0]]
                p2 = pts[pair[1]]

                v = p2 - p1
                nv = np.linalg.norm(v)
                if nv < EPS:
                    continue

                d = np.abs(
                    (pts[:, 0] - p1[0]) * v[1] -
                    (pts[:, 1] - p1[1]) * v[0]
                ) / nv

                inlier_count = int(np.sum(d < dist_thr))
                if inlier_count > best_inlier:
                    best_inlier = inlier_count

            ransac_inlier_ratio = float(best_inlier / k)

    return np.array([
        point_count,
        std_dev_to_centroid,
        segment_width,
        circle_fit_radius,
        boundary_std_dev,
        mean_curvature,
        mean_angular_difference,
        min_line_fitting_error,
        max_line_fitting_error,
        ransac_inlier_ratio
    ], dtype=np.float64)


# =========================================================
# 5. AdaBoost 分數
# =========================================================
def stump_predict_value(x, theta, s):
    return 1.0 if s * (x - theta) >= 0 else -1.0


def adaboost_score_single(feats):
    score = 0.0
    for alpha, stump in zip(ALPHA_LEG, STUMPS_LEG):
        j, theta, s = stump
        f_idx = int(j)
        vote = stump_predict_value(feats[f_idx], theta, s)
        score += alpha * vote
    return score


# =========================================================
# 6. 腳配對成人
# =========================================================
def pair_legs_to_people(detected_legs, max_leg_distance=0.6):
    people = []
    used = set()

    for i in range(len(detected_legs)):
        if i in used:
            continue

        xi, yi = detected_legs[i]['cx'], detected_legs[i]['cy']

        best_j = -1
        best_dist = float('inf')

        for j in range(i + 1, len(detected_legs)):
            if j in used:
                continue

            xj, yj = detected_legs[j]['cx'], detected_legs[j]['cy']
            dist = math.hypot(xi - xj, yi - yj)

            if dist < best_dist and dist <= max_leg_distance:
                best_dist = dist
                best_j = j

        if best_j != -1:
            used.add(i)
            used.add(best_j)

            xj, yj = detected_legs[best_j]['cx'], detected_legs[best_j]['cy']

            people.append({
                'x': 0.5 * (xi + xj),
                'y': 0.5 * (yi + yj),
                'leg1_id': detected_legs[i]['id'],
                'leg2_id': detected_legs[best_j]['id'],
                'leg_distance': best_dist
            })

    return people


# =========================================================
# 7. 小規模 greedy assignment
# =========================================================
def min_weight_assignment(cost_matrix):
    n, m = cost_matrix.shape

    if n == 0 or m == 0:
        return []

    rows, cols = np.where(cost_matrix < 1e8)
    potential_pairs = sorted(zip(rows, cols), key=lambda x: cost_matrix[x[0], x[1]])

    matched_r = set()
    matched_c = set()
    final_pairs = []

    for r, c in potential_pairs:
        if r not in matched_r and c not in matched_c:
            final_pairs.append((r, c))
            matched_r.add(r)
            matched_c.add(c)

    return final_pairs


# =========================================================
# 8. EKF Track and Tracker, CTRV model
# =========================================================
class Track:
    def __init__(self, track_id, z):
        self.id = track_id

        # State: [x, y, v, theta, omega]
        self.X = np.array([z[0], z[1], 0.0, 0.0, 1e-4], dtype=np.float64)

        self.P = np.eye(5, dtype=np.float64) * 0.1
        self.age = 1
        self.miss = 0

    def predict(self, dt, Q):
        # 避免 dt 異常太大，造成 EKF 爆掉
        dt = max(1e-3, min(float(dt), 0.5))

        x, y, v, th, om = self.X

        if abs(om) < 1e-3:
            self.X[0] += v * math.cos(th) * dt
            self.X[1] += v * math.sin(th) * dt

            F = np.eye(5)
            F[0, 2] = math.cos(th) * dt
            F[0, 3] = -v * math.sin(th) * dt
            F[1, 2] = math.sin(th) * dt
            F[1, 3] = v * math.cos(th) * dt

        else:
            self.X[0] += (v / om) * (math.sin(th + om * dt) - math.sin(th))
            self.X[1] += (v / om) * (-math.cos(th + om * dt) + math.cos(th))
            self.X[3] += om * dt

            F = np.eye(5)

            F[0, 2] = (math.sin(th + om * dt) - math.sin(th)) / om
            F[0, 3] = (v / om) * (math.cos(th + om * dt) - math.cos(th))
            F[0, 4] = (
                (v * dt * math.cos(th + om * dt) / om)
                - (v * (math.sin(th + om * dt) - math.sin(th)) / (om ** 2))
            )

            F[1, 2] = (-math.cos(th + om * dt) + math.cos(th)) / om
            F[1, 3] = (v / om) * (math.sin(th + om * dt) - math.sin(th))
            F[1, 4] = (
                (v * dt * math.sin(th + om * dt) / om)
                - (v * (-math.cos(th + om * dt) + math.cos(th)) / (om ** 2))
            )

            F[3, 4] = dt

        self.P = F @ self.P @ F.T + Q
        self.X[3] = wrap_to_pi(self.X[3])

    def update(self, z, R):
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ], dtype=np.float64)

        S = H @ self.P @ H.T + R

        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return

        innovation = z - self.X[0:2]
        self.X += K @ innovation
        self.P = (np.eye(5) - K @ H) @ self.P
        self.X[3] = wrap_to_pi(self.X[3])


class EKFTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

        # Process noise / measurement noise，可依現場調整
        self.Q = np.diag([0.001, 0.001, 0.02, 0.02, 0.02])
        self.R = np.eye(2) * 0.05

        self.dist_gate = 0.8
        self.max_missed = 5

    def update_tracks(self, measurements, dt):
        # 1. Predict all tracks
        for t in self.tracks:
            t.predict(dt, self.Q)
            t.age += 1

        n, m = len(self.tracks), len(measurements)
        cost_matrix = np.full((n, m), 1e9, dtype=np.float64)

        # 2. Build cost matrix
        for i in range(n):
            for j in range(m):
                d = math.hypot(
                    self.tracks[i].X[0] - measurements[j][0],
                    self.tracks[i].X[1] - measurements[j][1]
                )

                if d < self.dist_gate:
                    cost_matrix[i, j] = d

        # 3. Assign detections to tracks
        pairs = min_weight_assignment(cost_matrix)

        matched_t = [p[0] for p in pairs]
        matched_m = [p[1] for p in pairs]

        # 4. Update matched tracks
        for i, j in pairs:
            self.tracks[i].update(measurements[j], self.R)
            self.tracks[i].miss = 0

        # 5. Increase miss count for unmatched tracks
        for i in range(n):
            if i not in matched_t:
                self.tracks[i].miss += 1

        # 6. Create new tracks for unmatched measurements
        for j in range(m):
            if j not in matched_m:
                self.tracks.append(Track(self.next_id, measurements[j]))
                self.next_id += 1

        # 7. Remove stale tracks
        self.tracks = [t for t in self.tracks if t.miss <= self.max_missed]

        return self.tracks


# =========================================================
# 9. ROS node
# =========================================================
class RealTimeLegDetector:
    def __init__(self):
        rospy.init_node('realtime_leg_detector', anonymous=True)

        # -----------------------------
        # ROS params
        # -----------------------------
        scan_topic = rospy.get_param('~scan_topic', '/scan')

        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.05)
        self.segment_threshold = rospy.get_param('~segment_threshold', 0.1)
        self.max_leg_distance = rospy.get_param('~max_leg_distance', 0.6)

        # EKF 預測時間，給 /people_states 和 RViz 用
        self.predict_time = rospy.get_param('~predict_time', 1.0)

        # track 剛建立時速度很不穩，至少 age >= min_track_age 才發布給 logger 用
        self.min_track_age = rospy.get_param('~min_track_age', 3)

        self.debug = rospy.get_param('~debug', False)

        self.tracker = EKFTracker()
        self.last_time = None

        # -----------------------------
        # Subscriber / Publisher
        # -----------------------------
        self.scan_sub = rospy.Subscriber(
            scan_topic,
            LaserScan,
            self.scan_callback,
            queue_size=1
        )

        self.marker_pub = rospy.Publisher(
            '/people_markers',
            MarkerArray,
            queue_size=1
        )

        # 新增：給 IRL logger 用的 machine-readable people states
        self.people_state_pub = rospy.Publisher(
            '/people_states',
            String,
            queue_size=1
        )

        rospy.loginfo("Real-time leg tracker is ready.")
        rospy.loginfo("Publishing RViz markers on /people_markers.")
        rospy.loginfo("Publishing JSON people states on /people_states.")
        rospy.loginfo("predict_time = %.2f sec, min_track_age = %d",
                      self.predict_time, self.min_track_age)

    # -----------------------------------------------------
    # Main scan callback
    # -----------------------------------------------------
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = (
            np.isfinite(ranges)
            & (ranges > self.min_range)
            & (ranges < self.max_range)
        )

        ranges = ranges[valid]
        angles = angles[valid]

        if ranges.size == 0:
            self.publish_delete_all(msg.header.frame_id)
            self.publish_people_states([], msg.header.frame_id, msg.header.stamp)
            return

        xy_points = np.column_stack((
            ranges * np.cos(angles),
            ranges * np.sin(angles)
        ))

        clusters = segment_lidar(xy_points, threshold=self.segment_threshold)

        # -----------------------------
        # AdaBoost 偵測腿部
        # -----------------------------
        detected_legs = []

        for i, pts in enumerate(clusters):
            feats = extract_10_features(pts)

            if feats is None:
                continue

            if not np.all(np.isfinite(feats)):
                continue

            if not np.any(feats != 0):
                continue

            score = adaboost_score_single(feats)

            if score > THR_LEG:
                detected_legs.append({
                    'id': i,
                    'cx': float(np.mean(pts[:, 0])),
                    'cy': float(np.mean(pts[:, 1])),
                    'score': float(score),
                    'point_count': int(pts.shape[0])
                })

        # -----------------------------
        # 腳配對成人
        # -----------------------------
        people_positions = pair_legs_to_people(
            detected_legs,
            max_leg_distance=self.max_leg_distance
        )

        # -----------------------------
        # dt
        # -----------------------------
        current_time = msg.header.stamp

        if self.last_time is None:
            dt = 0.1
        else:
            dt = (current_time - self.last_time).to_sec()

            if dt <= 0.0 or dt > 1.0:
                dt = 0.1

        self.last_time = current_time

        # -----------------------------
        # EKF tracking
        # -----------------------------
        measurements = [
            np.array([p['x'], p['y']], dtype=np.float64)
            for p in people_positions
        ]

        active_tracks = self.tracker.update_tracks(measurements, dt)

        # -----------------------------
        # Publish
        # -----------------------------
        self.publish_track_markers(active_tracks, msg.header.frame_id)
        self.publish_people_states(active_tracks, msg.header.frame_id, current_time)

        if self.debug:
            rospy.loginfo(
                "legs=%d, people_meas=%d, active_tracks=%d",
                len(detected_legs),
                len(people_positions),
                len(active_tracks)
            )

    # -----------------------------------------------------
    # Prediction utility
    # -----------------------------------------------------
    def predict_track_state(self, track, predict_time):
        """
        根據目前 EKF state 預測未來 predict_time 秒的位置。
        回傳：
            pred_x, pred_y
            vx, vy
            sigma
        """
        x, y, v, th, om = track.X

        if abs(om) < 1e-3:
            px = x + v * math.cos(th) * predict_time
            py = y + v * math.sin(th) * predict_time

            F_p = np.eye(5)
            F_p[0, 2] = math.cos(th) * predict_time
            F_p[0, 3] = -v * math.sin(th) * predict_time
            F_p[1, 2] = math.sin(th) * predict_time
            F_p[1, 3] = v * math.cos(th) * predict_time

        else:
            px = x + (v / om) * (math.sin(th + om * predict_time) - math.sin(th))
            py = y + (v / om) * (-math.cos(th + om * predict_time) + math.cos(th))

            F_p = np.eye(5)
            F_p[0, 2] = (math.sin(th + om * predict_time) - math.sin(th)) / om
            F_p[0, 3] = (v / om) * (math.cos(th + om * predict_time) - math.cos(th))
            F_p[1, 2] = (-math.cos(th + om * predict_time) + math.cos(th)) / om
            F_p[1, 3] = (v / om) * (math.sin(th + om * predict_time) - math.sin(th))
            F_p[3, 4] = predict_time

        P_pred = F_p @ track.P @ F_p.T + (self.tracker.Q * predict_time)

        sigma_r = math.sqrt(max(P_pred[0, 0], P_pred[1, 1]))
        sigma_r = clamp(sigma_r, 0.04, 0.30)

        vx = v * math.cos(th)
        vy = v * math.sin(th)

        return {
            'pred_x': float(px),
            'pred_y': float(py),
            'sigma': float(sigma_r),
            'vx': float(vx),
            'vy': float(vy)
        }

    # -----------------------------------------------------
    # Publish JSON people states
    # -----------------------------------------------------
    def publish_people_states(self, tracks, frame_id, stamp):
        """
        發布 /people_states 給 irl_demo_logger.py 使用。

        注意：
            這裡的人位置仍然是在 LiDAR / robot frame 下。
            logger 會再用 /odom 把它轉成起點 local frame。
        """
        people = []

        for t in tracks:
            # 避免剛出現的 track 太不穩
            if t.age < self.min_track_age:
                continue

            pred = self.predict_track_state(t, self.predict_time)

            x, y, v, th, om = t.X

            people.append({
                'id': int(t.id),

                # current estimated state in scan / robot frame
                'x': float(x),
                'y': float(y),
                'v': float(v),
                'theta': float(th),
                'omega': float(om),

                # velocity components in scan / robot frame
                'vx': pred['vx'],
                'vy': pred['vy'],

                # predicted position in scan / robot frame
                'pred_x_1s': pred['pred_x'],
                'pred_y_1s': pred['pred_y'],

                # uncertainty
                'sigma': pred['sigma'],

                # track quality
                'age': int(t.age),
                'miss': int(t.miss)
            })

        if hasattr(stamp, 'to_sec'):
            stamp_float = float(stamp.to_sec())
        else:
            stamp_float = rospy.Time.now().to_sec()

        payload = {
            'stamp': stamp_float,
            'frame_id': str(frame_id),
            'predict_time': float(self.predict_time),
            'people': people
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.people_state_pub.publish(msg)

    # -----------------------------------------------------
    # RViz marker helpers
    # -----------------------------------------------------
    def publish_delete_all(self, frame_id):
        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.action = Marker.DELETEALL

        arr.markers.append(m)
        self.marker_pub.publish(arr)

    def create_circle_points(self, cx, cy, r, num_points=30):
        points = []

        for i in range(num_points + 1):
            angle = 2.0 * math.pi * i / num_points

            p = Point()
            p.x = cx + r * math.cos(angle)
            p.y = cy + r * math.sin(angle)
            p.z = 0.05

            points.append(p)

        return points

    def publish_track_markers(self, tracks, frame_id):
        arr = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.action = Marker.DELETEALL
        arr.markers.append(delete_marker)

        marker_id = 0

        # Robot marker at LiDAR frame origin
        m_robot = Marker()
        m_robot.header.frame_id = frame_id
        m_robot.header.stamp = rospy.Time.now()
        m_robot.ns = "robot_base"
        m_robot.id = marker_id
        marker_id += 1
        m_robot.type = Marker.SPHERE
        m_robot.action = Marker.ADD
        m_robot.pose.position.x = 0.0
        m_robot.pose.position.y = 0.0
        m_robot.pose.position.z = 0.1
        m_robot.pose.orientation.w = 1.0
        m_robot.scale.x = 0.2
        m_robot.scale.y = 0.2
        m_robot.scale.z = 0.2
        m_robot.color.r = 1.0
        m_robot.color.g = 1.0
        m_robot.color.b = 1.0
        m_robot.color.a = 1.0
        arr.markers.append(m_robot)

        for t in tracks:
            pred = self.predict_track_state(t, self.predict_time)

            px = pred['pred_x']
            py = pred['pred_y']
            sigma_r = pred['sigma']

            color_r = (t.id * 0.2) % 1.0
            color_g = 1.0 - (t.id * 0.3) % 1.0
            color_b = (t.id * 0.5) % 1.0

            # Current person cylinder
            m_person = Marker()
            m_person.header.frame_id = frame_id
            m_person.header.stamp = rospy.Time.now()
            m_person.ns = "tracked_people"
            m_person.id = marker_id
            marker_id += 1
            m_person.type = Marker.CYLINDER
            m_person.action = Marker.ADD
            m_person.pose.position.x = float(t.X[0])
            m_person.pose.position.y = float(t.X[1])
            m_person.pose.position.z = 0.25
            m_person.pose.orientation.w = 1.0
            m_person.scale.x = 0.3
            m_person.scale.y = 0.3
            m_person.scale.z = 0.5
            m_person.color.r = color_r
            m_person.color.g = color_g
            m_person.color.b = color_b
            m_person.color.a = 0.8
            arr.markers.append(m_person)

            # Track ID text
            m_text = Marker()
            m_text.header.frame_id = frame_id
            m_text.header.stamp = rospy.Time.now()
            m_text.ns = "track_id"
            m_text.id = marker_id
            marker_id += 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD
            m_text.pose.position.x = float(t.X[0])
            m_text.pose.position.y = float(t.X[1])
            m_text.pose.position.z = 0.7
            m_text.pose.orientation.w = 1.0
            m_text.scale.z = 0.15
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = "ID:{} age:{} miss:{}".format(t.id, t.age, t.miss)
            arr.markers.append(m_text)

            # Prediction point
            m_pred_pt = Marker()
            m_pred_pt.header.frame_id = frame_id
            m_pred_pt.header.stamp = rospy.Time.now()
            m_pred_pt.ns = "prediction_point"
            m_pred_pt.id = marker_id
            marker_id += 1
            m_pred_pt.type = Marker.SPHERE
            m_pred_pt.action = Marker.ADD
            m_pred_pt.pose.position.x = px
            m_pred_pt.pose.position.y = py
            m_pred_pt.pose.position.z = 0.1
            m_pred_pt.pose.orientation.w = 1.0
            m_pred_pt.scale.x = 0.1
            m_pred_pt.scale.y = 0.1
            m_pred_pt.scale.z = 0.1
            m_pred_pt.color.r = color_r
            m_pred_pt.color.g = color_g
            m_pred_pt.color.b = color_b
            m_pred_pt.color.a = 0.5
            arr.markers.append(m_pred_pt)

            # Line from current to prediction
            m_line = Marker()
            m_line.header.frame_id = frame_id
            m_line.header.stamp = rospy.Time.now()
            m_line.ns = "current_to_prediction_line"
            m_line.id = marker_id
            marker_id += 1
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.pose.orientation.w = 1.0
            m_line.scale.x = 0.035
            m_line.color.r = color_r
            m_line.color.g = color_g
            m_line.color.b = color_b
            m_line.color.a = 0.9

            p_now = Point()
            p_now.x = float(t.X[0])
            p_now.y = float(t.X[1])
            p_now.z = 0.15

            p_pred = Point()
            p_pred.x = float(px)
            p_pred.y = float(py)
            p_pred.z = 0.15

            m_line.points.append(p_now)
            m_line.points.append(p_pred)
            arr.markers.append(m_line)

            # Sigma circle
            m_sigma = Marker()
            m_sigma.header.frame_id = frame_id
            m_sigma.header.stamp = rospy.Time.now()
            m_sigma.ns = "sigma_range"
            m_sigma.id = marker_id
            marker_id += 1
            m_sigma.type = Marker.LINE_STRIP
            m_sigma.action = Marker.ADD
            m_sigma.pose.orientation.w = 1.0
            m_sigma.scale.x = 0.03
            m_sigma.color.r = color_r
            m_sigma.color.g = color_g
            m_sigma.color.b = color_b
            m_sigma.color.a = 0.6
            m_sigma.points = self.create_circle_points(px, py, sigma_r)
            arr.markers.append(m_sigma)

        self.marker_pub.publish(arr)


# =========================================================
# 10. Main
# =========================================================
if __name__ == '__main__':
    try:
        detector = RealTimeLegDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
