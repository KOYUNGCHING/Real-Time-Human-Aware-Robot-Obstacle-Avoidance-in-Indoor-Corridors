#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Velocity Obstacle / ORCA-inspired human-aware corridor navigation.

Main behavior:
1. Detect legs from LiDAR using segmentation + handcrafted features + AdaBoost.
2. Pair two legs into one pedestrian position.
3. Track pedestrians with EKF.
4. Use simplified Velocity Obstacle logic:
   - sample candidate robot velocities (v, w)
   - predict robot trajectory under each candidate velocity
   - predict pedestrian future positions using EKF
   - reject candidate velocities that will collide with predicted people
   - reject candidate velocities that will collide with walls / LiDAR obstacles
   - choose the safest velocity that still moves toward the goal
5. Move from initial odom pose to a goal 8.5 m ahead.
6. Publish RViz markers for people, predictions, chosen trajectory, and forbidden velocities.

Subscribed:
    /scan
    /odom

Published:
    /cmd_vel
    /people_markers
    /vo_markers
"""

# 中文說明：VO/ORCA 版在速度空間中挑選不會撞到牆與預測行人的速度。
# 減速繞人時保留 0.07 m/s 以上的前進速度，且最高不超過 0.10 m/s。

import math
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist


# =========================================================
# 0. Utility functions
# =========================================================
def wrap_to_pi(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


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


# =========================================================
# 2. LiDAR segmentation
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
# 3. 10 handcrafted features
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

    boundary_std_dev = float(np.std(step_dist, ddof=1)) if step_dist.size >= 2 else 0.0

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
# 4. AdaBoost score
# =========================================================
def stump_predict_value(x, theta, s):
    return 1.0 if s * (x - theta) >= 0 else -1.0


def adaboost_score_single(feats):
    score = 0.0

    for alpha, stump in zip(ALPHA_LEG, STUMPS_LEG):
        j, theta, s = stump
        vote = stump_predict_value(feats[int(j)], theta, s)
        score += alpha * vote

    return score


# =========================================================
# 5. Pair two legs into one person
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
# 6. Greedy assignment for small tracking problem
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
# 7. EKF Track and Tracker
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
        dt = clamp(dt, 0.01, 0.3)

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
                v * dt * math.cos(th + om * dt) / om
                - v * (math.sin(th + om * dt) - math.sin(th)) / (om ** 2)
            )

            F[1, 2] = (-math.cos(th + om * dt) + math.cos(th)) / om
            F[1, 3] = (v / om) * (math.sin(th + om * dt) - math.sin(th))
            F[1, 4] = (
                v * dt * math.sin(th + om * dt) / om
                - v * (-math.cos(th + om * dt) + math.cos(th)) / (om ** 2)
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
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X += K @ (z - self.X[0:2])
        self.P = (np.eye(5) - K @ H) @ self.P
        self.X[3] = wrap_to_pi(self.X[3])
        self.age += 1

    def predict_position(self, predict_time):
        x, y, v, th, om = self.X

        if abs(om) < 1e-3:
            px = x + v * math.cos(th) * predict_time
            py = y + v * math.sin(th) * predict_time
        else:
            px = x + (v / om) * (math.sin(th + om * predict_time) - math.sin(th))
            py = y + (v / om) * (-math.cos(th + om * predict_time) + math.cos(th))

        return np.array([px, py], dtype=np.float64)


class EKFTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

        self.Q = np.diag([0.002, 0.002, 0.03, 0.02, 0.02])
        self.R = np.eye(2) * 0.04

        self.dist_gate = 0.8
        self.max_missed = 5

    def update_tracks(self, measurements, dt):
        for t in self.tracks:
            t.predict(dt, self.Q)

        n = len(self.tracks)
        m = len(measurements)

        cost_matrix = np.full((n, m), 1e9, dtype=np.float64)

        for i in range(n):
            for j in range(m):
                d = math.hypot(
                    self.tracks[i].X[0] - measurements[j][0],
                    self.tracks[i].X[1] - measurements[j][1]
                )

                if d < self.dist_gate:
                    cost_matrix[i, j] = d

        pairs = min_weight_assignment(cost_matrix)

        matched_t = [p[0] for p in pairs]
        matched_m = [p[1] for p in pairs]

        for i, j in pairs:
            self.tracks[i].update(measurements[j], self.R)
            self.tracks[i].miss = 0

        for i in range(n):
            if i not in matched_t:
                self.tracks[i].miss += 1

        for j in range(m):
            if j not in matched_m:
                self.tracks.append(Track(self.next_id, measurements[j]))
                self.next_id += 1

        self.tracks = [t for t in self.tracks if t.miss <= self.max_missed]

        return self.tracks


# =========================================================
# 8. VO / ORCA-inspired Navigator
# =========================================================
class VelocityObstacleNavigator:
    def __init__(self):
        rospy.init_node('vo_human_aware_navigator', anonymous=True)

        # ---------------- Topics ----------------
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.odom_topic = rospy.get_param('~odom_topic', '/odom')
        self.cmd_topic = rospy.get_param('~cmd_topic', '/cmd_vel')

        # ---------------- Detection ----------------
        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.05)
        self.segment_threshold = rospy.get_param('~segment_threshold', 0.1)
        self.max_leg_distance = rospy.get_param('~max_leg_distance', 0.6)

        # ---------------- Goal ----------------
        self.goal_distance = rospy.get_param('~goal_distance', 8.5)
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 0.15)

        # ---------------- Robot ----------------
        self.robot_radius = rospy.get_param('~robot_radius', 0.18)
        self.wall_safety_margin = rospy.get_param('~wall_safety_margin', 0.16)
        self.obstacle_clearance = self.robot_radius + self.wall_safety_margin
        # VO 找不到完全安全速度時仍盡量慢速前進；只有前方非常貼近才緊急停車。
        self.emergency_stop_distance = rospy.get_param('~emergency_stop_distance', 0.18)

        # ---------------- Velocity sampling ----------------
        self.v_nominal = rospy.get_param('~v_nominal', 0.10)
        self.v_max = rospy.get_param('~v_max', 0.10)
        self.v_min = rospy.get_param('~v_min', 0.07)
        self.min_moving_speed = rospy.get_param('~min_moving_speed', 0.07)
        self.w_max = rospy.get_param('~w_max', 1.10)

        self.v_samples = rospy.get_param('~v_samples', 7)
        self.w_samples = rospy.get_param('~w_samples', 21)

        # ---------------- VO prediction ----------------
        self.time_horizon = rospy.get_param('~time_horizon', 3.0)
        self.sim_dt = rospy.get_param('~sim_dt', 0.15)

        self.person_radius_static = rospy.get_param('~person_radius_static', 0.38)
        self.person_radius_dynamic = rospy.get_param('~person_radius_dynamic', 0.48)
        self.prediction_margin = rospy.get_param('~prediction_margin', 0.14)
        self.early_avoid_margin = rospy.get_param('~early_avoid_margin', 0.18)

        # ---------------- Cost weights ----------------
        self.weight_goal = rospy.get_param('~weight_goal', 6.5)
        self.weight_heading = rospy.get_param('~weight_heading', 1.9)
        self.weight_clearance = rospy.get_param('~weight_clearance', 2.2)
        self.weight_smooth = rospy.get_param('~weight_smooth', 1.2)
        self.weight_speed = rospy.get_param('~weight_speed', 0.8)
        self.weight_ttc = rospy.get_param('~weight_ttc', 2.4)

        # ---------------- Runtime states ----------------
        self.tracker = EKFTracker()
        self.last_scan_time = None

        self.latest_scan_points = None
        self.latest_tracks = []
        self.latest_frame = "laser"

        self.odom_ready = False
        self.start_pose_set = False

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        self.start_x = 0.0
        self.start_y = 0.0
        self.start_yaw = 0.0

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.prev_v = 0.0
        self.prev_w = 0.0

        self.mode = "INIT"
        self.best_traj = []
        self.forbidden_velocity_points = []
        self.safe_velocity_points = []
        self.best_velocity_point = None

        # ---------------- ROS ----------------
        self.scan_sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=1)

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.people_marker_pub = rospy.Publisher('/people_markers', MarkerArray, queue_size=1)
        self.vo_marker_pub = rospy.Publisher('/vo_markers', MarkerArray, queue_size=1)

        self.control_timer = rospy.Timer(rospy.Duration(0.10), self.control_loop)

        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("VO / ORCA-inspired human-aware navigator is ready.")
        rospy.loginfo("Goal distance = %.2f m", self.goal_distance)

    # =====================================================
    # Odom callback
    # =====================================================
    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        self.robot_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        self.odom_ready = True

        if not self.start_pose_set:
            self.start_x = self.robot_x
            self.start_y = self.robot_y
            self.start_yaw = self.robot_yaw

            self.goal_x = self.start_x + self.goal_distance * math.cos(self.start_yaw)
            self.goal_y = self.start_y + self.goal_distance * math.sin(self.start_yaw)

            self.start_pose_set = True

            rospy.loginfo(
                "Start set. Start=(%.2f, %.2f, %.2f), Goal=(%.2f, %.2f)",
                self.start_x,
                self.start_y,
                self.start_yaw,
                self.goal_x,
                self.goal_y
            )

    # =====================================================
    # Scan callback
    # =====================================================
    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = (
            np.isfinite(ranges)
            & (ranges > self.min_range)
            & (ranges < self.max_range)
        )

        ranges_valid = ranges[valid]
        angles_valid = angles[valid]

        if ranges_valid.size == 0:
            self.latest_scan_points = None
            self.publish_delete_all_people(msg.header.frame_id)
            return

        xy_points = np.column_stack((
            ranges_valid * np.cos(angles_valid),
            ranges_valid * np.sin(angles_valid)
        ))

        self.latest_scan_points = xy_points
        self.latest_frame = msg.header.frame_id

        clusters = segment_lidar(xy_points, threshold=self.segment_threshold)

        detected_legs = []

        for i, pts in enumerate(clusters):
            feats = extract_10_features(pts)

            if feats is None:
                continue

            if not np.any(feats != 0):
                continue

            score = adaboost_score_single(feats)

            if score > THR_LEG:
                detected_legs.append({
                    'id': i,
                    'cx': float(np.mean(pts[:, 0])),
                    'cy': float(np.mean(pts[:, 1])),
                    'score': float(score)
                })

        people_positions = pair_legs_to_people(
            detected_legs,
            max_leg_distance=self.max_leg_distance
        )

        current_time = msg.header.stamp

        if self.last_scan_time is None:
            dt = 0.1
        else:
            dt = (current_time - self.last_scan_time).to_sec()
            dt = clamp(dt, 0.03, 0.3)

        self.last_scan_time = current_time

        measurements = [
            np.array([p['x'], p['y']], dtype=np.float64)
            for p in people_positions
        ]

        self.latest_tracks = self.tracker.update_tracks(measurements, dt)

        self.publish_people_markers(self.latest_tracks, msg.header.frame_id)

    # =====================================================
    # Main control
    # =====================================================
    def control_loop(self, event):
        if not self.odom_ready or not self.start_pose_set:
            self.publish_stop()
            return

        goal_local = self.get_goal_in_robot_frame()
        goal_dist = float(np.linalg.norm(goal_local))

        if goal_dist < self.goal_tolerance:
            self.mode = "GOAL_REACHED"
            self.publish_stop()
            self.publish_vo_markers()
            rospy.loginfo_throttle(1.0, "Goal reached.")
            return

        if self.latest_scan_points is None:
            self.mode = "NO_SCAN"
            self.publish_stop()
            return

        cmd = self.choose_velocity_by_vo(
            goal_local=goal_local,
            tracks=self.latest_tracks,
            obstacle_points=self.latest_scan_points
        )

        self.cmd_pub.publish(cmd)

        self.prev_v = cmd.linear.x
        self.prev_w = cmd.angular.z

        self.publish_vo_markers()

        rospy.loginfo_throttle(
            0.5,
            "Mode=%s | GoalDist=%.2f | People=%d | safe=%d forbidden=%d | v=%.3f w=%.3f",
            self.mode,
            goal_dist,
            len(self.latest_tracks),
            len(self.safe_velocity_points),
            len(self.forbidden_velocity_points),
            cmd.linear.x,
            cmd.angular.z
        )

    # =====================================================
    # Goal transform
    # =====================================================
    def get_goal_in_robot_frame(self):
        dx = self.goal_x - self.robot_x
        dy = self.goal_y - self.robot_y

        c = math.cos(-self.robot_yaw)
        s = math.sin(-self.robot_yaw)

        gx = c * dx - s * dy
        gy = s * dx + c * dy

        return np.array([gx, gy], dtype=np.float64)

    # =====================================================
    # VO velocity selection
    # =====================================================
    def choose_velocity_by_vo(self, goal_local, tracks, obstacle_points):
        people = self.build_people_predictions(tracks)

        v_candidates = np.linspace(self.v_min, self.v_max, self.v_samples)
        w_candidates = np.linspace(-self.w_max, self.w_max, self.w_samples)

        if 0.0 not in w_candidates:
            w_candidates = np.append(w_candidates, 0.0)

        best_cost = float('inf')
        best_v = 0.0
        best_w = 0.0
        best_traj = []

        self.forbidden_velocity_points = []
        self.safe_velocity_points = []
        self.best_velocity_point = None

        for v in v_candidates:
            for w in w_candidates:
                traj = self.simulate_robot_trajectory(v, w)

                obstacle_collision, min_obs_dist = self.check_wall_collision(traj, obstacle_points)
                people_collision, min_people_dist, min_ttc = self.check_velocity_obstacle(traj, people)

                velocity_point = self.velocity_to_marker_point(v, w)

                if obstacle_collision or people_collision:
                    self.forbidden_velocity_points.append(velocity_point)
                    continue

                cost = self.compute_velocity_cost(
                    traj=traj,
                    v=v,
                    w=w,
                    goal_local=goal_local,
                    min_obs_dist=min_obs_dist,
                    min_people_dist=min_people_dist,
                    min_ttc=min_ttc
                )

                self.safe_velocity_points.append(velocity_point)

                if cost < best_cost:
                    best_cost = cost
                    best_v = v
                    best_w = w
                    best_traj = traj
                    self.best_velocity_point = velocity_point

        cmd = Twist()

        if best_cost == float('inf'):
            self.mode = "VO_BLOCKED"

            # 找不到完全安全速度時，優先慢速偏向比較空的一側，不要直接停在原地。
            left_clear = self.side_clearance(obstacle_points, side="left")
            right_clear = self.side_clearance(obstacle_points, side="right")
            front_clear = self.front_clearance(obstacle_points)

            if front_clear <= self.emergency_stop_distance:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = self.min_moving_speed

            if left_clear > right_clear + 0.05:
                cmd.angular.z = 0.35
            elif right_clear > left_clear + 0.05:
                cmd.angular.z = -0.35
            else:
                cmd.angular.z = 0.0

            self.best_traj = self.simulate_robot_trajectory(cmd.linear.x, cmd.angular.z)
        else:
            cmd.linear.x = max(float(best_v), self.min_moving_speed)
            cmd.angular.z = float(best_w)
            self.best_traj = best_traj

            if len(people) > 0:
                self.mode = "VO_AVOID_PEOPLE"
            else:
                self.mode = "VO_GOAL"

        if cmd.linear.x > 1e-4:
            v_smooth = 0.80 * self.prev_v + 0.20 * cmd.linear.x
            w_smooth = 0.75 * self.prev_w + 0.25 * cmd.angular.z
            cmd.linear.x = clamp(max(v_smooth, self.min_moving_speed), 0.0, self.v_max)
            cmd.angular.z = clamp(w_smooth, -self.w_max, self.w_max)

        return cmd

    # =====================================================
    # Build people predictions from EKF
    # =====================================================
    def build_people_predictions(self, tracks):
        people = []

        for t in tracks:
            x = float(t.X[0])
            y = float(t.X[1])
            d = math.hypot(x, y)

            if d > self.max_range:
                continue

            speed = abs(float(t.X[2]))

            if speed >= 0.25:
                person_radius = self.person_radius_dynamic
            else:
                person_radius = self.person_radius_static

            sigma = 0.06

            combined_radius = self.robot_radius + person_radius + self.prediction_margin + self.early_avoid_margin + sigma

            people.append({
                'id': t.id,
                'track': t,
                'combined_radius': combined_radius,
                'speed': speed
            })

        return people

    # =====================================================
    # Simulate robot trajectory for a candidate velocity
    # =====================================================
    def simulate_robot_trajectory(self, v, w):
        traj = []

        x = 0.0
        y = 0.0
        th = 0.0

        steps = int(self.time_horizon / self.sim_dt)

        for i in range(steps + 1):
            tau = i * self.sim_dt
            traj.append((x, y, th, tau))

            x += v * math.cos(th) * self.sim_dt
            y += v * math.sin(th) * self.sim_dt
            th = wrap_to_pi(th + w * self.sim_dt)

        return traj

    # =====================================================
    # Simplified Velocity Obstacle check
    # =====================================================
    def check_velocity_obstacle(self, traj, people):
        if len(people) == 0:
            return False, 5.0, self.time_horizon

        min_clearance = 5.0
        min_ttc = self.time_horizon

        for x, y, th, tau in traj:
            robot_pos = np.array([x, y], dtype=np.float64)

            for person in people:
                person_pos = person['track'].predict_position(tau)
                radius = person['combined_radius']

                dist = float(np.linalg.norm(robot_pos - person_pos))
                clearance = dist - radius
                min_clearance = min(min_clearance, clearance)

                # 這裡用 EKF 在 tau 秒後的位置做判斷；radius 已包含 early_avoid_margin，
                # 因此人還沒靠很近時，VO 就會先把可能撞上的速度排除。
                if dist < radius:
                    return True, min_clearance, tau

                # 距離越接近且時間越短，代表這個速度越接近 VO 邊界。
                if clearance < 0.8:
                    min_ttc = min(min_ttc, tau)

        return False, min_clearance, min_ttc

    # =====================================================
    # Wall / obstacle collision checking
    # =====================================================
    def check_wall_collision(self, traj, obstacle_points):
        if obstacle_points is None or obstacle_points.shape[0] == 0:
            return False, 5.0

        pts = obstacle_points[::2]
        min_dist = 5.0

        for x, y, th, tau in traj:
            dx = pts[:, 0] - x
            dy = pts[:, 1] - y
            d = np.sqrt(dx * dx + dy * dy)

            current_min = float(np.min(d))
            min_dist = min(min_dist, current_min)

            if current_min < self.obstacle_clearance:
                return True, min_dist

        return False, min_dist

    # =====================================================
    # Velocity cost
    # =====================================================
    def compute_velocity_cost(self, traj, v, w, goal_local, min_obs_dist, min_people_dist, min_ttc):
        end_x, end_y, end_th, _ = traj[-1]

        goal_dist_end = math.hypot(goal_local[0] - end_x, goal_local[1] - end_y)
        goal_heading = math.atan2(goal_local[1] - end_y, goal_local[0] - end_x)
        heading_error = abs(wrap_to_pi(goal_heading - end_th))

        clearance_cost = 1.0 / max(min_obs_dist - self.obstacle_clearance + 0.05, 0.05)

        if min_people_dist < 5.0:
            people_clearance_cost = 1.0 / max(min_people_dist + 0.05, 0.05)
        else:
            people_clearance_cost = 0.0

        smooth_cost = abs(v - self.prev_v) + 0.45 * abs(w - self.prev_w)

        speed_cost = abs(v - self.v_nominal)

        # min_ttc 越小代表越快接近碰撞邊界，因此成本越高
        ttc_cost = 1.0 / max(min_ttc + 0.2, 0.2)

        cost = (
            self.weight_goal * goal_dist_end
            + self.weight_heading * heading_error
            + self.weight_clearance * (clearance_cost + people_clearance_cost)
            + self.weight_smooth * smooth_cost
            + self.weight_speed * speed_cost
            + self.weight_ttc * ttc_cost
        )

        return cost

    # =====================================================
    # Side clearance for recovery turn
    # =====================================================
    def side_clearance(self, obstacle_points, side="left"):
        if obstacle_points is None or obstacle_points.shape[0] == 0:
            return 5.0

        pts = obstacle_points

        if side == "left":
            mask = (pts[:, 0] > -0.2) & (pts[:, 0] < 1.0) & (pts[:, 1] > 0.0)
        else:
            mask = (pts[:, 0] > -0.2) & (pts[:, 0] < 1.0) & (pts[:, 1] < 0.0)

        if not np.any(mask):
            return 5.0

        return float(np.min(np.linalg.norm(pts[mask], axis=1)))

    def front_clearance(self, obstacle_points):
        if obstacle_points is None or obstacle_points.shape[0] == 0:
            return 5.0

        pts = obstacle_points
        mask = (
            (pts[:, 0] > 0.0)
            & (pts[:, 0] < 1.2)
            & (np.abs(pts[:, 1]) < 0.32)
        )

        if not np.any(mask):
            return 5.0

        return float(np.min(pts[mask, 0]))

    # =====================================================
    # Velocity space marker point
    # =====================================================
    def velocity_to_marker_point(self, v, w):
        # 在 RViz 中畫一個速度空間：
        # x 軸 = v 放大
        # y 軸 = w 放大
        return np.array([v * 4.0, w * 0.6 - 1.5], dtype=np.float64)

    # =====================================================
    # People visualization
    # =====================================================
    def publish_delete_all_people(self, frame_id):
        arr = MarkerArray()

        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = rospy.Time.now()
        m.action = Marker.DELETEALL

        arr.markers.append(m)
        self.people_marker_pub.publish(arr)

    def create_circle_points(self, cx, cy, r, z=0.05, num_points=36):
        points = []

        for i in range(num_points + 1):
            angle = 2.0 * math.pi * i / num_points

            p = Point()
            p.x = cx + r * math.cos(angle)
            p.y = cy + r * math.sin(angle)
            p.z = z

            points.append(p)

        return points

    def publish_people_markers(self, tracks, frame_id):
        arr = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.action = Marker.DELETEALL
        arr.markers.append(delete_marker)

        marker_id = 0

        # Robot base.
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
        m_robot.scale.x = 0.2
        m_robot.scale.y = 0.2
        m_robot.scale.z = 0.2
        m_robot.color.r = 1.0
        m_robot.color.g = 1.0
        m_robot.color.b = 1.0
        m_robot.color.a = 1.0
        arr.markers.append(m_robot)

        predict_time = 1.0

        for t in tracks:
            color_r = (t.id * 0.2) % 1.0
            color_g = 1.0 - (t.id * 0.3) % 1.0
            color_b = (t.id * 0.5) % 1.0

            # Current person.
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
            m_person.scale.x = 0.3
            m_person.scale.y = 0.3
            m_person.scale.z = 0.5
            m_person.color.r = color_r
            m_person.color.g = color_g
            m_person.color.b = color_b
            m_person.color.a = 0.8
            arr.markers.append(m_person)

            # ID text.
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
            m_text.pose.position.z = 0.75
            m_text.scale.z = 0.15
            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = "ID:{} v={:.2f}".format(t.id, abs(t.X[2]))
            arr.markers.append(m_text)

            # Predicted point.
            pred = t.predict_position(predict_time)

            m_pred = Marker()
            m_pred.header.frame_id = frame_id
            m_pred.header.stamp = rospy.Time.now()
            m_pred.ns = "prediction_point"
            m_pred.id = marker_id
            marker_id += 1
            m_pred.type = Marker.SPHERE
            m_pred.action = Marker.ADD
            m_pred.pose.position.x = float(pred[0])
            m_pred.pose.position.y = float(pred[1])
            m_pred.pose.position.z = 0.1
            m_pred.scale.x = 0.12
            m_pred.scale.y = 0.12
            m_pred.scale.z = 0.12
            m_pred.color.r = color_r
            m_pred.color.g = color_g
            m_pred.color.b = color_b
            m_pred.color.a = 0.55
            arr.markers.append(m_pred)

            # Line current -> predicted.
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
            p_pred.x = float(pred[0])
            p_pred.y = float(pred[1])
            p_pred.z = 0.15

            m_line.points.append(p_now)
            m_line.points.append(p_pred)
            arr.markers.append(m_line)

            # Sigma circle.
            sigma_r = math.sqrt(max(t.P[0, 0], t.P[1, 1]))
            sigma_r = clamp(sigma_r, 0.04, 0.20)

            m_sigma = Marker()
            m_sigma.header.frame_id = frame_id
            m_sigma.header.stamp = rospy.Time.now()
            m_sigma.ns = "sigma_range"
            m_sigma.id = marker_id
            marker_id += 1
            m_sigma.type = Marker.LINE_STRIP
            m_sigma.action = Marker.ADD
            m_sigma.pose.orientation.w = 1.0
            m_sigma.scale.x = 0.025
            m_sigma.color.r = color_r
            m_sigma.color.g = color_g
            m_sigma.color.b = color_b
            m_sigma.color.a = 0.55
            m_sigma.points = self.create_circle_points(float(pred[0]), float(pred[1]), sigma_r)
            arr.markers.append(m_sigma)

            # Collision radius circle used by VO.
            speed = abs(float(t.X[2]))

            if speed >= 0.25:
                person_radius = self.person_radius_dynamic
            else:
                person_radius = self.person_radius_static

            combined_radius = self.robot_radius + person_radius + self.prediction_margin + sigma_r

            m_vo_circle = Marker()
            m_vo_circle.header.frame_id = frame_id
            m_vo_circle.header.stamp = rospy.Time.now()
            m_vo_circle.ns = "vo_collision_radius"
            m_vo_circle.id = marker_id
            marker_id += 1
            m_vo_circle.type = Marker.LINE_STRIP
            m_vo_circle.action = Marker.ADD
            m_vo_circle.pose.orientation.w = 1.0
            m_vo_circle.scale.x = 0.025
            m_vo_circle.color.r = 1.0
            m_vo_circle.color.g = 0.25
            m_vo_circle.color.b = 0.0
            m_vo_circle.color.a = 0.75
            m_vo_circle.points = self.create_circle_points(float(pred[0]), float(pred[1]), combined_radius)
            arr.markers.append(m_vo_circle)

        self.people_marker_pub.publish(arr)

    # =====================================================
    # VO visualization
    # =====================================================
    def publish_vo_markers(self):
        frame_id = self.latest_frame
        arr = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.action = Marker.DELETEALL
        arr.markers.append(delete_marker)

        marker_id = 0

        # Selected trajectory.
        m_traj = Marker()
        m_traj.header.frame_id = frame_id
        m_traj.header.stamp = rospy.Time.now()
        m_traj.ns = "selected_trajectory"
        m_traj.id = marker_id
        marker_id += 1
        m_traj.type = Marker.LINE_STRIP
        m_traj.action = Marker.ADD
        m_traj.pose.orientation.w = 1.0
        m_traj.scale.x = 0.045
        m_traj.color.r = 0.0
        m_traj.color.g = 1.0
        m_traj.color.b = 1.0
        m_traj.color.a = 1.0

        for x, y, th, tau in self.best_traj:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            p.z = 0.08
            m_traj.points.append(p)

        arr.markers.append(m_traj)

        # Local goal marker.
        if self.odom_ready and self.start_pose_set:
            goal_local = self.get_goal_in_robot_frame()

            m_goal = Marker()
            m_goal.header.frame_id = frame_id
            m_goal.header.stamp = rospy.Time.now()
            m_goal.ns = "local_goal"
            m_goal.id = marker_id
            marker_id += 1
            m_goal.type = Marker.SPHERE
            m_goal.action = Marker.ADD
            m_goal.pose.position.x = float(goal_local[0])
            m_goal.pose.position.y = float(goal_local[1])
            m_goal.pose.position.z = 0.15
            m_goal.scale.x = 0.25
            m_goal.scale.y = 0.25
            m_goal.scale.z = 0.25
            m_goal.color.r = 0.0
            m_goal.color.g = 1.0
            m_goal.color.b = 0.0
            m_goal.color.a = 1.0
            arr.markers.append(m_goal)

        # Safe velocity points.
        m_safe = Marker()
        m_safe.header.frame_id = frame_id
        m_safe.header.stamp = rospy.Time.now()
        m_safe.ns = "velocity_space_safe"
        m_safe.id = marker_id
        marker_id += 1
        m_safe.type = Marker.POINTS
        m_safe.action = Marker.ADD
        m_safe.pose.orientation.w = 1.0
        m_safe.scale.x = 0.045
        m_safe.scale.y = 0.045
        m_safe.color.r = 0.0
        m_safe.color.g = 1.0
        m_safe.color.b = 0.0
        m_safe.color.a = 0.7

        for vp in self.safe_velocity_points:
            p = Point()
            p.x = float(vp[0])
            p.y = float(vp[1])
            p.z = 0.12
            m_safe.points.append(p)

        arr.markers.append(m_safe)

        # Forbidden velocity points.
        m_forbid = Marker()
        m_forbid.header.frame_id = frame_id
        m_forbid.header.stamp = rospy.Time.now()
        m_forbid.ns = "velocity_space_forbidden"
        m_forbid.id = marker_id
        marker_id += 1
        m_forbid.type = Marker.POINTS
        m_forbid.action = Marker.ADD
        m_forbid.pose.orientation.w = 1.0
        m_forbid.scale.x = 0.055
        m_forbid.scale.y = 0.055
        m_forbid.color.r = 1.0
        m_forbid.color.g = 0.0
        m_forbid.color.b = 0.0
        m_forbid.color.a = 0.8

        for vp in self.forbidden_velocity_points:
            p = Point()
            p.x = float(vp[0])
            p.y = float(vp[1])
            p.z = 0.12
            m_forbid.points.append(p)

        arr.markers.append(m_forbid)

        # Best velocity point.
        if self.best_velocity_point is not None:
            m_best = Marker()
            m_best.header.frame_id = frame_id
            m_best.header.stamp = rospy.Time.now()
            m_best.ns = "velocity_space_best"
            m_best.id = marker_id
            marker_id += 1
            m_best.type = Marker.SPHERE
            m_best.action = Marker.ADD
            m_best.pose.position.x = float(self.best_velocity_point[0])
            m_best.pose.position.y = float(self.best_velocity_point[1])
            m_best.pose.position.z = 0.18
            m_best.scale.x = 0.13
            m_best.scale.y = 0.13
            m_best.scale.z = 0.13
            m_best.color.r = 1.0
            m_best.color.g = 1.0
            m_best.color.b = 0.0
            m_best.color.a = 1.0
            arr.markers.append(m_best)

        # Mode text.
        m_text = Marker()
        m_text.header.frame_id = frame_id
        m_text.header.stamp = rospy.Time.now()
        m_text.ns = "vo_mode"
        m_text.id = marker_id
        marker_id += 1
        m_text.type = Marker.TEXT_VIEW_FACING
        m_text.action = Marker.ADD
        m_text.pose.position.x = 0.0
        m_text.pose.position.y = -1.0
        m_text.pose.position.z = 0.7
        m_text.scale.z = 0.18
        m_text.color.r = 1.0
        m_text.color.g = 1.0
        m_text.color.b = 0.0
        m_text.color.a = 1.0
        m_text.text = "Mode: {} | v={:.2f}, w={:.2f}".format(
            self.mode,
            self.prev_v,
            self.prev_w
        )
        arr.markers.append(m_text)

        self.vo_marker_pub.publish(arr)

    # =====================================================
    # Stop / shutdown
    # =====================================================
    def publish_stop(self):
        cmd = Twist()
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
        self.cmd_pub.publish(cmd)

        self.prev_v = 0.0
        self.prev_w = 0.0

    def on_shutdown(self):
        self.publish_stop()
        rospy.sleep(0.2)
        rospy.loginfo("Robot stopped.")


# =========================================================
# 9. Main
# =========================================================
if __name__ == '__main__':
    try:
        navigator = VelocityObstacleNavigator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
