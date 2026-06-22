#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time person position detection using:
1. LiDAR segmentation
2. 10 handcrafted features
3. Exported AdaBoost model from MATLAB
4. Pairing two legs into one person position
5. EKF people tracking
6. DWA avoidance
7. Return to original straight path and stop at 8m goal
8. Early front-person avoidance:
   EKF predicts whether a tracked person will appear in front 2m region.
9. Constant-speed motion:
   The robot does NOT stop or slow down during avoidance.
   DWA only selects angular velocity w. Linear velocity v is fixed.
"""

import math
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry


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
# 2. segmentation
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
# 3. 10 維特徵
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
# 4. AdaBoost 分數
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
# 5. 腳配對成人
# =========================================================
def pair_legs_to_people(detected_legs, max_leg_distance=0.6):
    people = []
    used = set()

    for i in range(len(detected_legs)):
        if i in used:
            continue

        xi = detected_legs[i]['cx']
        yi = detected_legs[i]['cy']

        best_j = -1
        best_dist = float('inf')

        for j in range(i + 1, len(detected_legs)):
            if j in used:
                continue

            xj = detected_legs[j]['cx']
            yj = detected_legs[j]['cy']

            dist = math.hypot(xi - xj, yi - yj)

            if dist < best_dist and dist <= max_leg_distance:
                best_dist = dist
                best_j = j

        if best_j != -1:
            used.add(i)
            used.add(best_j)

            xj = detected_legs[best_j]['cx']
            yj = detected_legs[best_j]['cy']

            people.append({
                'x': 0.5 * (xi + xj),
                'y': 0.5 * (yi + yj),
                'leg1_id': detected_legs[i]['id'],
                'leg2_id': detected_legs[best_j]['id'],
                'leg_distance': best_dist
            })

    return people


# =========================================================
# 手寫簡化 assignment
# =========================================================
def min_weight_assignment(cost_matrix):
    n, m = cost_matrix.shape

    if n == 0 or m == 0:
        return []

    rows, cols = np.where(cost_matrix < 1e8)

    potential_pairs = sorted(
        zip(rows, cols),
        key=lambda x: cost_matrix[x[0], x[1]]
    )

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
# EKF Track & Tracker
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
        self.X[3] = math.atan2(math.sin(self.X[3]), math.cos(self.X[3]))

    def update(self, z, R):
        H = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0]
        ], dtype=np.float64)

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.X += K @ (z - self.X[0:2])
        self.P = (np.eye(5) - K @ H) @ self.P


class EKFTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

        self.Q = np.diag([0.001, 0.001, 0.01, 0.01, 0.01])
        self.R = np.eye(2) * 0.05

        self.dist_gate = 0.8
        self.max_missed = 5

    def update_tracks(self, measurements, dt):
        if dt <= 0.0 or dt > 1.0:
            dt = 0.1

        for t in self.tracks:
            t.predict(dt, self.Q)

        n = len(self.tracks)
        m = len(measurements)

        cost_matrix = np.full((n, m), 1e9)

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
            self.tracks[i].age += 1

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
# DWA Local Planner
# =========================================================
class DWAPlanner:
    def __init__(self):
        self.max_speed = 0.15
        self.min_speed = 0.15

        self.max_yaw_rate = 1.4
        self.max_accel = 0.5
        self.max_delta_yaw_rate = 2.8

        self.min_forward_speed_when_avoiding = 0.05

        self.v_res = 0.02
        self.yaw_rate_res = 0.05
        self.dt = 0.1

        self.predict_time = 3.0

        self.robot_radius = 0.20
        self.safe_margin = 0.20

        self.heading_weight = 0.7
        self.wall_weight = 0.5
        self.ped_weight = 1.5
        self.vo_weight = 3.0

        self.velocity_weight = 0.15

        self.yaw_rate_weight = 0.08
        self.spin_in_place_weight = 1.5

    def plan(self, state, goal, static_scan, active_tracks, robot_pose_global):
        Vs = [
            self.min_speed,
            self.max_speed,
            -self.max_yaw_rate,
            self.max_yaw_rate
        ]

        Vd = [
            state[3] - self.max_accel * self.dt,
            state[3] + self.max_accel * self.dt,
            state[4] - self.max_delta_yaw_rate * self.dt,
            state[4] + self.max_delta_yaw_rate * self.dt
        ]

        dw = [
            max(Vs[0], Vd[0]),
            min(Vs[1], Vd[1]),
            max(Vs[2], Vd[2]),
            min(Vs[3], Vd[3])
        ]

        best_v = self.max_speed
        best_w = 0.0
        min_cost = float('inf')

        static_scan_fast = static_scan[::4] if len(static_scan) > 0 else static_scan

        is_avoiding = abs(goal[1]) > 0.30 and goal[0] > 0.0

        for v in np.arange(dw[0], dw[1] + self.v_res, self.v_res):
            if is_avoiding and v < self.min_forward_speed_when_avoiding:
                continue

            for w in np.arange(dw[2], dw[3] + self.yaw_rate_res, self.yaw_rate_res):
                trajectory = self.predict_trajectory(state, v, w)

                cost = self.calc_trajectory_cost(
                    trajectory,
                    goal,
                    static_scan_fast,
                    active_tracks,
                    v,
                    w,
                    robot_pose_global
                )

                if cost < min_cost:
                    min_cost = cost
                    best_v = v
                    best_w = w

        if min_cost == float('inf'):
            return self.max_speed, 0.5

        return best_v, best_w

    def predict_trajectory(self, state, v, w):
        trajectory = []

        x = state[0]
        y = state[1]
        yaw = state[2]

        time = 0.0

        while time <= self.predict_time:
            x += v * math.cos(yaw) * self.dt
            y += v * math.sin(yaw) * self.dt
            yaw += w * self.dt

            trajectory.append([x, y, yaw, time])
            time += self.dt

        return trajectory

    def calc_vo_cost(self, v_cmd, w_cmd, active_tracks, robot_pose_global):
        cx, cy, cyaw = robot_pose_global

        global_v_rx = v_cmd * math.cos(cyaw)
        global_v_ry = v_cmd * math.sin(cyaw)

        vo_cost = 0.0

        for track in active_tracks:
            px = track.X[0]
            py = track.X[1]

            p_vx = track.X[2] * math.cos(track.X[3])
            p_vy = track.X[2] * math.sin(track.X[3])

            rel_x = px - cx
            rel_y = py - cy

            rel_vx = global_v_rx - p_vx
            rel_vy = global_v_ry - p_vy

            dist = math.hypot(rel_x, rel_y)

            if dist < 0.1:
                continue

            r_safe = self.robot_radius + 0.35

            if dist < r_safe:
                vo_cost += 100.0
                continue

            alpha = math.asin(min(r_safe / dist, 0.999))
            phi = math.atan2(rel_y, rel_x)
            theta_rel = math.atan2(rel_vy, rel_vx)

            angle_diff = abs(
                math.atan2(
                    math.sin(theta_rel - phi),
                    math.cos(theta_rel - phi)
                )
            )

            if angle_diff < alpha:
                rel_v_norm = math.hypot(rel_vx, rel_vy)

                if rel_v_norm > 0.05:
                    ttc = dist / rel_v_norm

                    if ttc < self.predict_time:
                        vo_cost += math.exp(-ttc + 3.0)

        return vo_cost

    def calc_trajectory_cost(
        self,
        trajectory,
        goal,
        static_scan,
        active_tracks,
        v_cmd,
        w_cmd,
        robot_pose_global
    ):
        cx, cy, cyaw = robot_pose_global

        end_state = trajectory[-1]

        target_angle = math.atan2(
            goal[1] - end_state[1],
            goal[0] - end_state[0]
        )

        heading_error = abs(target_angle - end_state[2])
        heading_error = min(heading_error, 2.0 * math.pi - heading_error)
        heading_cost = heading_error / math.pi

        min_wall_dist = float('inf')
        min_ped_dist = float('inf')

        for step in trajectory[::2]:
            local_rx, local_ry, _, t = step

            if len(static_scan) > 0:
                dists = np.linalg.norm(
                    static_scan - np.array([local_rx, local_ry]),
                    axis=1
                )

                step_min_wall = np.min(dists)

                if step_min_wall < min_wall_dist:
                    min_wall_dist = step_min_wall

            global_rx = cx + local_rx * math.cos(cyaw) - local_ry * math.sin(cyaw)
            global_ry = cy + local_rx * math.sin(cyaw) + local_ry * math.cos(cyaw)

            for track in active_tracks:
                pv = track.X[2]
                pth = track.X[3]
                pom = track.X[4]

                if abs(pom) < 1e-3:
                    px = track.X[0] + pv * math.cos(pth) * t
                    py = track.X[1] + pv * math.sin(pth) * t
                else:
                    px = track.X[0] + (pv / pom) * (
                        math.sin(pth + pom * t) - math.sin(pth)
                    )
                    py = track.X[1] + (pv / pom) * (
                        -math.cos(pth + pom * t) + math.cos(pth)
                    )

                dist_to_ped = math.hypot(global_rx - px, global_ry - py)
                clear_dist = dist_to_ped - 0.20

                if clear_dist < min_ped_dist:
                    min_ped_dist = clear_dist

        wall_collision_penalty = 0.0
        if min_wall_dist < (self.robot_radius + 0.05):
            wall_collision_penalty = 800.0

        ped_collision_penalty = 0.0
        if min_ped_dist < (self.robot_radius + 0.05):
            ped_collision_penalty = 1000.0

        wall_cost = math.exp(-3.0 * max(0.0, min_wall_dist))
        ped_cost = math.exp(-2.0 * max(0.0, min_ped_dist))

        vo_cost = self.calc_vo_cost(
            v_cmd,
            w_cmd,
            active_tracks,
            robot_pose_global
        )

        clearance_cost = (
            self.wall_weight * wall_cost
            + self.ped_weight * ped_cost
            + self.vo_weight * vo_cost
            + wall_collision_penalty
            + ped_collision_penalty
        )

        velocity_cost = (self.max_speed - v_cmd) / max(
            self.max_speed - self.min_speed,
            1e-6
        )

        yaw_rate_cost = abs(w_cmd) / max(self.max_yaw_rate, 1e-6)

        spin_in_place_cost = 0.0
        if abs(w_cmd) > 0.25 and v_cmd < 0.06:
            spin_in_place_cost = (0.06 - v_cmd) / 0.06

        total_cost = (
            self.heading_weight * heading_cost
            + clearance_cost
            + self.velocity_weight * velocity_cost
            + self.yaw_rate_weight * yaw_rate_cost
            + self.spin_in_place_weight * spin_in_place_cost
        )

        return total_cost


# =========================================================
# 6. ROS node
# =========================================================
class RealTimeLegDetector:
    def __init__(self):
        rospy.init_node('realtime_leg_detector', anonymous=True)

        scan_topic = rospy.get_param('~scan_topic', '/scan')

        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.05)
        self.segment_threshold = rospy.get_param('~segment_threshold', 0.1)
        self.max_leg_distance = rospy.get_param('~max_leg_distance', 0.6)
        self.debug = rospy.get_param('~debug', False)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.marker_pub = rospy.Publisher('/people_markers', MarkerArray, queue_size=1)

        self.sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)

        self.tracker = EKFTracker()
        self.last_time = None

        self.dwa = DWAPlanner()
        self.current_v = 0.0
        self.current_w = 0.0

        self.latest_scan_pts = np.empty((0, 2), dtype=np.float64)
        self.active_tracks = []

        self.start_pose = None
        self.current_pose = None
        self.global_goal = None
        self.goal_reached = False

        self.target_distance = 8.0
        self.path_lookahead = 1.0

        self.front_avoid_distance = 2.0
        self.front_avoid_width = 0.75
        self.avoid_side_offset = 0.85
        self.avoid_forward_goal = 1.2
        self.center_dead_zone = 0.10

        self.latest_avoid_goal = None
        self.cruise_speed = 0.15
        self.cmd_smooth_alpha_w = 0.50

        self.odom_frame_id = "odom"

        self.move_timer = rospy.Timer(rospy.Duration(0.05), self.move_robot)

        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("Real-time leg tracker with constant-speed EKF + DWA avoidance is ready.")

    # =====================================================
    # 座標轉換
    # =====================================================
    def global_to_local(self, gx, gy, cx, cy, cyaw):
        dx = gx - cx
        dy = gy - cy

        local_x = dx * math.cos(-cyaw) - dy * math.sin(-cyaw)
        local_y = dx * math.sin(-cyaw) + dy * math.cos(-cyaw)

        return [local_x, local_y]

    def local_to_global(self, lx, ly, cx, cy, cyaw):
        gx = cx + lx * math.cos(cyaw) - ly * math.sin(cyaw)
        gy = cy + lx * math.sin(cyaw) + ly * math.cos(cyaw)

        return gx, gy

    def predict_track_position(self, track, dt):
        x = track.X[0]
        y = track.X[1]
        v = track.X[2]
        th = track.X[3]
        om = track.X[4]

        if abs(om) < 1e-3:
            px = x + v * math.cos(th) * dt
            py = y + v * math.sin(th) * dt
        else:
            px = x + (v / om) * (math.sin(th + om * dt) - math.sin(th))
            py = y + (v / om) * (-math.cos(th + om * dt) + math.cos(th))

        return px, py

    def get_return_goal_on_start_line(self):
        if self.start_pose is None or self.current_pose is None or self.global_goal is None:
            return self.global_goal

        sx, sy, syaw = self.start_pose
        cx, cy, cyaw = self.current_pose
        gx, gy = self.global_goal

        dist_to_goal = math.hypot(gx - cx, gy - cy)

        if dist_to_goal < 1.0:
            return self.global_goal

        dir_x = math.cos(syaw)
        dir_y = math.sin(syaw)

        rel_x = cx - sx
        rel_y = cy - sy

        progress = rel_x * dir_x + rel_y * dir_y
        target_s = progress + self.path_lookahead

        target_s = max(0.0, min(self.target_distance, target_s))

        return_x = sx + target_s * dir_x
        return_y = sy + target_s * dir_y

        return return_x, return_y

    def get_front_person_avoid_goal(self):
        self.latest_avoid_goal = None

        if self.current_pose is None:
            return None

        if len(self.active_tracks) == 0:
            return None

        cx, cy, cyaw = self.current_pose

        closest_front_person = None
        closest_x = float('inf')

        for track in self.active_tracks:
            candidate_global_positions = [
                (track.X[0], track.X[1], 0.0),
                (*self.predict_track_position(track, 1.0), 1.0),
                (*self.predict_track_position(track, 2.0), 2.0)
            ]

            for px, py, pred_t in candidate_global_positions:
                dx = px - cx
                dy = py - cy

                local_x = dx * math.cos(-cyaw) - dy * math.sin(-cyaw)
                local_y = dx * math.sin(-cyaw) + dy * math.cos(-cyaw)

                if (
                    0.0 < local_x < self.front_avoid_distance
                    and abs(local_y) < self.front_avoid_width
                ):
                    if local_x < closest_x:
                        closest_x = local_x
                        closest_front_person = (local_x, local_y, pred_t)

        if closest_front_person is None:
            return None

        person_local_x, person_local_y, pred_t = closest_front_person

        if person_local_y > self.center_dead_zone:
            avoid_y = -self.avoid_side_offset
        elif person_local_y < -self.center_dead_zone:
            avoid_y = self.avoid_side_offset
        else:
            avoid_y = -self.avoid_side_offset

        if person_local_x < 1.0:
            avoid_x = 0.9
        else:
            avoid_x = self.avoid_forward_goal

        self.latest_avoid_goal = [avoid_x, avoid_y]

        rospy.loginfo_throttle(
            0.5,
            "Front predicted person within 2m: person local=({:.2f}, {:.2f}), pred_t={:.1f}s, avoid local goal=({:.2f}, {:.2f})".format(
                person_local_x,
                person_local_y,
                pred_t,
                avoid_x,
                avoid_y
            )
        )

        return [avoid_x, avoid_y]

    # =====================================================
    # Odometry callback
    # =====================================================
    def odom_callback(self, msg):
        self.odom_frame_id = msg.header.frame_id if msg.header.frame_id else "odom"

        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        yaw = math.atan2(siny_cosp, cosy_cosp)

        self.current_pose = (x, y, yaw)

        if self.start_pose is None:
            self.start_pose = (x, y, yaw)

            goal_x = x + self.target_distance * math.cos(yaw)
            goal_y = y + self.target_distance * math.sin(yaw)

            self.global_goal = (goal_x, goal_y)

            rospy.loginfo(
                "=== 起點記錄完畢: X={:.2f}, Y={:.2f}, Yaw={:.2f} ===".format(
                    x, y, yaw
                )
            )

            rospy.loginfo(
                "=== 終點設定完成: 起點前方 {:.1f} m, Goal X={:.2f}, Goal Y={:.2f} ===".format(
                    self.target_distance,
                    goal_x,
                    goal_y
                )
            )

    # =====================================================
    # Robot control
    # =====================================================
    def move_robot(self, event):
        if self.global_goal is None or self.current_pose is None:
            return

        cx, cy, cyaw = self.current_pose
        gx, gy = self.global_goal

        dist_to_goal = math.hypot(gx - cx, gy - cy)

        if dist_to_goal < 0.3:
            self.goal_reached = True
            self.current_v = 0.0
            self.current_w = 0.0
            self.cmd_pub.publish(Twist())
            rospy.loginfo("=== 已抵達起點前方 8m 的終點，停止機器人 ===")
            return

        avoid_goal = self.get_front_person_avoid_goal()

        if avoid_goal is not None:
            local_goal = avoid_goal
        else:
            return_goal = self.get_return_goal_on_start_line()

            if return_goal is None:
                return

            rgx, rgy = return_goal
            local_goal = self.global_to_local(rgx, rgy, cx, cy, cyaw)

        state = [0.0, 0.0, 0.0, self.cruise_speed, self.current_w]
        robot_pose_global = self.current_pose

        raw_v, raw_w = self.dwa.plan(
            state,
            local_goal,
            self.latest_scan_pts,
            self.active_tracks,
            robot_pose_global
        )

        alpha_w = self.cmd_smooth_alpha_w
        best_v = self.cruise_speed
        best_w = (1.0 - alpha_w) * self.current_w + alpha_w * raw_w

        best_w = max(-self.dwa.max_yaw_rate, min(self.dwa.max_yaw_rate, best_w))

        self.current_v = best_v
        self.current_w = best_w

        move_cmd = Twist()
        move_cmd.linear.x = best_v
        move_cmd.angular.z = best_w

        self.cmd_pub.publish(move_cmd)

    # =====================================================
    # LiDAR callback
    # =====================================================
    def scan_callback(self, msg):
        if self.current_pose is None:
            return

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
            # [修改] 使用雷射的 frame_id 確保在 RViz 能正常清除
            self.publish_delete_all(msg.header.frame_id)
            self.latest_scan_pts = np.empty((0, 2), dtype=np.float64)
            return

        xy_points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        self.latest_scan_pts = xy_points

        clusters = segment_lidar(xy_points, threshold=self.segment_threshold)

        detected_legs = []

        for i, pts in enumerate(clusters):
            feats = extract_10_features(pts)

            if feats is None or not np.any(feats != 0):
                continue

            score = adaboost_score_single(feats)

            if score > THR_LEG:
                detected_legs.append({
                    'id': i,
                    'cx': float(np.mean(pts[:, 0])),
                    'cy': float(np.mean(pts[:, 1]))
                })

        people_positions = pair_legs_to_people(
            detected_legs,
            max_leg_distance=self.max_leg_distance
        )

        cx, cy, cyaw = self.current_pose

        global_measurements = []

        for p in people_positions:
            lx = p['x']
            ly = p['y']

            gx = cx + lx * math.cos(cyaw) - ly * math.sin(cyaw)
            gy = cy + lx * math.sin(cyaw) + ly * math.cos(cyaw)

            global_measurements.append(np.array([gx, gy], dtype=np.float64))

        current_time = msg.header.stamp

        if self.last_time is None:
            dt = 0.1
        else:
            dt = (current_time - self.last_time).to_sec()

        self.last_time = current_time

        active_tracks = self.tracker.update_tracks(global_measurements, dt)
        self.active_tracks = active_tracks

        # [修改] 使用雷射的 frame_id，並傳入 current_pose 供投影轉換使用
        self.publish_track_markers(
            active_tracks,
            msg.header.frame_id,
            dt,
            self.current_pose
        )

    # =====================================================
    # RViz markers
    # =====================================================
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

    # [修改] 增加 current_pose 參數，並在內部把全域座標換算回相對機器人的局部座標
    def publish_track_markers(self, tracks, frame_id, dt, current_pose):
        if current_pose is None:
            return
            
        cx, cy, cyaw = current_pose
        arr = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.header.stamp = rospy.Time.now()
        delete_marker.action = Marker.DELETEALL
        arr.markers.append(delete_marker)

        marker_id = 0

        # [修改] 顯示機器人自身位置 (因為是局部座標，機器人固定在 (0,0))
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

        # [修改] 顯示真正終點 (全域轉局部)
        if self.global_goal is not None:
            lx, ly = self.global_to_local(self.global_goal[0], self.global_goal[1], cx, cy, cyaw)
            
            m_goal = Marker()
            m_goal.header.frame_id = frame_id
            m_goal.header.stamp = rospy.Time.now()
            m_goal.ns = "goal"
            m_goal.id = marker_id
            marker_id += 1

            m_goal.type = Marker.SPHERE
            m_goal.action = Marker.ADD

            m_goal.pose.position.x = lx
            m_goal.pose.position.y = ly
            m_goal.pose.position.z = 0.1

            m_goal.scale.x = 0.25
            m_goal.scale.y = 0.25
            m_goal.scale.z = 0.25

            m_goal.color.r = 0.0
            m_goal.color.g = 1.0
            m_goal.color.b = 0.0
            m_goal.color.a = 1.0

            arr.markers.append(m_goal)

        # [修改] 顯示回歸路線前視點 (全域轉局部)
        if self.global_goal is not None:
            return_goal = self.get_return_goal_on_start_line()

            if return_goal is not None:
                lx, ly = self.global_to_local(return_goal[0], return_goal[1], cx, cy, cyaw)
                
                m_return = Marker()
                m_return.header.frame_id = frame_id
                m_return.header.stamp = rospy.Time.now()
                m_return.ns = "return_goal"
                m_return.id = marker_id
                marker_id += 1

                m_return.type = Marker.SPHERE
                m_return.action = Marker.ADD

                m_return.pose.position.x = lx
                m_return.pose.position.y = ly
                m_return.pose.position.z = 0.08

                m_return.scale.x = 0.15
                m_return.scale.y = 0.15
                m_return.scale.z = 0.15

                m_return.color.r = 0.0
                m_return.color.g = 0.5
                m_return.color.b = 1.0
                m_return.color.a = 1.0

                arr.markers.append(m_return)

        # [修改] 顯示目前避障 local goal (本身即為局部座標，直接顯示)
        if self.latest_avoid_goal is not None:
            m_avoid = Marker()
            m_avoid.header.frame_id = frame_id
            m_avoid.header.stamp = rospy.Time.now()
            m_avoid.ns = "front_avoid_goal"
            m_avoid.id = marker_id
            marker_id += 1

            m_avoid.type = Marker.SPHERE
            m_avoid.action = Marker.ADD

            m_avoid.pose.position.x = self.latest_avoid_goal[0]
            m_avoid.pose.position.y = self.latest_avoid_goal[1]
            m_avoid.pose.position.z = 0.12

            m_avoid.scale.x = 0.18
            m_avoid.scale.y = 0.18
            m_avoid.scale.z = 0.18

            m_avoid.color.r = 1.0
            m_avoid.color.g = 0.3
            m_avoid.color.b = 0.0
            m_avoid.color.a = 1.0

            arr.markers.append(m_avoid)

        predict_time = 2.0

        for t in tracks:
            # [修改] 人體追蹤位置：全域轉局部
            lx, ly = self.global_to_local(t.X[0], t.X[1], cx, cy, cyaw)
            
            m_person = Marker()
            m_person.header.frame_id = frame_id
            m_person.header.stamp = rospy.Time.now()
            m_person.ns = "tracked_people"
            m_person.id = marker_id
            marker_id += 1

            m_person.type = Marker.CYLINDER
            m_person.action = Marker.ADD

            m_person.pose.position.x = lx
            m_person.pose.position.y = ly
            m_person.pose.position.z = 0.25

            m_person.scale.x = 0.3
            m_person.scale.y = 0.3
            m_person.scale.z = 0.5

            m_person.color.r = (t.id * 0.2) % 1.0
            m_person.color.g = 1.0 - (t.id * 0.3) % 1.0
            m_person.color.b = (t.id * 0.5) % 1.0
            m_person.color.a = 0.8

            arr.markers.append(m_person)

            m_text = Marker()
            m_text.header.frame_id = frame_id
            m_text.header.stamp = rospy.Time.now()
            m_text.ns = "track_id"
            m_text.id = marker_id
            marker_id += 1

            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.action = Marker.ADD

            m_text.pose.position.x = lx
            m_text.pose.position.y = ly
            m_text.pose.position.z = 0.7

            m_text.scale.z = 0.15

            m_text.color.r = 1.0
            m_text.color.g = 1.0
            m_text.color.b = 1.0
            m_text.color.a = 1.0

            m_text.text = "ID: {}".format(t.id)

            arr.markers.append(m_text)

            v = t.X[2]
            th = t.X[3]
            om = t.X[4]

            # 計算預測點的「全域座標」
            if abs(om) < 1e-3:
                px = t.X[0] + v * math.cos(th) * predict_time
                py = t.X[1] + v * math.sin(th) * predict_time

                F_p = np.eye(5)
                F_p[0, 2] = math.cos(th) * predict_time
                F_p[0, 3] = -v * math.sin(th) * predict_time
                F_p[1, 2] = math.sin(th) * predict_time
                F_p[1, 3] = v * math.cos(th) * predict_time
            else:
                px = t.X[0] + (v / om) * (
                    math.sin(th + om * predict_time) - math.sin(th)
                )
                py = t.X[1] + (v / om) * (
                    -math.cos(th + om * predict_time) + math.cos(th)
                )

                F_p = np.eye(5)
                F_p[0, 2] = (math.sin(th + om * predict_time) - math.sin(th)) / om
                F_p[0, 3] = (v / om) * (
                    math.cos(th + om * predict_time) - math.cos(th)
                )
                F_p[1, 2] = (-math.cos(th + om * predict_time) + math.cos(th)) / om
                F_p[1, 3] = (v / om) * (
                    math.sin(th + om * predict_time) - math.sin(th)
                )
                
            # [修改] 預測點：全域轉局部
            lpx, lpy = self.global_to_local(px, py, cx, cy, cyaw)

            P_pred = F_p @ t.P @ F_p.T + (self.tracker.Q * predict_time)
            sigma_r = math.sqrt(max(P_pred[0, 0], P_pred[1, 1]))

            m_pred_pt = Marker()
            m_pred_pt.header.frame_id = frame_id
            m_pred_pt.header.stamp = rospy.Time.now()
            m_pred_pt.ns = "prediction_point"
            m_pred_pt.id = marker_id
            marker_id += 1

            m_pred_pt.type = Marker.SPHERE
            m_pred_pt.action = Marker.ADD

            m_pred_pt.pose.position.x = lpx
            m_pred_pt.pose.position.y = lpy
            m_pred_pt.pose.position.z = 0.1

            m_pred_pt.scale.x = 0.1
            m_pred_pt.scale.y = 0.1
            m_pred_pt.scale.z = 0.1

            m_pred_pt.color = m_person.color
            m_pred_pt.color.a = 0.5

            arr.markers.append(m_pred_pt)

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

            m_sigma.color = m_person.color
            m_sigma.color.a = 0.6

            # [修改] 直接給入局部座標繪製圓圈
            m_sigma.points = self.create_circle_points(lpx, lpy, sigma_r)

            arr.markers.append(m_sigma)

        self.marker_pub.publish(arr)

    # =====================================================
    # Shutdown
    # =====================================================
    def on_shutdown(self):
        self.current_v = 0.0
        self.current_w = 0.0
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Robot stopped.")


# =========================================================
# 7. main
# =========================================================
if __name__ == '__main__':
    try:
        detector = RealTimeLegDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
