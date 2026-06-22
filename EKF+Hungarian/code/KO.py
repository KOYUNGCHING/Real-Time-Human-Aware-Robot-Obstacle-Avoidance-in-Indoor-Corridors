#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import rospy

from itertools import combinations, permutations
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


# =========================================================
# 1. AdaBoost 模型參數
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


# =========================================================
# 2. LiDAR segmentation
# =========================================================
def segment_lidar(xy_points, threshold=0.1):
    """依照相鄰 LiDAR 點距離，把雷射點分成多個 segment。"""
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
# 3. 10 維特徵萃取
# =========================================================
def extract_10_features(pts):
    """對單一 segment 萃取 10 個幾何特徵，給 AdaBoost 判斷是否為腳。"""
    k = pts.shape[0]
    if k < 2:
        return None

    EPS = 1e-12
    x = pts[:, 0]
    y = pts[:, 1]

    # 1. segment 點數
    point_count = float(k)

    # 2. 點到 centroid 的標準差
    mu = np.mean(pts, axis=0)
    diff_mu = pts - mu
    dist2_mu = np.sum(diff_mu ** 2, axis=1)
    std_dev_to_centroid = math.sqrt(np.sum(dist2_mu) / (k - 1))

    # 3. segment 首尾寬度
    segment_width = float(np.linalg.norm(pts[-1] - pts[0]))

    # 4. 圓擬合半徑
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

    # 5. boundary regularity
    step_vec = np.diff(pts, axis=0)
    step_dist = np.sqrt(np.sum(step_vec ** 2, axis=1))
    boundary_std_dev = float(np.std(step_dist, ddof=1)) if step_dist.size >= 2 else 0.0

    # 6. mean curvature
    mean_curvature = 0.0
    if k >= 3:
        curvatures = []
        for t in range(1, k - 1):
            A_pt, B_pt, C_pt = pts[t - 1], pts[t], pts[t + 1]
            dAB = np.linalg.norm(B_pt - A_pt)
            dBC = np.linalg.norm(C_pt - B_pt)
            dAC = np.linalg.norm(C_pt - A_pt)
            area2 = abs(
                (B_pt[0] - A_pt[0]) * (C_pt[1] - A_pt[1])
                - (B_pt[1] - A_pt[1]) * (C_pt[0] - A_pt[0])
            )
            denom = dAB * dBC * dAC
            curvatures.append((2.0 * area2 / denom) if denom > EPS else 0.0)
        mean_curvature = float(np.mean(curvatures))

    # 7. mean angular difference
    mean_angular_difference = 0.0
    if k >= 3:
        betas = []
        for t in range(1, k - 1):
            v1 = pts[t - 1] - pts[t]
            v2 = pts[t + 1] - pts[t]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

            if n1 > EPS and n2 > EPS:
                cos_beta = np.dot(v1, v2) / (n1 * n2)
                cos_beta = np.clip(cos_beta, -1.0, 1.0)
                betas.append(math.acos(cos_beta))
            else:
                betas.append(0.0)

        mean_angular_difference = float(np.mean(betas))

    # 8, 9. line fitting min/max error
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

    # 10. RANSAC line inlier ratio
    ransac_inlier_ratio = 0.0
    if k >= 2:
        best_inlier = 0
        dist_thr = 0.02
        max_iter = min(30, k * (k - 1) // 2)

        for _ in range(max_iter):
            pair = np.random.choice(k, 2, replace=False)
            p1, p2 = pts[pair[0]], pts[pair[1]]
            v = p2 - p1
            nv = np.linalg.norm(v)

            if nv < EPS:
                continue

            d = np.abs(
                (pts[:, 0] - p1[0]) * v[1]
                - (pts[:, 1] - p1[1]) * v[0]
            ) / nv

            best_inlier = max(best_inlier, int(np.sum(d < dist_thr)))

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
# 4. AdaBoost prediction
# =========================================================
def stump_predict_value(x, theta, s):
    """單一 decision stump 輸出 +1 或 -1。"""
    return 1.0 if s * (x - theta) >= 0 else -1.0


def adaboost_score_single(feats):
    """計算 AdaBoost 強分類器分數。"""
    score = 0.0
    for alpha, stump in zip(ALPHA_LEG, STUMPS_LEG):
        j, theta, s = stump
        vote = stump_predict_value(feats[int(j)], theta, s)
        score += alpha * vote
    return score


# =========================================================
# 5. 兩腳合併成人
# =========================================================
def pair_legs_to_people(detected_legs, max_leg_distance=0.6):
    """把距離足夠近的兩個 leg segment centroid 合併成人中心。"""
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
                'leg_distance': best_dist
            })

    return people


# =========================================================
# 6. EKF functions
# State X = [x, y, v, theta, omega]
# =========================================================
def normalize_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def ekf_predict_ctrv(X, P, Q, dt):
    """CTRV 模型 EKF prediction。"""
    v = X[2]
    th = X[3]
    om = X[4]

    if abs(om) < 1e-3:
        X_pred = np.array([
            X[0] + v * math.cos(th) * dt,
            X[1] + v * math.sin(th) * dt,
            X[2],
            X[3],
            X[4]
        ], dtype=np.float64)

        F = np.eye(5)
        F[0, 2] = math.cos(th) * dt
        F[0, 3] = -v * math.sin(th) * dt
        F[1, 2] = math.sin(th) * dt
        F[1, 3] = v * math.cos(th) * dt

    else:
        X_pred = np.array([
            X[0] + (v / om) * (math.sin(th + om * dt) - math.sin(th)),
            X[1] + (v / om) * (-math.cos(th + om * dt) + math.cos(th)),
            X[2],
            X[3] + om * dt,
            X[4]
        ], dtype=np.float64)

        F = np.eye(5)
        F[0, 2] = (math.sin(th + om * dt) - math.sin(th)) / om
        F[0, 3] = (v / om) * (math.cos(th + om * dt) - math.cos(th))
        F[0, 4] = (v * dt * math.cos(th + om * dt) / om) - \
                  (v * (math.sin(th + om * dt) - math.sin(th)) / (om ** 2))
        F[1, 2] = (-math.cos(th + om * dt) + math.cos(th)) / om
        F[1, 3] = (v / om) * (math.sin(th + om * dt) - math.sin(th))
        F[1, 4] = (v * dt * math.sin(th + om * dt) / om) - \
                  (v * (-math.cos(th + om * dt) + math.cos(th)) / (om ** 2))
        F[3, 4] = dt

    P_pred = F @ P @ F.T + Q
    X_pred[3] = normalize_angle(X_pred[3])
    return X_pred, P_pred


def ekf_update_pos(X, P, z, R):
    """EKF update，只用位置 measurement z=[x,y] 修正。"""
    H = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ], dtype=np.float64)

    z_hat = X[0:2]
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    innovation = z - z_hat
    X_upd = X + K @ innovation
    P_upd = (np.eye(5) - K @ H) @ P

    X_upd[3] = normalize_angle(X_upd[3])
    return X_upd, P_upd


def predict_future(X, P, Q, dt, steps_future):
    """往未來 predict 多步，用於畫預測點與 covariance。"""
    X_fut = X.copy()
    P_fut = P.copy()

    for _ in range(steps_future):
        X_fut, P_fut = ekf_predict_ctrv(X_fut, P_fut, Q, dt)

    return X_fut, P_fut


def covariance_ellipse_points(mu, Sigma, sigma_scale=1, num_points=80):
    """根據 2D covariance 畫 sigma ellipse。"""
    Sigma = (Sigma + Sigma.T) / 2.0

    eigvals, eigvecs = np.linalg.eig(Sigma)
    eigvals = np.maximum(eigvals, 0.0)

    angles = np.linspace(0, 2.0 * math.pi, num_points)
    circle = np.array([np.cos(angles), np.sin(angles)])

    ellipse = eigvecs @ np.diag(np.sqrt(eigvals) * sigma_scale) @ circle
    ellipse[0, :] += mu[0]
    ellipse[1, :] += mu[1]

    return ellipse.T


# =========================================================
# 7. Collision Risk: TCPA / DCPA + Monte Carlo
# =========================================================
def velocity_from_state(X):
    """把 CTRV state 的 speed, theta 轉成 vx, vy。"""
    speed = X[2]
    theta = X[3]
    return np.array([
        speed * math.cos(theta),
        speed * math.sin(theta)
    ], dtype=np.float64)


def compute_tcpa_dcpa(position, velocity, horizon=1.0):
    """
    TCPA: 未來幾秒後，人與機器人距離最近。
    DCPA: 在 TCPA 時，最近距離是多少。
    """
    EPS = 1e-9
    v_norm2 = np.dot(velocity, velocity)

    if v_norm2 < EPS:
        return 0.0, np.linalg.norm(position)

    tcpa = -np.dot(position, velocity) / v_norm2
    tcpa = np.clip(tcpa, 0.0, horizon)

    closest_point = position + velocity * tcpa
    dcpa = np.linalg.norm(closest_point)

    return tcpa, dcpa


def estimate_collision_probability_tcpa(X, P, collision_radius=0.7, horizon=1.0, num_samples=300):
    """
    用 Monte Carlo 估計未來 horizon 秒內，人軌跡進入機器人安全區的機率。
    不需要 scipy，自己用 eigen decomposition 抽 Gaussian samples。
    """
    pos_mean = X[0:2]
    vel_mean = velocity_from_state(X)
    tcpa_mean, dcpa_mean = compute_tcpa_dcpa(pos_mean, vel_mean, horizon=horizon)

    # 保證 covariance 對稱且穩定
    P_safe = (P + P.T) / 2.0
    P_safe = P_safe + np.eye(5) * 1e-6

    try:
        eigvals, eigvecs = np.linalg.eigh(P_safe)
        eigvals = np.maximum(eigvals, 0.0)
        A = eigvecs @ np.diag(np.sqrt(eigvals))
    except np.linalg.LinAlgError:
        A = np.eye(5) * 1e-3

    collision_count = 0

    for _ in range(num_samples):
        # X_sample ~ N(X, P)
        X_sample = X + A @ np.random.randn(5)

        pos = X_sample[0:2]
        vel = velocity_from_state(X_sample)

        _, dcpa = compute_tcpa_dcpa(pos, vel, horizon=horizon)
        approaching_speed = -np.dot(pos, vel) / (np.linalg.norm(pos) + 1e-9)

        # 條件：未來 horizon 內進入安全區，而且正在靠近機器人
        if dcpa <= collision_radius and approaching_speed > 0:
            collision_count += 1

    prob = collision_count / float(num_samples)
    return prob, tcpa_mean, dcpa_mean


# =========================================================
# 8. ROS node
# =========================================================
class RealTimeLegEKFTracker:

    def get_color_by_track_id(self, track_id):

        """
        根據 track ID 回傳固定顏色。
        同一個人只要 ID 不變，顏色就會固定。
        """
        colors = [
            (0.0, 1.0, 0.0),   # green
            (1.0, 0.0, 0.0),   # red
            (0.0, 0.4, 1.0),   # blue
            (1.0, 1.0, 0.0),   # yellow
            (1.0, 0.0, 1.0),   # magenta
            (0.0, 1.0, 1.0),   # cyan
            (1.0, 0.5, 0.0),   # orange
            (0.6, 0.2, 1.0),   # purple
        ]

        return colors[(track_id - 1) % len(colors)]
    def __init__(self):
        rospy.init_node('realtime_leg_ekf_tracker', anonymous=True)

        # 基本感測與偵測參數
        self.scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.05)
        self.segment_threshold = rospy.get_param('~segment_threshold', 0.1)
        self.max_leg_distance = rospy.get_param('~max_leg_distance', 0.6)

        # EKF / prediction 參數
        self.dt = rospy.get_param('~dt', 0.1)
        self.predict_time = rospy.get_param('~predict_time', 0.3)
        self.steps_future = int(round(self.predict_time / self.dt))
        self.dist_gate = rospy.get_param('~dist_gate', 1.0)
        self.max_missed = rospy.get_param('~max_missed', 10)
        self.sigma_scale = rospy.get_param('~sigma_scale',1)

        # 風險估計參數
        self.risk_horizon = rospy.get_param('~risk_horizon', 1.0)
        self.robot_radius = rospy.get_param('~robot_radius', 0.25)
        self.human_radius = rospy.get_param('~human_radius', 0.25)
        self.safety_margin = rospy.get_param('~safety_margin', 0.20)
        self.collision_radius = self.robot_radius + self.human_radius + self.safety_margin
        self.num_mc = rospy.get_param('~num_mc', 300)
        self.stop_threshold = rospy.get_param('~stop_threshold', 0.7)
        self.slow_threshold = rospy.get_param('~slow_threshold', 0.4)

        self.debug = rospy.get_param('~debug', False)

        self.Q = np.diag([0.02, 0.02, 0.08, 0.08, 0.08])
        self.R = np.eye(2) * 0.2

        self.tracks = []
        self.next_track_id = 1

        self.sub = rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.marker_pub = rospy.Publisher('/people_markers', MarkerArray, queue_size=1)

        rospy.loginfo("Real-time AdaBoost + EKF + Risk tracker is ready.")
        rospy.loginfo("No scipy is used.")
        rospy.loginfo("Scan topic: %s", self.scan_topic)
        rospy.loginfo("Predict time: %.2f s", self.predict_time)
        rospy.loginfo("Risk horizon: %.2f s", self.risk_horizon)
        rospy.loginfo("Collision radius: %.2f m", self.collision_radius)

    def scan_callback(self, msg):
        """每收到一筆 LaserScan，就做 detection -> tracking -> risk visualization。"""
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
            return

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        xy_points = np.column_stack((x, y))

        clusters = segment_lidar(xy_points, threshold=self.segment_threshold)

        detected_legs = []

        # 對每個 segment 算特徵並用 AdaBoost 判斷是不是腳
        for i, pts in enumerate(clusters):
            feats = extract_10_features(pts)

            if feats is None or not np.any(feats != 0):
                continue

            score = adaboost_score_single(feats)
            is_leg = score > THR_LEG

            cx = float(np.mean(pts[:, 0]))
            cy = float(np.mean(pts[:, 1]))

            if is_leg:
                detected_legs.append({
                    'id': i,
                    'cx': cx,
                    'cy': cy,
                    'score': score
                })

        people_positions = pair_legs_to_people(
            detected_legs,
            max_leg_distance=self.max_leg_distance
        )

        if len(people_positions) > 0:
            person_meas = np.array([[p['x'], p['y']] for p in people_positions], dtype=np.float64)
        else:
            person_meas = np.empty((0, 2), dtype=np.float64)

        self.update_tracks(person_meas)
        self.publish_tracking_markers(msg.header.frame_id)

    def update_tracks(self, person_meas):
        """EKF prediction + assignment + EKF update + track birth/death。"""
        # 1. 先 predict 所有舊 track
        for tr in self.tracks:
            tr['X'], tr['P'] = ekf_predict_ctrv(tr['X'], tr['P'], self.Q, self.dt)

        N = len(self.tracks)
        M = person_meas.shape[0]

        matched_pairs = []
        unmatched_tracks = set(range(N))
        unmatched_meas = set(range(M))

        # 2. 建立 cost matrix，做全域最佳配對
        if N > 0 and M > 0:
            cost = np.full((N, M), np.inf)

            for i, tr in enumerate(self.tracks):
                pred_pos = tr['X'][0:2]

                for j in range(M):
                    d = np.linalg.norm(person_meas[j] - pred_pos)
                    if d < self.dist_gate:
                        cost[i, j] = d

            matched_pairs = self.assignment_without_scipy(cost)

            for i, j in matched_pairs:
                unmatched_tracks.discard(i)
                unmatched_meas.discard(j)

        # 3. 成功配對的 track 做 update
        for i, j in matched_pairs:
            z = person_meas[j]
            self.tracks[i]['X'], self.tracks[i]['P'] = ekf_update_pos(
                self.tracks[i]['X'],
                self.tracks[i]['P'],
                z,
                self.R
            )

            self.tracks[i]['age'] += 1
            self.tracks[i]['miss'] = 0
            self.tracks[i]['history'].append(self.tracks[i]['X'][0:2].copy())

        # 4. 沒配到的 track 保留 prediction
        for i in unmatched_tracks:
            self.tracks[i]['age'] += 1
            self.tracks[i]['miss'] += 1
            self.tracks[i]['history'].append(self.tracks[i]['X'][0:2].copy())

        # 5. 新 measurement 生成新 track
        for j in unmatched_meas:
            z = person_meas[j]

            X0 = np.array([z[0], z[1], 0.0, 0.0, 1e-4], dtype=np.float64)
            P0 = np.eye(5) * 0.1

            self.tracks.append({
                'id': self.next_track_id,
                'X': X0,
                'P': P0,
                'age': 1,
                'miss': 0,
                'history': [z.copy()]
            })

            self.next_track_id += 1

        # 6. 刪除太久沒看到的 track
        self.tracks = [tr for tr in self.tracks if tr['miss'] <= self.max_missed]

    def assignment_without_scipy(self, cost):
        """不用 scipy，窮舉所有配對，找總成本最小的 assignment。"""
        matched = []

        if cost.size == 0:
            return matched

        N, M = cost.shape
        best_cost = float('inf')
        best_pairs = []

        if N <= M:
            for meas_indices in combinations(range(M), N):
                for perm in permutations(meas_indices):
                    total_cost = 0.0
                    pairs = []
                    valid = True

                    for i in range(N):
                        j = perm[i]
                        if not np.isfinite(cost[i, j]):
                            valid = False
                            break
                        total_cost += cost[i, j]
                        pairs.append((i, j))

                    if valid and total_cost < best_cost:
                        best_cost = total_cost
                        best_pairs = pairs
        else:
            for track_indices in combinations(range(N), M):
                for perm in permutations(track_indices):
                    total_cost = 0.0
                    pairs = []
                    valid = True

                    for j in range(M):
                        i = perm[j]
                        if not np.isfinite(cost[i, j]):
                            valid = False
                            break
                        total_cost += cost[i, j]
                        pairs.append((i, j))

                    if valid and total_cost < best_cost:
                        best_cost = total_cost
                        best_pairs = pairs

        return best_pairs

    def publish_delete_all(self, frame_id):
        arr = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL
        arr.markers.append(marker)
        self.marker_pub.publish(arr)

    def publish_tracking_markers(self, frame_id):
        """在 RViz 顯示 robot、人、未來位置、sigma、風險文字。"""
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = frame_id
        delete_all.header.stamp = rospy.Time.now()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        marker_id = 0

        # 機器人本體：白色球
        m_robot = Marker()
        m_robot.header.frame_id = frame_id
        m_robot.header.stamp = rospy.Time.now()
        m_robot.ns = "robot_position"
        m_robot.id = marker_id
        marker_id += 1
        m_robot.type = Marker.SPHERE
        m_robot.action = Marker.ADD
        m_robot.pose.position.x = 0.0
        m_robot.pose.position.y = 0.0
        m_robot.pose.position.z = 0.15
        m_robot.pose.orientation.w = 1.0
        m_robot.scale.x = 0.25
        m_robot.scale.y = 0.25
        m_robot.scale.z = 0.25
        m_robot.color.r = 1.0
        m_robot.color.g = 1.0
        m_robot.color.b = 1.0
        m_robot.color.a = 1.0
        m_robot.lifetime = rospy.Duration(0.2)
        arr.markers.append(m_robot)

        # 機器人安全區：白色半透明圓
        m_safe = Marker()
        m_safe.header.frame_id = frame_id
        m_safe.header.stamp = rospy.Time.now()
        m_safe.ns = "robot_collision_radius"
        m_safe.id = marker_id
        marker_id += 1
        m_safe.type = Marker.LINE_STRIP
        m_safe.action = Marker.ADD
        m_safe.scale.x = 0.02
        m_safe.color.r = 1.0
        m_safe.color.g = 1.0
        m_safe.color.b = 1.0
        m_safe.color.a = 0.8
        m_safe.lifetime = rospy.Duration(0.2)

        for a in np.linspace(0.0, 2.0 * math.pi, 80):
            pt = Point()
            pt.x = self.collision_radius * math.cos(a)
            pt.y = self.collision_radius * math.sin(a)
            pt.z = 0.05
            m_safe.points.append(pt)

        arr.markers.append(m_safe)

        for tr in self.tracks:
            Xnow = tr['X']
            Pnow = tr['P']

            Xfut, Pfut = predict_future(
                Xnow,
                Pnow,
                self.Q,
                self.dt,
                self.steps_future
            )

            # 計算碰撞機率
            collision_prob, tcpa, dcpa = estimate_collision_probability_tcpa(
                Xnow,
                Pnow,
                collision_radius=self.collision_radius,
                horizon=self.risk_horizon,
                num_samples=self.num_mc
            )

            if collision_prob >= self.stop_threshold:
                decision = "STOP"
                text_color = (1.0, 0.0, 0.0)
            elif collision_prob >= self.slow_threshold:
                decision = "SLOW"
                text_color = (1.0, 0.5, 0.0)
            else:
                decision = "GO"
                text_color = (0.0, 1.0, 0.0)

            # 目前人的位置：綠色圓柱
            m_now = Marker()
            m_now.header.frame_id = frame_id
            m_now.header.stamp = rospy.Time.now()
            m_now.ns = "ekf_people_now"
            m_now.id = marker_id
            marker_id += 1
            m_now.type = Marker.CYLINDER
            m_now.action = Marker.ADD
            m_now.pose.position.x = Xnow[0]
            m_now.pose.position.y = Xnow[1]
            m_now.pose.position.z = 0.25
            m_now.pose.orientation.w = 1.0
            m_now.scale.x = 0.22
            m_now.scale.y = 0.22
            m_now.scale.z = 0.5
            person_color = self.get_color_by_track_id(tr['id'])
            m_now.color.r = person_color[0]
            m_now.color.g = person_color[1]
            m_now.color.b = person_color[2]
            m_now.color.a = 0.9
            m_now.lifetime = rospy.Duration(0.2)
            arr.markers.append(m_now)

            # 預測位置：橘色球
            m_future = Marker()
            m_future.header.frame_id = frame_id
            m_future.header.stamp = rospy.Time.now()
            m_future.ns = "ekf_people_future"
            m_future.id = marker_id
            marker_id += 1
            m_future.type = Marker.SPHERE
            m_future.action = Marker.ADD
            m_future.pose.position.x = Xfut[0]
            m_future.pose.position.y = Xfut[1]
            m_future.pose.position.z = 0.35
            m_future.pose.orientation.w = 1.0
            m_future.scale.x = 0.18
            m_future.scale.y = 0.18
            m_future.scale.z = 0.18
            m_future.color.r = person_color[0]
            m_future.color.g = person_color[1]
            m_future.color.b = person_color[2]
            m_future.color.a = 0.95
            m_future.lifetime = rospy.Duration(0.2)
            arr.markers.append(m_future)

            # 預測 covariance：淡藍色 sigma ellipse
            m_sigma = Marker()
            m_sigma.header.frame_id = frame_id
            m_sigma.header.stamp = rospy.Time.now()
            m_sigma.ns = "ekf_future_sigma"
            m_sigma.id = marker_id
            marker_id += 1
            m_sigma.type = Marker.LINE_STRIP
            m_sigma.action = Marker.ADD
            m_sigma.scale.x = 0.025
            m_sigma.color.r = person_color[0]
            m_sigma.color.g = person_color[1]
            m_sigma.color.b = person_color[2]
            m_sigma.color.a = 0.6
            m_sigma.lifetime = rospy.Duration(0.2)

            ellipse_pts = covariance_ellipse_points(
                Xfut[0:2],
                Pfut[0:2, 0:2],
                sigma_scale=self.sigma_scale,
                num_points=80
            )

            for p in ellipse_pts:
                pt = Point()
                pt.x = p[0]
                pt.y = p[1]
                pt.z = 0.12
                m_sigma.points.append(pt)

            if len(ellipse_pts) > 0:
                pt0 = Point()
                pt0.x = ellipse_pts[0, 0]
                pt0.y = ellipse_pts[0, 1]
                pt0.z = 0.12
                m_sigma.points.append(pt0)

            arr.markers.append(m_sigma)

            # 現在到未來的方向線：黃色
            m_line = Marker()
            m_line.header.frame_id = frame_id
            m_line.header.stamp = rospy.Time.now()
            m_line.ns = "ekf_prediction_line"
            m_line.id = marker_id
            marker_id += 1
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.scale.x = 0.03
            m_line.color.r = person_color[0]
            m_line.color.g = person_color[1]
            m_line.color.b = person_color[2]
            m_line.color.a = 0.9
            m_line.lifetime = rospy.Duration(0.2)

            p1 = Point()
            p1.x = Xnow[0]
            p1.y = Xnow[1]
            p1.z = 0.1

            p2 = Point()
            p2.x = Xfut[0]
            p2.y = Xfut[1]
            p2.z = 0.1

            m_line.points.append(p1)
            m_line.points.append(p2)
            arr.markers.append(m_line)

            # 風險文字
            m_txt = Marker()
            m_txt.header.frame_id = frame_id
            m_txt.header.stamp = rospy.Time.now()
            m_txt.ns = "ekf_people_text"
            m_txt.id = marker_id
            marker_id += 1
            m_txt.type = Marker.TEXT_VIEW_FACING
            m_txt.action = Marker.ADD
            m_txt.pose.position.x = Xnow[0]
            m_txt.pose.position.y = Xnow[1]
            m_txt.pose.position.z = 0.75
            m_txt.pose.orientation.w = 1.0
            m_txt.scale.z = 0.25
            m_txt.color.r = text_color[0]
            m_txt.color.g = text_color[1]
            m_txt.color.b = text_color[2]
            m_txt.color.a = 1.0
            m_txt.lifetime = rospy.Duration(0.2)
            m_txt.text = "Risk {:.1f}%".format(
                tr['id'],
                collision_prob * 100.0,
                decision,
                tcpa,
                dcpa
            )
            arr.markers.append(m_txt)

        self.marker_pub.publish(arr)


if __name__ == '__main__':
    try:
        RealTimeLegEKFTracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass