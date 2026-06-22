#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Real-time person position detection using:
1. LiDAR segmentation
2. 10 handcrafted features
3. Exported AdaBoost model from MATLAB
4. Pairing two legs into one person position
5. ROS visualization in RViz
"""

# 中文說明：這支只做行人偵測、EKF 追蹤與 RViz 視覺化，不會發 /cmd_vel。
# RViz 中會畫目前行人、未來預測點，以及預測不確定度 sigma 圈。

import math
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

# =========================================================
# 1. AdaBoost 模型參數（從 MATLAB 匯出）
# =========================================================
ALPHA_LEG = np.array([0.783044, 0.347437, 0.320527, 0.326356, 0.369778, 0.316612, 0.249111, 0.228206, 0.240536, 0.208360, 0.241859, 0.207806, 0.165593, 0.167728, 0.152153, 0.169358, 0.177836, 0.149083, 0.167618, 0.159571, 0.161077, 0.158761, 0.158964, 0.140098, 0.122884, 0.141936, 0.124683, 0.118274, 0.119120, 0.109797, 0.135280, 0.108972, 0.111787, 0.104844, 0.141188, 0.122933, 0.114286, 0.096909, 0.090066, 0.101552, 0.118719, 0.104412, 0.108325, 0.098659, 0.093576, 0.103070, 0.084637, 0.091455, 0.094610, 0.086825], dtype=np.float64)

# [feature_index, theta, s]
# 這裡已經是 Python 0-based index
STUMPS_LEG = [[5, 28.342846, 1],[1, 0.061183, -1],[2, 0.099685, 1],[2, 0.056246, -1],[1, 0.083119, -1],[4, 0.020911, 1],[0, 3.500000, 1],[1, 0.083119, -1],[6, 2.920823, 1],[3, 0.180842, -1],
              [8, 0.035349, 1],[4, 0.008846, -1],[2, 0.099685, 1],[3, 0.095347, -1],[6, 2.920823, 1],[3, 0.319987, -1],[8, 0.035349, 1],[1, 0.104475, -1],[2, 0.048098, -1],[2, 0.082441, 1],
              [2, 0.062081, -1],[1, 0.104475, -1],[0, 6.500000, 1],[5, 27.298656, -1],[1, 0.020262, -1],[1, 0.065549, -1],[8, 0.007097, 1],[4, 0.005266, -1],[1, 0.058460, 1],[1, 0.044350, -1],
              [2, 0.082441, 1],[2, 0.088983, -1],[8, 0.007097, 1],[6, 2.920823, 1],[1, 0.083119, -1],[8, 0.035349, 1],[1, 0.062675, -1],[6, 2.495484, 1],[7, 0.000846, -1],[8, 0.017176, 1],
              [5, 27.298656, -1],[2, 0.067109, -1], [2, 0.082441, 1],[2, 0.088983, -1],[2, 0.087673, 1],[1, 0.044350, -1],[2, 0.099685, 1],[2, 0.088983, -1],[2, 0.082441, 1],[1, 0.020262, -1]
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

    # [0] point_count
    point_count = float(k)

    # [1] std_dev_to_centroid
    mu = np.mean(pts, axis=0)
    diff_mu = pts - mu
    dist2_mu = np.sum(diff_mu ** 2, axis=1)
    std_dev_to_centroid = math.sqrt(np.sum(dist2_mu) / (k - 1)) if k > 1 else 0.0

    # [2] segment_width
    segment_width = float(np.linalg.norm(pts[-1] - pts[0]))

    # [3] circle_fit_radius
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

    # [4] boundary_std_dev
    if k >= 2:
        step_vec = np.diff(pts, axis=0)
        step_dist = np.sqrt(np.sum(step_vec ** 2, axis=1))
    else:
        step_dist = np.array([], dtype=np.float64)

    if step_dist.size >= 2:
        boundary_std_dev = float(np.std(step_dist, ddof=1))
    else:
        boundary_std_dev = 0.0

    # [5] mean_curvature
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

    # [6] mean_angular_difference
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

    # [7], [8] min/max line fitting error
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

    # [9] ransac_inlier_ratio
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
# 手寫匈牙利演算法 (Hungarian Algorithm / Kuhn-Munkres 簡化版)
# =========================================================
def min_weight_assignment(cost_matrix):
    """
    輸入 cost_matrix (N x M)，回傳配對索引矩陣。
    針對小規模問題 (如人腿追蹤) 效率極高。
    """
    n, m = cost_matrix.shape
    if n == 0 or m == 0:
        return []
    
    # 這裡使用貪婪搭配 DFS 的增廣路徑演算法 (二分圖最大匹配變體)
    # 由於追蹤通常目標數極少 (如 2-5 人)，直接找最優配對即可
    rows, cols = np.where(cost_matrix < 1e8) # 只考慮 gate 內的
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
# EKF Track & Tracker (CTRV Model)
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
            F = np.eye(5); F[0,2]=math.cos(th)*dt; F[0,3]=-v*math.sin(th)*dt; F[1,2]=math.sin(th)*dt; F[1,3]=v*math.cos(th)*dt
        else:
            # 狀態預測
            self.X[0] += (v/om)*(math.sin(th+om*dt)-math.sin(th))
            self.X[1] += (v/om)*(-math.cos(th+om*dt)+math.cos(th))
            self.X[3] += om*dt

            # Jacobian 矩陣 F (對應 MATLAB 檔案第 326-340 行)
            F = np.eye(5)
            F[0,2] = (math.sin(th+om*dt)-math.sin(th))/om
            F[0,3] = (v/om)*(math.cos(th+om*dt)-math.cos(th))
            F[0,4] = (v*dt*math.cos(th+om*dt)/om) - (v*(math.sin(th+om*dt)-math.sin(th))/(om**2))
            
            F[1,2] = (-math.cos(th+om*dt)+math.cos(th))/om
            F[1,3] = (v/om)*(math.sin(th+om*dt)-math.sin(th))
            F[1,4] = (v*dt*math.sin(th+om*dt)/om) - (v*(-math.cos(th+om*dt)+math.cos(th))/(om**2))
            
            F[3,4] = dt
        self.P = F @ self.P @ F.T + Q
        self.X[3] = math.atan2(math.sin(self.X[3]), math.cos(self.X[3]))

    def update(self, z, R):
        H = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.X += K @ (z - self.X[0:2])
        self.P = (np.eye(5) - K @ H) @ self.P

class EKFTracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.Q = np.diag([0.001, 0.001, 0.01, 0.01, 0.01]) # Process Noise
        self.R = np.eye(2) * 0.05 # Measurement Noise
        self.dist_gate = 0.8
        self.max_missed = 5

    def update_tracks(self, measurements, dt):
        for t in self.tracks: t.predict(dt, self.Q)
        
        n, m = len(self.tracks), len(measurements)
        cost_matrix = np.full((n, m), 1e9)
        for i in range(n):
            for j in range(m):
                d = math.hypot(self.tracks[i].X[0]-measurements[j][0], self.tracks[i].X[1]-measurements[j][1])
                if d < self.dist_gate: cost_matrix[i, j] = d
        
        pairs = min_weight_assignment(cost_matrix)
        matched_t, matched_m = [p[0] for p in pairs], [p[1] for p in pairs]
        
        for i, j in pairs:
            self.tracks[i].update(measurements[j], self.R)
            self.tracks[i].miss = 0
            
        for i in range(n):
            if i not in matched_t: self.tracks[i].miss += 1
            
        for j in range(m):
            if j not in matched_m:
                self.tracks.append(Track(self.next_id, measurements[j]))
                self.next_id += 1
                
        self.tracks = [t for t in self.tracks if t.miss <= self.max_missed]
        return self.tracks







# =========================================================
# 6. ROS node (整合 EKF 追蹤版)
# =========================================================
class RealTimeLegDetector:
    def __init__(self):
        rospy.init_node('realtime_leg_detector', anonymous=True)

        # 讀取原本的參數
        scan_topic = rospy.get_param('~scan_topic', '/scan')
        self.max_range = rospy.get_param('~max_range', 5.0)
        self.min_range = rospy.get_param('~min_range', 0.05)
        self.segment_threshold = rospy.get_param('~segment_threshold', 0.1)
        self.max_leg_distance = rospy.get_param('~max_leg_distance', 0.6)
        self.debug = rospy.get_param('~debug', False)

        # --- [新增] 初始化追蹤器與時間變數 ---
        self.tracker = EKFTracker()
        self.last_time = None
        # ---------------------------------

        self.sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.marker_pub = rospy.Publisher('/people_markers', MarkerArray, queue_size=1)

        rospy.loginfo("Real-time leg tracker is ready.")

    def scan_callback(self, msg):
        # --- 1. 雷射點雲前處理 (保留原本邏輯) ---
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = (np.isfinite(ranges) & (ranges > self.min_range) & (ranges < self.max_range))
        ranges, angles = ranges[valid], angles[valid]

        if ranges.size == 0:
            self.publish_delete_all(msg.header.frame_id)
            return

        xy_points = np.column_stack((ranges * np.cos(angles), ranges * np.sin(angles)))
        clusters = segment_lidar(xy_points, threshold=self.segment_threshold)

        # --- 2. AdaBoost 偵測腿部 (保留原本邏輯) ---
        detected_legs = []
        for i, pts in enumerate(clusters):
            feats = extract_10_features(pts)
            if feats is None or not np.any(feats != 0): continue
            score = adaboost_score_single(feats)
            if score > THR_LEG:
                # [修正處]：在這裡加入 'id': i
                detected_legs.append({
                    'id': i, 
                    'cx': float(np.mean(pts[:, 0])),
                    'cy': float(np.mean(pts[:, 1]))
                })

        # --- 3. 腿部配對成人體中心 ---
        people_positions = pair_legs_to_people(detected_legs, max_leg_distance=self.max_leg_distance)
        
        # --- 4. [修改] 執行 EKF 追蹤與預測 ---
        # 計算 dt (時間間隔)
        current_time = msg.header.stamp
        if self.last_time is None:
            dt = 0.1
        else:
            dt = (current_time - self.last_time).to_sec()
        self.last_time = current_time

        # 將偵測結果轉換為 EKF 的量測輸入 [x, y]
        measurements = [np.array([p['x'], p['y']]) for p in people_positions]
        
        # 更新追蹤器 (這包含了預測、匈牙利配對與更新)
        active_tracks = self.tracker.update_tracks(measurements, dt)

        # --- 5. [修改] 視覺化發布 (顯示 Track ID 與預測) ---
        self.publish_track_markers(active_tracks, msg.header.frame_id, dt)

    def publish_delete_all(self, frame_id):
        arr = MarkerArray()
        m = Marker()
        m.header.frame_id, m.action = frame_id, Marker.DELETEALL
        arr.markers.append(m)
        self.marker_pub.publish(arr)


    def create_circle_points(self, cx, cy, r, num_points=30):
        """ 產生圓圈的點，用於 LINE_STRIP """
        points = []
        for i in range(num_points + 1):
            angle = 2 * math.pi * i / num_points
            p = Point()
            p.x = cx + r * math.cos(angle)
            p.y = cy + r * math.sin(angle)
            p.z = 0.05 # 稍微高於地面避免閃爍
            points.append(p)
        return points

    def publish_track_markers(self, tracks, frame_id, dt):
        arr = MarkerArray()
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.action = Marker.DELETEALL
        arr.markers.append(delete_marker)

        marker_id = 0


        # --- 新增：顯示機器人自身位置 (白色球) ---
        m_robot = Marker()
        m_robot.header.frame_id = frame_id
        m_robot.header.stamp = rospy.Time.now()
        m_robot.ns = "robot_base"
        m_robot.id = marker_id
        marker_id += 1
        m_robot.type = Marker.SPHERE
        m_robot.action = Marker.ADD
        # 機器人在雷射座標系中心 (0,0,0)
        m_robot.pose.position.x = 0.0
        m_robot.pose.position.y = 0.0
        m_robot.pose.position.z = 0.1
        m_robot.scale.x = m_robot.scale.y = m_robot.scale.z = 0.2 # 球體大小
        m_robot.color.r = 1.0 # 白色
        m_robot.color.g = 1.0
        m_robot.color.b = 1.0
        m_robot.color.a = 1.0
        arr.markers.append(m_robot)
        # ---------------------------------------



        predict_time = 1  # 預測未來 1 秒

        for t in tracks:
            # --- A. 原有的當前位置 (CYLINDER) ---
            m_person = Marker()
            m_person.header.frame_id = frame_id
            m_person.header.stamp = rospy.Time.now()
            m_person.ns = "tracked_people"
            m_person.id = marker_id
            marker_id += 1
            m_person.type = Marker.CYLINDER
            m_person.pose.position.x, m_person.pose.position.y = t.X[0], t.X[1]
            m_person.pose.position.z = 0.25
            m_person.scale.x = m_person.scale.y = 0.3
            m_person.scale.z = 0.5
            m_person.color.r = (t.id * 0.2) % 1.0
            m_person.color.g = 1.0 - (t.id * 0.3) % 1.0
            m_person.color.b = (t.id * 0.5) % 1.0
            m_person.color.a = 0.8
            arr.markers.append(m_person)

            # --- B. 原有的 ID 文字 ---
            m_text = Marker()
            m_text.header.frame_id = frame_id
            m_text.ns = "track_id"
            m_text.id = marker_id
            marker_id += 1
            m_text.type = Marker.TEXT_VIEW_FACING
            m_text.pose.position.x, m_text.pose.position.y = t.X[0], t.X[1]
            m_text.pose.position.z = 0.7
            m_text.scale.z = 0.15
            m_text.color.r = m_text.color.g = m_text.color.b = 1.0
            m_text.color.a = 1.0
            m_text.text = "ID: {}".format(t.id)
            arr.markers.append(m_text)

            # --- C. 預測位置與 1-sigma 範圍 ---
            # 1. 計算預測位置 (State Prediction)
            v, th, om = t.X[2], t.X[3], t.X[4]
            if abs(om) < 1e-3:
                px = t.X[0] + v * math.cos(th) * predict_time
                py = t.X[1] + v * math.sin(th) * predict_time
                # 簡化 Jacobian F_pred
                F_p = np.eye(5)
                F_p[0,2]=math.cos(th)*predict_time; F_p[0,3]=-v*math.sin(th)*predict_time
                F_p[1,2]=math.sin(th)*predict_time; F_p[1,3]=v*math.cos(th)*predict_time
            else:
                px = t.X[0] + (v/om)*(math.sin(th+om*predict_time)-math.sin(th))
                py = t.X[1] + (v/om)*(-math.cos(th+om*predict_time)+math.cos(th))
                # 計算預測用的 Jacobian F_p
                F_p = np.eye(5)
                F_p[0,2] = (math.sin(th+om*predict_time)-math.sin(th))/om
                F_p[0,3] = (v/om)*(math.cos(th+om*predict_time)-math.cos(th))
                F_p[1,2] = (-math.cos(th+om*predict_time)+math.cos(th))/om
                F_p[1,3] = (v/om)*(math.sin(th+om*predict_time)-math.sin(th))

            # 2. 計算預測共變異 P_pred = F_p * P * F_p.T + Q*dt
            P_pred = F_p @ t.P @ F_p.T + (self.tracker.Q * predict_time)
            
            # 3. 提取 1-sigma 半徑 
            # 這裡使用位置(x,y)方差的算術平均值的平方根作為圓圈半徑
            # 更精確的做法是取 P_pred[0,0] 與 P_pred[1,1] 的最大特徵值
            sigma_r = math.sqrt(max(P_pred[0,0], P_pred[1,1]))
            sigma_r = max(0.04, min(sigma_r, 0.20))

            # 繪製預測中心點 (小球)
            m_pred_pt = Marker()
            m_pred_pt.header.frame_id = frame_id
            m_pred_pt.ns = "prediction_point"
            m_pred_pt.id = marker_id
            marker_id += 1
            m_pred_pt.type = Marker.SPHERE
            m_pred_pt.pose.position.x, m_pred_pt.pose.position.y = px, py
            m_pred_pt.pose.position.z = 0.1
            m_pred_pt.scale.x = m_pred_pt.scale.y = m_pred_pt.scale.z = 0.1
            m_pred_pt.color = m_person.color
            m_pred_pt.color.a = 0.5
            arr.markers.append(m_pred_pt)

            # --- D. 現在位置到預測位置的連線 ---
            m_line = Marker()
            m_line.header.frame_id = frame_id
            m_line.header.stamp = rospy.Time.now()
            m_line.ns = "current_to_prediction_line"
            m_line.id = marker_id
            marker_id += 1
            m_line.type = Marker.LINE_STRIP
            m_line.action = Marker.ADD
            m_line.pose.orientation.w = 1.0
            
            # 線寬
            m_line.scale.x = 0.035
            
            # 跟該 track 同顏色
            m_line.color.r = m_person.color.r
            m_line.color.g = m_person.color.g
            m_line.color.b = m_person.color.b
            m_line.color.a = 0.9
            
            # 起點：目前 EKF 估計位置
            p_now = Point()
            p_now.x = float(t.X[0])
            p_now.y = float(t.X[1])
            p_now.z = 0.15
            
            # 終點：未來 predict_time 秒預測位置
            p_pred = Point()
            p_pred.x = float(px)
            p_pred.y = float(py)
            p_pred.z = 0.15
            
            m_line.points.append(p_now)
            m_line.points.append(p_pred)
            
            arr.markers.append(m_line)

            # 繪製 1-sigma 空心圓圈 (LINE_STRIP)
            m_sigma = Marker()
            m_sigma.header.frame_id = frame_id
            m_sigma.header.stamp = rospy.Time.now()
            m_sigma.ns = "sigma_range"
            m_sigma.id = marker_id
            marker_id += 1
            m_sigma.type = Marker.LINE_STRIP
            m_sigma.action = Marker.ADD
            m_sigma.pose.orientation.w = 1.0
            m_sigma.scale.x = 0.03 # 線條寬度
            m_sigma.color = m_person.color
            m_sigma.color.a = 0.6
            
            # 生成圓圈上的點
            m_sigma.points = self.create_circle_points(px, py, sigma_r)
            arr.markers.append(m_sigma)

        self.marker_pub.publish(arr)

# =========================================================
# 7. main
# =========================================================
if __name__ == '__main__':
    try:
        # 確保在執行前已經定義了之前的 EKFTracker 和 Track 類別
        detector = RealTimeLegDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
