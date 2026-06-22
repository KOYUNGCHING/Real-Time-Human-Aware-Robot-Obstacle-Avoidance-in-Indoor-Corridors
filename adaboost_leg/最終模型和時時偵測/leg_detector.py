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

import math
import numpy as np
import rospy

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


# =========================================================
# 1. AdaBoost 模型參數（從 MATLAB 匯出）
# =========================================================
ALPHA_LEG = np.array([0.783044, 0.347437, 0.320527, 0.326356, 0.369778, 0.316612, 0.249111, 0.228206, 0.240536, 0.208360, 0.241859, 0.207806, 0.165593, 0.167728, 0.152153, 0.169358, 0.177836, 0.149083, 0.167618, 0.159571, 0.161077, 0.158761, 0.158964, 0.140098, 0.122884, 0.141936, 0.124683, 0.118274, 0.119120, 0.109797, 0.135280, 0.108972, 0.111787, 0.104844, 0.141188, 0.122933, 0.114286, 0.096909, 0.090066, 0.101552, 0.118719, 0.104412, 0.108325, 0.098659, 0.093576, 0.103070, 0.084637, 0.091455, 0.094610, 0.086825], dtype=np.float64)

# [feature_index, theta, s]
# 這裡已經是 Python 0-based index
STUMPS_LEG = [
    [5, 28.342846, 1],
    [1, 0.061183, -1],
    [2, 0.099685, 1],
    [2, 0.056246, -1],
    [1, 0.083119, -1],
    [4, 0.020911, 1],
    [0, 3.500000, 1],
    [1, 0.083119, -1],
    [6, 2.920823, 1],
    [3, 0.180842, -1],
    [8, 0.035349, 1],
    [4, 0.008846, -1],
    [2, 0.099685, 1],
    [3, 0.095347, -1],
    [6, 2.920823, 1],
    [3, 0.319987, -1],
    [8, 0.035349, 1],
    [1, 0.104475, -1],
    [2, 0.048098, -1],
    [2, 0.082441, 1],
    [2, 0.062081, -1],
    [1, 0.104475, -1],
    [0, 6.500000, 1],
    [5, 27.298656, -1],
    [1, 0.020262, -1],
    [1, 0.065549, -1],
    [8, 0.007097, 1],
    [4, 0.005266, -1],
    [1, 0.058460, 1],
    [1, 0.044350, -1],
    [2, 0.082441, 1],
    [2, 0.088983, -1],
    [8, 0.007097, 1],
    [6, 2.920823, 1],
    [1, 0.083119, -1],
    [8, 0.035349, 1],
    [1, 0.062675, -1],
    [6, 2.495484, 1],
    [7, 0.000846, -1],
    [8, 0.017176, 1],
    [5, 27.298656, -1],
    [2, 0.067109, -1],
    [2, 0.082441, 1],
    [2, 0.088983, -1],
    [2, 0.087673, 1],
    [1, 0.044350, -1],
    [2, 0.099685, 1],
    [2, 0.088983, -1],
    [2, 0.082441, 1],
    [1, 0.020262, -1]
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

        self.sub = rospy.Subscriber(scan_topic, LaserScan, self.scan_callback, queue_size=1)
        self.marker_pub = rospy.Publisher('/people_markers', MarkerArray, queue_size=1)

        rospy.loginfo("Real-time leg detector is ready.")
        rospy.loginfo("Scan topic: %s", scan_topic)
        rospy.loginfo("Threshold: %.6f", THR_LEG)

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float64)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment

        valid = (
            np.isfinite(ranges) &
            (ranges > self.min_range) &
            (ranges < self.max_range)
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

        for i, pts in enumerate(clusters):
            feats = extract_10_features(pts)
            if feats is None:
                continue

            if not np.any(feats != 0):
                continue

            score = adaboost_score_single(feats)
            is_leg = score > THR_LEG

            if self.debug:
                rospy.loginfo(
                    "seg=%d pts=%d score=%.6f leg=%s center=(%.3f, %.3f)",
                    i, pts.shape[0], score, str(is_leg),
                    np.mean(pts[:, 0]), np.mean(pts[:, 1])
                )

            if is_leg:
                detected_legs.append({
                    'id': i,
                    'cx': float(np.mean(pts[:, 0])),
                    'cy': float(np.mean(pts[:, 1])),
                    'score': score
                })

        people_positions = pair_legs_to_people(
            detected_legs,
            max_leg_distance=self.max_leg_distance
        )

        if self.debug:
            rospy.loginfo("detected legs = %d, people = %d", len(detected_legs), len(people_positions))
            for k, person in enumerate(people_positions):
                rospy.loginfo(
                    "person=%d x=%.3f y=%.3f leg_dist=%.3f",
                    k, person['x'], person['y'], person['leg_distance']
                )

        self.publish_people_markers(people_positions, msg.header.frame_id)

    def publish_delete_all(self, frame_id):
        arr = MarkerArray()
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.Time.now()
        marker.action = Marker.DELETEALL
        arr.markers.append(marker)
        self.marker_pub.publish(arr)

    def publish_people_markers(self, people_positions, frame_id):
        arr = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = frame_id
        delete_all.header.stamp = rospy.Time.now()
        delete_all.action = Marker.DELETEALL
        arr.markers.append(delete_all)

        marker_id = 0

        for i, person in enumerate(people_positions):
            px = person['x']
            py = person['y']

            m_person = Marker()
            m_person.header.frame_id = frame_id
            m_person.header.stamp = rospy.Time.now()
            m_person.ns = "detected_people"
            m_person.id = marker_id
            marker_id += 1

            m_person.type = Marker.CYLINDER
            m_person.action = Marker.ADD
            m_person.pose.position.x = px
            m_person.pose.position.y = py
            m_person.pose.position.z = 0.25
            m_person.pose.orientation.w = 1.0

            m_person.scale.x = 0.22
            m_person.scale.y = 0.22
            m_person.scale.z = 0.5

            m_person.color.r = 0.0
            m_person.color.g = 1.0
            m_person.color.b = 0.0
            m_person.color.a = 0.9
            m_person.lifetime = rospy.Duration(0.2)

            arr.markers.append(m_person)

            m_txt = Marker()
            m_txt.header.frame_id = frame_id
            m_txt.header.stamp = rospy.Time.now()
            m_txt.ns = "detected_people_text"
            m_txt.id = marker_id
            marker_id += 1

            m_txt.type = Marker.TEXT_VIEW_FACING
            m_txt.action = Marker.ADD
            m_txt.pose.position.x = px
            m_txt.pose.position.y = py
            m_txt.pose.position.z = 0.6
            m_txt.pose.orientation.w = 1.0

            m_txt.scale.z = 0.12
            m_txt.color.r = 1.0
            m_txt.color.g = 1.0
            m_txt.color.b = 1.0
            m_txt.color.a = 1.0
            m_txt.lifetime = rospy.Duration(0.2)

            m_txt.text = "person {}".format(i)
            arr.markers.append(m_txt)

        self.marker_pub.publish(arr)


# =========================================================
# 7. main
# =========================================================
if __name__ == '__main__':
    try:
        RealTimeLegDetector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
