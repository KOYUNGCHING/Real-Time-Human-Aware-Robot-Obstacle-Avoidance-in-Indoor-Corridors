#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS2 version of the Stage 5 linear IRL human-avoidance node.

It uses the same 18 MATLAB features and the same JSON model as
success_ROS1.py.  The expected /people_states message is std_msgs/String
containing JSON with a ``people`` array; each person must provide x, y, vx,
vy in the x-forward / y-left local frame.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class IRLStage5NavNodeSafeROS2(Node):
    """ROS2 wrapper for the trained linear Stage 5 IRL policy."""

    def __init__(self):
        super().__init__("irl_stage5_nav_node_safe")

        # ROS2 parameters use ordinary names (not ROS1's ~private syntax).
        default_model = str(
            Path(__file__).resolve().parent
            / "config"
            / "stage5_irl_model_4actions.json"
        )
        self.model_path = os.path.expanduser(str(self.param("model_path", default_model)))
        self.control_rate = float(self.param("control_rate", 20.0))
        self.goal_forward = float(self.param("goal_forward", 8.0))
        self.goal_tolerance = float(self.param("goal_tolerance", 0.15))
        self.corridor_half_width = float(self.param("corridor_half_width", 1.0))
        self.forward_axis = str(self.param("forward_axis", "x"))
        self.use_start_heading_frame = bool(self.param("use_start_heading_frame", True))

        self.v_forward = float(self.param("v_forward", 0.10))
        self.v_slow = float(self.param("v_slow", 0.5 * self.v_forward))
        self.v_turn = float(self.param("v_turn", self.v_forward))
        # The original 0.055 rad/s turn was too shallow on the physical base.
        # Use a stronger but still moderate avoidance turn.
        self.w_slight = float(self.param("w_slight", 0.20))
        self.w_turn = float(self.param("w_turn", self.w_slight))
        self.turn_sign = float(self.param("turn_sign", 1.0))

        self.no_person_center_deadband = float(self.param("no_person_center_deadband", 0.10))
        self.same_side_block_limit = float(self.param("same_side_block_limit", 0.45))
        self.return_center_y = float(self.param("return_center_y", 0.45))
        self.clear_lateral_y = float(self.param("clear_lateral_y", 0.45))
        self.outer_limit_y = float(self.param("outer_limit_y", 0.60))

        # Once the person has cleared, use odometry feedback to rejoin y=0
        # and the original heading.  This is deliberately separate from IRL:
        # avoidance remains learned, while route recovery is deterministic.
        self.use_centerline_recovery = bool(self.param("use_centerline_recovery", True))
        self.center_recovery_y_tolerance = float(self.param("center_recovery_y_tolerance", 0.08))
        self.center_recovery_heading_tolerance = math.radians(
            float(self.param("center_recovery_heading_tolerance_deg", 4.0))
        )
        self.center_recovery_lookahead = float(self.param("center_recovery_lookahead", 0.50))
        self.center_recovery_heading_gain = float(self.param("center_recovery_heading_gain", 1.5))
        self.center_recovery_max_w = float(self.param("center_recovery_max_w", 0.20))
        self.center_recovery_speed = float(self.param("center_recovery_speed", self.v_forward))

        self.use_emergency_wall_stop = bool(self.param("use_emergency_wall_stop", True))
        self.emergency_wall_stop_limit = float(self.param("emergency_wall_stop_limit", 0.98))
        self.use_safety_scan = bool(self.param("use_safety_scan", True))
        self.stop_distance = float(self.param("stop_distance", 0.22))
        self.slow_distance = float(self.param("slow_distance", 0.35))
        self.front_angle_deg = float(self.param("front_angle_deg", 18.0))

        self.people_timeout = float(self.param("people_timeout", 1.2))
        self.people_hold_time = float(self.param("people_hold_time", 1.8))
        self.use_human_safety_override = bool(self.param("use_human_safety_override", True))
        self.person_stop_x = float(self.param("person_stop_x", 0.20))
        self.person_stop_y = float(self.param("person_stop_y", 0.35))
        # Begin reacting earlier and include slightly wider crossing paths.
        self.person_avoid_x = float(self.param("person_avoid_x", 4.5))
        self.person_avoid_y = float(self.param("person_avoid_y", 1.00))
        self.prediction_horizon_s = float(self.param("prediction_horizon_s", 1.0))
        self.safety_prediction_horizon_s = float(self.param("safety_prediction_horizon_s", 2.5))
        self.crossing_stop_distance = float(self.param("crossing_stop_distance", 0.32))
        self.safety_clearance_m = float(self.param("safety_clearance_m", 0.70))
        self.moving_person_lateral_speed_threshold = float(
            self.param("moving_person_lateral_speed_threshold", 0.12)
        )
        self.dynamic_avoid_speed = float(self.param("dynamic_avoid_speed", 0.08))
        self.dynamic_avoid_angular = float(self.param("dynamic_avoid_angular", 0.40))
        self.debug_every_n = int(self.param("debug_every_n", 10))

        self.loop_count = 0
        self._last_log_time = {}
        self.load_model(self.model_path)

        self.odom_ready = False
        self.robot_forward = 0.0
        self.robot_lateral = 0.0
        self.odom_origin_set = False
        self.odom_origin_x = 0.0
        self.odom_origin_y = 0.0
        self.odom_origin_yaw = 0.0
        self.robot_heading = 0.0
        self.people = []
        self.last_people_time_s = -math.inf
        self.last_real_person = None
        self.last_real_person_time_s = -math.inf
        self.scan_ready = False
        self.front_min_range = float("inf")
        self.goal_reached = False

        # Sensor topics commonly use best-effort QoS in ROS2.
        self.odom_sub = self.create_subscription(Odometry, "/odom", self.odom_callback, qos_profile_sensor_data)
        self.people_sub = self.create_subscription(String, "/people_states", self.people_callback, QoSProfile(depth=10))
        if self.use_safety_scan:
            self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data)
        # This robot's base_control subscribes to TwistStamped rather than
        # the unstamped Twist used by many ROS2 mobile bases.
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", QoSProfile(depth=10))
        self.debug_pub = self.create_publisher(String, "/irl_nav_debug", QoSProfile(depth=10))
        self.control_timer = self.create_timer(1.0 / max(self.control_rate, 1e-3), self.control_step)

        self.get_logger().info("[IRL NAV ROS2] Node initialized.")
        self.get_logger().info("[IRL NAV ROS2] model_path = {}".format(self.model_path))
        self.get_logger().info("[IRL NAV ROS2] model features = {}, actions = {}".format(self.num_features, self.action_names))
        self.get_logger().info(
            "[IRL NAV ROS2] avoidance trigger: x < {:.2f} m, |y| < {:.2f} m, w = {:.3f} rad/s".format(
                self.person_avoid_x, self.person_avoid_y, self.w_slight
            )
        )
        self.get_logger().info(
            "[IRL NAV ROS2] dynamic avoid: |vy| >= {:.2f} m/s -> v = {:.3f} m/s, w = {:.3f} rad/s".format(
                self.moving_person_lateral_speed_threshold,
                self.dynamic_avoid_speed,
                self.dynamic_avoid_angular,
            )
        )
        self.get_logger().info(
            "[IRL NAV ROS2] center recovery: |y| <= {:.2f} m, heading <= {:.1f} deg".format(
                self.center_recovery_y_tolerance,
                math.degrees(self.center_recovery_heading_tolerance),
            )
        )

    def param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def log_throttled(self, level, key, period_s, message):
        now = self.now_s()
        if now - self._last_log_time.get(key, -math.inf) < period_s:
            return
        self._last_log_time[key] = now
        # ROS2 Jazzy binds severity to the Python call site.  Keep one fixed
        # source line for each level instead of changing severity via getattr.
        logger = self.get_logger()
        if level == "debug":
            logger.debug(message)
        elif level == "info":
            logger.info(message)
        elif level in ("warn", "warning"):
            logger.warning(message)
        elif level == "error":
            logger.error(message)
        elif level == "fatal":
            logger.fatal(message)
        else:
            logger.info(message)

    # Model loading -----------------------------------------------------
    def load_model(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError("Cannot find IRL model JSON: {}".format(path))
        with open(path, "r", encoding="utf-8") as file:
            model = json.load(file)

        self.theta = np.array(model["theta"], dtype=np.float64).reshape(-1)
        self.action_bias = np.array(model["action_bias"], dtype=np.float64).reshape(-1)
        self.feature_mean = np.array(model["feature_mean"], dtype=np.float64).reshape(-1)
        self.feature_std = np.array(model["feature_std"], dtype=np.float64).reshape(-1)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        self.feature_names = list(model["feature_names"])
        self.action_names = list(model["action_names"])
        self.action_waypoints = np.array(model["action_waypoints"], dtype=np.float64)
        self.num_actions = len(self.action_names)
        self.num_features = len(self.feature_names)

        if len(self.theta) != self.num_features:
            raise ValueError("theta length does not match feature number.")
        if len(self.action_bias) != self.num_actions:
            raise ValueError("action_bias length does not match action number.")
        self.get_logger().info("[IRL NAV ROS2] Loaded model: {}".format(model.get("model_name", "unknown")))

    # ROS callbacks -----------------------------------------------------
    def odom_callback(self, msg):
        ox_raw = msg.pose.pose.position.x
        oy_raw = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        yaw_raw = self.yaw_from_quaternion(q.x, q.y, q.z, q.w)

        if not self.odom_origin_set:
            self.odom_origin_x = ox_raw
            self.odom_origin_y = oy_raw
            self.odom_origin_yaw = yaw_raw
            self.odom_origin_set = True
            self.get_logger().info("[IRL NAV ROS2] Odom origin set.")

        dx = ox_raw - self.odom_origin_x
        dy = oy_raw - self.odom_origin_y
        if self.use_start_heading_frame:
            c = math.cos(self.odom_origin_yaw)
            s = math.sin(self.odom_origin_yaw)
            self.robot_forward = c * dx + s * dy
            self.robot_lateral = -s * dx + c * dy
            self.robot_heading = self.wrap_angle(yaw_raw - self.odom_origin_yaw)
        elif self.forward_axis == "neg_x":
            self.robot_forward, self.robot_lateral = -dx, -dy
            self.robot_heading = self.wrap_angle(yaw_raw - math.pi)
        elif self.forward_axis == "y":
            self.robot_forward, self.robot_lateral = dy, -dx
            self.robot_heading = self.wrap_angle(yaw_raw - math.pi / 2.0)
        elif self.forward_axis == "neg_y":
            self.robot_forward, self.robot_lateral = -dy, dx
            self.robot_heading = self.wrap_angle(yaw_raw + math.pi / 2.0)
        else:
            self.robot_forward, self.robot_lateral = dx, dy
            self.robot_heading = self.wrap_angle(yaw_raw)
        self.odom_ready = True

    def people_callback(self, msg):
        try:
            self.people = json.loads(msg.data).get("people", [])
            self.last_people_time_s = self.now_s()
        except Exception as exc:
            self.log_throttled("warn", "people_parse", 1.0, "[IRL NAV ROS2] Failed to parse /people_states: {}".format(exc))

    def scan_callback(self, msg):
        front_angle = math.radians(self.front_angle_deg)
        min_range = float("inf")
        angle = msg.angle_min
        for distance in msg.ranges:
            if -front_angle <= angle <= front_angle and math.isfinite(distance) and msg.range_min < distance < msg.range_max:
                min_range = min(min_range, distance)
            angle += msg.angle_increment
        self.front_min_range = min_range
        self.scan_ready = True

    # Human selection and feature calculation -------------------------
    def select_most_relevant_person(self):
        now = self.now_s()
        if now - self.last_people_time_s <= self.people_timeout and self.people:
            best_person = None
            best_score = -math.inf
            for raw in self.people:
                px = float(raw.get("x", 20.0))
                py = float(raw.get("y", 0.0))
                vx = float(raw.get("vx", 0.0))
                vy = float(raw.get("vy", 0.0))
                pred_x = float(raw.get("pred_x_1s", raw.get("pred_x", px)))
                pred_y = float(raw.get("pred_y_1s", raw.get("pred_y", py)))
                t_ca, d_ca, imminent, _ = self.closest_approach_metrics(px, py, vx, vy, self.v_forward, 0.0)
                dca_score = math.exp(-(d_ca ** 2) / (2.0 * 0.45 ** 2))
                ttc_score = dca_score * math.exp(-t_ca / 0.55) if imminent else 0.0
                front_score = 1.0 if pred_x > 0.0 else -1.0
                distance_score = math.exp(-(max(pred_x, 0.0) ** 2) / (2.0 * 2.5 ** 2))
                score = 4.0 * ttc_score + 0.5 * dca_score + 0.3 * front_score + distance_score
                if score > best_score:
                    best_score = score
                    best_person = {"x": px, "y": py, "vx": vx, "vy": vy, "pred_x": pred_x, "pred_y": pred_y, "is_fake": False, "source": "live"}
            if best_person is not None:
                self.last_real_person = dict(best_person)
                self.last_real_person_time_s = now
                return best_person

        if self.last_real_person is not None and now - self.last_real_person_time_s <= self.people_hold_time:
            held = dict(self.last_real_person)
            # Continue the constant-velocity EKF prediction while detections
            # are briefly missing.  Freezing the last person position made
            # the policy perform an unnecessary second avoidance curve.
            age = max(0.0, now - self.last_real_person_time_s)
            held["x"] += held["vx"] * age
            held["y"] += held["vy"] * age
            held["pred_x"] = held["x"] + held["vx"] * self.prediction_horizon_s
            held["pred_y"] = held["y"] + held["vy"] * self.prediction_horizon_s
            held["source"] = "held"
            return held
        return self.fake_far_person()

    @staticmethod
    def fake_far_person():
        return {"x": 20.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "pred_x": 20.0, "pred_y": 0.0, "is_fake": True, "source": "fake"}

    def closest_approach_metrics(self, px, py, vx, vy, robot_vx, robot_vy, horizon_s=None):
        horizon = self.prediction_horizon_s if horizon_s is None else float(horizon_s)
        relative_position = np.array([px, py], dtype=np.float64)
        relative_velocity = np.array([vx - robot_vx, vy - robot_vy], dtype=np.float64)
        t_unclamped = -float(np.dot(relative_position, relative_velocity)) / (float(np.dot(relative_velocity, relative_velocity)) + 1e-6)
        t_ca = min(max(t_unclamped, 0.0), horizon)
        d_ca = float(np.linalg.norm(relative_position + t_ca * relative_velocity))
        imminent = 0.0 < t_unclamped <= horizon
        closing_speed = max(0.0, -float(np.dot(relative_position, relative_velocity)) / max(float(np.linalg.norm(relative_position)), 1e-6))
        return t_ca, d_ca, imminent, closing_speed

    def human_hazard_metrics(self, person):
        px = float(person.get("x", 20.0))
        py = float(person.get("y", 0.0))
        vx = float(person.get("vx", 0.0))
        vy = float(person.get("vy", 0.0))
        pred_x = float(person.get("pred_x", px))
        pred_y = float(person.get("pred_y", py))
        horizon = max(self.safety_prediction_horizon_s, self.prediction_horizon_s, 1e-3)

        t_ca, d_ca, imminent, closing_speed = self.closest_approach_metrics(
            px, py, vx, vy, self.v_forward, 0.0, horizon_s=horizon
        )
        candidate_times = [0.0, min(self.prediction_horizon_s, horizon), horizon, t_ca]
        if abs(vy) > 1e-4:
            candidate_times.append(-py / vy)

        candidates = []
        for t in candidate_times:
            if not math.isfinite(t):
                continue
            t = min(max(float(t), 0.0), horizon)
            rel_x = px + (vx - self.v_forward) * t
            rel_y = py + vy * t
            candidates.append((rel_x, rel_y, t))

        # Keep the detector's one-second prediction as an explicit candidate,
        # because some trackers publish filtered prediction fields that are
        # better than vx/vy during startup.
        candidates.append((pred_x, pred_y, min(self.prediction_horizon_s, horizon)))
        in_front = [item for item in candidates if 0.0 < item[0] < self.person_avoid_x]
        search_space = in_front if in_front else candidates
        hazard_x, hazard_y, hazard_t = min(
            search_space,
            key=lambda item: (abs(item[1]), max(item[0], 0.0)),
        )
        side_y = py if abs(py) > 0.05 else pred_y
        if abs(side_y) <= 0.05:
            side_y = hazard_y

        return {
            "x": hazard_x,
            "y": hazard_y,
            "t": hazard_t,
            "side_y": side_y,
            "t_ca": t_ca,
            "d_ca": d_ca,
            "imminent": imminent,
            "closing_speed": closing_speed,
        }

    def stop_action_allowed(self, person):
        """Allow learned waiting only for a safe, near, lateral crossing."""
        if person.get("is_fake", False):
            return False

        px = float(person["x"])
        py = float(person["y"])
        vx = float(person["vx"])
        vy = float(person["vy"])
        if abs(vy) < 0.25 or not (0.0 < px < 1.5):
            return False

        _, d_ca, _, _ = self.closest_approach_metrics(px, py, vx, vy, 0.0, 0.0)
        return d_ca >= 0.50

    def compute_action_features_ros(self, action_wp, person):
        robot_xy = np.array([self.robot_forward, self.robot_lateral], dtype=np.float64)
        goal_xy = np.array([self.goal_forward, 0.0], dtype=np.float64)
        ax, ay = float(action_wp[0]), float(action_wp[1])
        action_x, action_y = robot_xy[0] + ax, robot_xy[1] + ay
        person_x, person_y = robot_xy[0] + float(person["x"]), robot_xy[1] + float(person["y"])
        pred_x, pred_y = robot_xy[0] + float(person["pred_x"]), robot_xy[1] + float(person["pred_y"])

        goal_distance = math.hypot(goal_xy[0] - action_x, goal_xy[1] - action_y) / max(float(np.linalg.norm(goal_xy)), 1e-6)
        wall_distance = self.corridor_half_width - abs(action_y)
        wall_risk = 30.0 if wall_distance <= 0.0 else 8.0 * math.exp(-wall_distance / 0.12)
        near_wall_cost = max(0.0, abs(action_y) - 0.70) ** 2 / 0.30 ** 2
        current_person_risk = math.exp(-math.hypot(action_x - person_x, action_y - person_y) ** 2 / (2.0 * 0.36 ** 2))
        predicted_person_risk = math.exp(-math.hypot(action_x - pred_x, action_y - pred_y) ** 2 / (2.0 * 0.42 ** 2))
        rel_px, rel_py = person_x - robot_xy[0], person_y - robot_xy[1]
        front_person_risk = 1.0 if rel_px > 0.0 and rel_px < 2.6 and abs(rel_py) < 0.42 else 0.0
        rel_pred_x, rel_pred_y = pred_x - robot_xy[0], pred_y - robot_xy[1]
        longitudinal_risk = math.exp(-((rel_pred_x - 1.05) ** 2) / (2.0 * 0.85 ** 2)) if 0.0 < rel_pred_x < 2.6 else 0.0
        predicted_collision_risk = longitudinal_risk * math.exp(-(abs(pred_y - action_y) ** 2) / (2.0 * 0.32 ** 2))
        center_deviation = abs(action_y)
        lateral_motion = abs(ay) / 0.08
        same_side_current_risk = predicted_collision_risk * 3.0 * max(0.0, self.sign_nonzero(rel_py) * ay)
        same_side_predicted_risk = predicted_collision_risk * 3.0 * max(0.0, self.sign_nonzero(rel_pred_y) * ay)
        opposite_side_bonus_as_cost = -predicted_collision_risk * 3.0 * max(0.0, -self.sign_nonzero(rel_pred_y) * ay)

        current_y = robot_xy[1]
        if abs(current_y) > 0.08 and abs(action_y) >= abs(current_y):
            return_to_center_cost = (1.0 - min(predicted_collision_risk, 1.0)) * (abs(current_y) ** 2 + 0.25)
        else:
            return_to_center_cost = 0.0
        outward_motion_cost = max(0.0, self.sign_nonzero(current_y) * ay) * abs(current_y) / 0.30 if abs(current_y) > 0.10 else 0.0

        action_velocity = np.array([ax, ay], dtype=np.float64) / self.prediction_horizon_s
        relative_position = np.array([person_x - robot_xy[0], person_y - robot_xy[1]], dtype=np.float64)
        relative_velocity = np.array([float(person["vx"]), float(person["vy"])], dtype=np.float64) - action_velocity
        t_unclamped = -float(np.dot(relative_position, relative_velocity)) / (float(np.dot(relative_velocity, relative_velocity)) + 1e-6)
        t_ca = min(max(t_unclamped, 0.0), self.prediction_horizon_s)
        d_ca = float(np.linalg.norm(relative_position + t_ca * relative_velocity))
        dca_risk = math.exp(-(d_ca ** 2) / (2.0 * 0.45 ** 2))
        approaching = 1.0 if 0.0 < t_unclamped <= self.prediction_horizon_s else 0.0
        ttc_risk = approaching * dca_risk * math.exp(-t_ca / 0.55)
        closing_speed = max(0.0, -float(np.dot(relative_position, relative_velocity)) / max(float(np.linalg.norm(relative_position)), 1e-6))
        safety_clearance_cost = max(0.0, self.safety_clearance_m - d_ca) ** 2 / self.safety_clearance_m ** 2

        return np.array([
            goal_distance, wall_risk, near_wall_cost, current_person_risk,
            predicted_person_risk, front_person_risk, predicted_collision_risk,
            center_deviation, lateral_motion, same_side_current_risk,
            same_side_predicted_risk, opposite_side_bonus_as_cost,
            return_to_center_cost, outward_motion_cost, dca_risk, ttc_risk,
            closing_speed, safety_clearance_cost,
        ], dtype=np.float64)

    def compute_costs(self):
        person = self.select_most_relevant_person()
        features, costs = [], []
        for index in range(self.num_actions):
            feature = self.compute_action_features_ros(self.action_waypoints[index], person)
            if len(feature) != self.num_features:
                raise ValueError("Feature length mismatch: Python computes {}, model expects {}.".format(len(feature), self.num_features))
            feature_norm = (feature - self.feature_mean) / self.feature_std
            cost = float(np.dot(self.theta, feature_norm) + self.action_bias[index])
            if self.action_names[index] == "stop" and not self.stop_action_allowed(person):
                cost = 1e6
            costs.append(cost)
            features.append(feature)
        costs = np.array(costs, dtype=np.float64)
        logits = -costs - np.max(-costs)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        return costs, probs, np.vstack(features), person

    # Action and safety -------------------------------------------------
    def choose_action(self):
        costs, probs, raw_features, person = self.compute_costs()
        if person.get("is_fake", False):
            if abs(self.robot_lateral) < self.no_person_center_deadband:
                best_idx = self.action_names.index("forward")
            elif self.robot_lateral < 0.0 and "slight_left" in self.action_names:
                best_idx = self.action_names.index("slight_left")
            elif self.robot_lateral > 0.0 and "slight_right" in self.action_names:
                best_idx = self.action_names.index("slight_right")
            else:
                best_idx = self.action_names.index("forward")
            return best_idx, costs, probs, raw_features, person

        best_idx = int(np.argmin(costs))
        best_name = self.action_names[best_idx]
        if self.robot_lateral > self.same_side_block_limit and best_name in ["slight_left", "left"] and "forward" in self.action_names:
            best_idx = self.action_names.index("forward")
        elif self.robot_lateral < -self.same_side_block_limit and best_name in ["slight_right", "right"] and "forward" in self.action_names:
            best_idx = self.action_names.index("forward")
        return best_idx, costs, probs, raw_features, person

    def action_to_cmd(self, action_idx):
        cmd = Twist()
        name = self.action_names[action_idx]
        if name == "slow_forward":
            cmd.linear.x, cmd.angular.z = self.v_slow, 0.0
        elif name == "forward":
            cmd.linear.x, cmd.angular.z = self.v_forward, 0.0
        elif name == "stop":
            # Learned yield action: hold position while the person passes.
            cmd.linear.x, cmd.angular.z = 0.0, 0.0
        elif name == "slight_left":
            cmd.linear.x, cmd.angular.z = self.v_forward, self.turn_sign * self.w_slight
        elif name == "slight_right":
            cmd.linear.x, cmd.angular.z = self.v_forward, -self.turn_sign * self.w_slight
        elif name == "left":
            cmd.linear.x, cmd.angular.z = self.v_forward, self.turn_sign * self.w_turn
        elif name == "right":
            cmd.linear.x, cmd.angular.z = self.v_forward, -self.turn_sign * self.w_turn
        else:
            self.log_throttled("warn", "unknown_action", 1.0, "[IRL NAV ROS2] Unknown action {}. Stop.".format(name))
        return cmd

    def apply_human_safety_override(self, cmd, person):
        if not self.use_human_safety_override or person.get("is_fake", False):
            return cmd, "none"
        px, py = float(person.get("x", 20.0)), float(person.get("y", 0.0))
        vx, vy = float(person.get("vx", 0.0)), float(person.get("vy", 0.0))
        hazard = self.human_hazard_metrics(person)
        use_x = float(hazard["x"])
        use_y = float(hazard["y"])
        side_y = float(hazard["side_y"])
        robot_y = self.robot_lateral
        t_ca, d_ca, imminent = float(hazard["t_ca"]), float(hazard["d_ca"]), bool(hazard["imminent"])

        if imminent and d_ca < self.crossing_stop_distance and 0.0 < use_x < 0.45:
            return Twist(), "human_stop_predicted_crossing"
        if 0.0 < use_x < self.person_stop_x and abs(use_y) < self.person_stop_y:
            return Twist(), "human_stop_too_close"
        if cmd.linear.x == 0.0 and cmd.angular.z == 0.0:
            return cmd, "irl_stop_waiting"
        if robot_y > self.return_center_y and use_y < -self.clear_lateral_y:
            return self.make_cmd(self.v_forward, -self.turn_sign * abs(self.w_slight)), "human_cleared_return_right"
        if robot_y < -self.return_center_y and use_y > self.clear_lateral_y:
            return self.make_cmd(self.v_forward, self.turn_sign * abs(self.w_slight)), "human_cleared_return_left"
        if robot_y > self.outer_limit_y:
            return self.make_cmd(self.v_forward, -self.turn_sign * abs(self.w_slight)), "limit_return_right"
        if robot_y < -self.outer_limit_y:
            return self.make_cmd(self.v_forward, self.turn_sign * abs(self.w_slight)), "limit_return_left"
        should_force_avoid = (
            0.0 < use_x < self.person_avoid_x
            and abs(use_y) < self.person_avoid_y
        ) or (
            imminent
            and d_ca < self.safety_clearance_m
            and 0.0 < use_x < self.person_avoid_x + 0.8
        )
        if should_force_avoid:
            use_dynamic_avoid = (
                abs(vy) >= self.moving_person_lateral_speed_threshold
                or (imminent and d_ca < self.safety_clearance_m)
            )
            avoid_speed = self.dynamic_avoid_speed if use_dynamic_avoid else self.v_forward
            avoid_angular = self.dynamic_avoid_angular if use_dynamic_avoid else self.w_slight
            angular = -self.turn_sign * abs(avoid_angular) if side_y >= 0.0 else self.turn_sign * abs(avoid_angular)
            if side_y >= 0.0:
                direction = "human_force_dynamic_right" if use_dynamic_avoid else "human_force_slight_right"
            else:
                direction = "human_force_dynamic_left" if use_dynamic_avoid else "human_force_slight_left"
            return self.make_cmd(avoid_speed, angular), direction
        return cmd, "none"

    def person_clear_for_center_recovery(self, person):
        """Return True only when rejoining the original path is safe."""
        if person.get("is_fake", False):
            return True

        px = float(person.get("x", 20.0))
        py = float(person.get("y", 0.0))
        vx = float(person.get("vx", 0.0))
        vy = float(person.get("vy", 0.0))
        pred_x = float(person.get("pred_x", px))
        pred_y = float(person.get("pred_y", py))

        _, d_ca, imminent, _ = self.closest_approach_metrics(
            px, py, vx, vy, self.v_forward, 0.0,
            horizon_s=self.safety_prediction_horizon_s,
        )
        if imminent and d_ca < self.safety_clearance_m:
            return False

        hazard = self.human_hazard_metrics(person)
        person_in_front = (
            0.0 < px < self.person_avoid_x
            or 0.0 < pred_x < self.person_avoid_x
            or 0.0 < float(hazard["x"]) < self.person_avoid_x
        )
        person_near_path = min(abs(py), abs(pred_y), abs(float(hazard["y"]))) < self.person_avoid_y
        return not (person_in_front and person_near_path)

    def apply_centerline_recovery(self, cmd, person):
        """Rejoin y=0 with heading feedback after avoidance is complete."""
        if not self.use_centerline_recovery:
            return cmd, "none"
        if not self.person_clear_for_center_recovery(person):
            return cmd, "none"

        y_error = self.robot_lateral
        heading_error = self.robot_heading
        if (
            abs(y_error) <= self.center_recovery_y_tolerance
            and abs(heading_error) <= self.center_recovery_heading_tolerance
        ):
            return cmd, "none"

        lookahead = max(self.center_recovery_lookahead, 0.10)
        desired_heading = math.atan2(-y_error, lookahead)
        steering_error = self.wrap_angle(desired_heading - heading_error)
        angular = self.center_recovery_heading_gain * steering_error
        angular = max(-self.center_recovery_max_w, min(self.center_recovery_max_w, angular))
        angular *= self.turn_sign

        speed = max(0.0, min(self.center_recovery_speed, self.v_forward))
        return self.make_cmd(speed, angular), "centerline_recovery"

    def apply_emergency_wall_stop(self, cmd):
        if not self.use_emergency_wall_stop or abs(self.robot_lateral) < self.emergency_wall_stop_limit:
            return cmd, "none"
        side = "left" if self.robot_lateral > 0.0 else "right"
        return Twist(), "emergency_wall_stop_{}_side".format(side)

    def apply_safety_scan_override(self, cmd):
        if not self.use_safety_scan or not self.scan_ready:
            return cmd, "none"
        if self.front_min_range < self.stop_distance:
            return Twist(), "scan_stop_front_obstacle"
        if self.front_min_range < self.slow_distance:
            cmd.linear.x = min(cmd.linear.x, 0.05)
            return cmd, "scan_slow_front_obstacle"
        return cmd, "none"

    def control_step(self):
        if not self.odom_ready:
            self.log_throttled("warn", "waiting_odom", 1.0, "[IRL NAV ROS2] Waiting for /odom...")
            self.stop_robot()
            return
        if self.robot_forward >= self.goal_forward - self.goal_tolerance:
            if not self.goal_reached:
                self.get_logger().info("[IRL NAV ROS2] Goal reached. Stop robot.")
                self.goal_reached = True
            self.stop_robot()
            return
        try:
            action_idx, costs, probs, _, person = self.choose_action()
            selected_name = self.action_names[action_idx]
            cmd = self.action_to_cmd(action_idx)
            cmd, human_state = self.apply_human_safety_override(cmd, person)
            cmd, recovery_state = self.apply_centerline_recovery(cmd, person)
            cmd, wall_state = self.apply_emergency_wall_stop(cmd)
            cmd, scan_state = self.apply_safety_scan_override(cmd)
            if scan_state != "none":
                safety_state = scan_state
            elif wall_state != "none":
                safety_state = wall_state
            elif recovery_state != "none":
                safety_state = recovery_state
            else:
                safety_state = human_state
            self.publish_cmd(cmd)
            self.publish_debug(costs, probs, person, safety_state, selected_name)
        except Exception as exc:
            self.log_throttled("error", "control_error", 1.0, "[IRL NAV ROS2] Control error: {}".format(exc))
            self.stop_robot()

    def publish_cmd(self, cmd):
        """Publish an internal Twist command as the base's TwistStamped type."""
        if not self.context.ok():
            return
        stamped = TwistStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "base_link"
        stamped.twist = cmd
        self.cmd_pub.publish(stamped)

    def stop_robot(self):
        self.publish_cmd(Twist())

    def publish_debug(self, costs, probs, person, safety_state, selected_name):
        self.loop_count += 1
        if self.loop_count % max(self.debug_every_n, 1) != 0:
            return
        hazard = None
        if not person.get("is_fake", False):
            hazard = self.human_hazard_metrics(person)
        debug = {
            "robot_forward": self.robot_forward,
            "robot_lateral": self.robot_lateral,
            "robot_heading_deg": math.degrees(self.robot_heading),
            "selected_action": selected_name,
            "costs": {self.action_names[i]: float(costs[i]) for i in range(self.num_actions)},
            "probs": {self.action_names[i]: float(probs[i]) for i in range(self.num_actions)},
            "selected_person": person,
            "front_min_range": float(self.front_min_range),
            "safety_state": safety_state,
            "hazard": None if hazard is None else {
                "x": round(float(hazard["x"]), 3),
                "y": round(float(hazard["y"]), 3),
                "t_ca": round(float(hazard["t_ca"]), 3),
                "d_ca": round(float(hazard["d_ca"]), 3),
                "imminent": bool(hazard["imminent"]),
                "closing_speed": round(float(hazard["closing_speed"]), 3),
            },
        }
        self.debug_pub.publish(String(data=json.dumps(debug)))
        self.log_throttled(
            "info", "decision", 1.0,
            "[IRL NAV ROS2] x={:.2f} y={:.2f} action={} costs={} person={} safety={}".format(
                self.robot_forward, self.robot_lateral, selected_name,
                np.round(costs, 3).tolist(), person.get("source", "unknown"), safety_state,
            ),
        )

    @staticmethod
    def make_cmd(linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        return cmd

    @staticmethod
    def sign_nonzero(value):
        return 1.0 if value >= 0.0 else -1.0

    @staticmethod
    def yaw_from_quaternion(x, y, z, w):
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def wrap_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IRLStage5NavNodeSafeROS2()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop_robot()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
