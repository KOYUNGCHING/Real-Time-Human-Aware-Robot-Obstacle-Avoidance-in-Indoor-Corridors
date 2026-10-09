#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS 2 port of ``irl_ori/success.py`` (the 4-action / 14-feature policy).

Topics and message types are deliberately kept compatible with the ROS 1
node: ``/odom`` (Odometry), ``/people_states`` (String JSON), ``/scan``
(LaserScan), and ``/cmd_vel`` (TwistStamped).  Set ``model_path`` if the model JSON
is installed somewhere other than this script's directory.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


class IRLStage5NavNodeSafe(Node):
    def __init__(self):
        super().__init__("irl_stage5_nav_node_safe")

        # ROS 2 parameter names do not use ROS 1's private-name ``~`` prefix.
        script_dir = Path(__file__).resolve().parent
        default_model_path_obj = (
            script_dir / "config" / "stage5_irl_model_4actions_14features.json"
        )
        if not default_model_path_obj.is_file():
            default_model_path_obj = (
                script_dir / "stage5_irl_model_4actions_14features.json"
            )
        default_model_path = str(default_model_path_obj)
        self.model_path = os.path.expanduser(
            str(self.parameter("model_path", default_model_path))
        )
        self.control_rate = float(self.parameter("control_rate", 20.0))
        self.goal_forward = float(self.parameter("goal_forward", 8.0))
        self.goal_tolerance = float(self.parameter("goal_tolerance", 0.15))
        self.goal_lateral_tolerance = float(
            self.parameter("goal_lateral_tolerance", 0.20)
        )
        self.final_centering_speed = float(
            self.parameter("final_centering_speed", 0.07)
        )
        self.final_centering_angular_gain = float(
            self.parameter("final_centering_angular_gain", 1.6)
        )
        self.final_centering_max_angular = float(
            self.parameter("final_centering_max_angular", 0.45)
        )
        self.final_centering_lookahead = float(
            self.parameter("final_centering_lookahead", 0.45)
        )
        self.final_centering_max_overshoot = float(
            self.parameter("final_centering_max_overshoot", 0.80)
        )
        self.corridor_half_width = float(self.parameter("corridor_half_width", 1.0))
        self.forward_axis = str(self.parameter("forward_axis", "x"))
        self.use_start_heading_frame = bool(
            self.parameter("use_start_heading_frame", True)
        )

        self.v_forward = float(self.parameter("v_forward", 0.08))
        self.v_slow = float(self.parameter("v_slow", self.v_forward))
        self.v_turn = float(self.parameter("v_turn", self.v_forward))
        self.w_slight = float(self.parameter("w_slight", 0.055))
        self.w_turn = float(self.parameter("w_turn", self.w_slight))
        self.turn_sign = float(self.parameter("turn_sign", 1.0))

        self.no_person_center_deadband = float(
            self.parameter("no_person_center_deadband", 0.30)
        )
        self.same_side_block_limit = float(
            self.parameter("same_side_block_limit", 0.45)
        )
        self.return_center_y = float(self.parameter("return_center_y", 0.45))
        self.clear_lateral_y = float(self.parameter("clear_lateral_y", 0.45))
        self.outer_limit_y = float(self.parameter("outer_limit_y", 0.60))
        self.use_goal_line_controller = bool(
            self.parameter("use_goal_line_controller", True)
        )
        self.goal_line_deadband = float(self.parameter("goal_line_deadband", 0.10))
        self.goal_line_k_y = float(self.parameter("goal_line_k_y", 0.35))
        self.goal_line_k_yaw = float(self.parameter("goal_line_k_yaw", 0.55))
        self.goal_line_max_w = float(self.parameter("goal_line_max_w", 0.045))

        self.use_emergency_wall_stop = bool(
            self.parameter("use_emergency_wall_stop", True)
        )
        self.emergency_wall_stop_limit = float(
            self.parameter("emergency_wall_stop_limit", 0.98)
        )
        self.use_safety_scan = bool(self.parameter("use_safety_scan", True))
        self.stop_distance = float(self.parameter("stop_distance", 0.22))
        self.slow_distance = float(self.parameter("slow_distance", 0.35))
        self.front_angle_deg = float(self.parameter("front_angle_deg", 18.0))

        self.people_timeout = float(self.parameter("people_timeout", 1.2))
        self.people_hold_time = float(self.parameter("people_hold_time", 1.8))
        self.use_human_safety_override = bool(
            self.parameter("use_human_safety_override", True)
        )
        self.person_stop_x = float(self.parameter("person_stop_x", 0.20))
        self.person_stop_y = float(self.parameter("person_stop_y", 0.35))
        self.person_avoid_x = float(self.parameter("person_avoid_x", 2.8))
        self.person_avoid_y = float(self.parameter("person_avoid_y", 0.70))
        self.debug_every_n = int(self.parameter("debug_every_n", 10))
        self.loop_count = 0
        self._last_log_times = {}

        self.load_model(self.model_path)

        self.odom_ready = False
        self.robot_forward = 0.0
        self.robot_lateral = 0.0
        self.robot_yaw = 0.0
        self.odom_origin_set = False
        self.odom_origin_x = 0.0
        self.odom_origin_y = 0.0
        self.odom_origin_yaw = 0.0
        self.people = []
        self.last_people_time_s = None
        self.last_real_person = None
        self.last_real_person_time_s = None
        self.scan_ready = False
        self.front_min_range = float("inf")
        self.goal_reached = False

        # Sensor QoS accepts the best-effort publishers commonly used for
        # odometry and laser scans in ROS 2.  The other topics use depth 10.
        topic_qos = QoSProfile(depth=10)
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_callback, qos_profile_sensor_data
        )
        self.people_sub = self.create_subscription(
            String, "/people_states", self.people_callback, topic_qos
        )
        if self.use_safety_scan:
            self.scan_sub = self.create_subscription(
                LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data
            )
        # The physical base_control node consumes TwistStamped commands.
        self.cmd_pub = self.create_publisher(TwistStamped, "/cmd_vel", topic_qos)
        self.debug_pub = self.create_publisher(String, "/irl_nav_debug", topic_qos)

        # A ROS 2 timer replaces ROS 1's blocking Rate/while loop, allowing
        # rclpy.spin() to service subscriptions and control updates together.
        self.control_timer = self.create_timer(
            1.0 / max(self.control_rate, 1e-3), self.control_step
        )

        self.get_logger().info("[IRL NAV ROS2] Node initialized.")
        self.get_logger().info("[IRL NAV ROS2] model_path = %s" % self.model_path)
        self.get_logger().info(
            "[IRL NAV ROS2] use_start_heading_frame = %s"
            % self.use_start_heading_frame
        )
        self.get_logger().info(
            "[IRL NAV ROS2] use_goal_line_controller = %s"
            % self.use_goal_line_controller
        )
        self.get_logger().info("[IRL NAV ROS2] turn_sign = %.2f" % self.turn_sign)
        self.get_logger().info("[IRL NAV ROS2] v_forward = %.3f" % self.v_forward)
        self.get_logger().info("[IRL NAV ROS2] w_slight = %.3f" % self.w_slight)

    def parameter(self, name, default):
        """Declare and read one ROS 2 parameter."""
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def log_throttled(self, level, key, period_s, message):
        """Small replacement for rospy.log*_throttle."""
        now = self.now_s()
        previous = self._last_log_times.get(key)
        if previous is not None and now - previous < period_s:
            return
        self._last_log_times[key] = now
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
        expected_features = [
            "goal_distance", "wall_risk", "near_wall_cost",
            "current_person_risk", "predicted_person_risk", "front_person_risk",
            "predicted_collision_risk", "center_deviation", "lateral_motion",
            "same_side_current_risk", "same_side_predicted_risk",
            "opposite_side_bonus_as_cost", "return_to_center_cost",
            "outward_motion_cost",
        ]
        if self.feature_names != expected_features:
            raise ValueError(
                "This exact ROS 2 port requires the original 14-feature model; "
                "pass it with model_path. The workspace's newer 18-feature model "
                "uses a different policy implementation."
            )

        self.get_logger().info(
            "[IRL NAV ROS2] Loaded model: %s" % model.get("model_name", "unknown")
        )
        self.get_logger().info("[IRL NAV ROS2] Actions: %s" % self.action_names)
        self.get_logger().info(
            "[IRL NAV ROS2] Number of features: %d" % self.num_features
        )

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
            self.get_logger().info(
                "[IRL NAV ROS2] Odom origin set: origin_x=%.3f, "
                "origin_y=%.3f, origin_yaw=%.3f rad"
                % (self.odom_origin_x, self.odom_origin_y, self.odom_origin_yaw)
            )

        dx = ox_raw - self.odom_origin_x
        dy = oy_raw - self.odom_origin_y
        if self.use_start_heading_frame:
            c = math.cos(self.odom_origin_yaw)
            s = math.sin(self.odom_origin_yaw)
            self.robot_forward = c * dx + s * dy
            self.robot_lateral = -s * dx + c * dy
            self.robot_yaw = self.wrap_angle(yaw_raw - self.odom_origin_yaw)
        elif self.forward_axis == "neg_x":
            self.robot_forward, self.robot_lateral = -dx, -dy
            self.robot_yaw = self.wrap_angle(yaw_raw - math.pi)
        elif self.forward_axis == "y":
            self.robot_forward, self.robot_lateral = dy, -dx
            self.robot_yaw = self.wrap_angle(yaw_raw - math.pi / 2.0)
        elif self.forward_axis == "neg_y":
            self.robot_forward, self.robot_lateral = -dy, dx
            self.robot_yaw = self.wrap_angle(yaw_raw + math.pi / 2.0)
        else:
            self.robot_forward, self.robot_lateral = dx, dy
            self.robot_yaw = self.wrap_angle(yaw_raw)
        self.odom_ready = True

    def people_callback(self, msg):
        try:
            self.people = json.loads(msg.data).get("people", [])
            self.last_people_time_s = self.now_s()
        except Exception as exc:
            self.log_throttled(
                "warning", "people_parse", 1.0,
                "[IRL NAV ROS2] Failed to parse /people_states: {}".format(exc),
            )

    def scan_callback(self, msg):
        front_angle = math.radians(self.front_angle_deg)
        min_range = float("inf")
        angle = msg.angle_min
        for distance in msg.ranges:
            if (
                -front_angle <= angle <= front_angle
                and math.isfinite(distance)
                and msg.range_min < distance < msg.range_max
            ):
                min_range = min(min_range, distance)
            angle += msg.angle_increment
        self.front_min_range = min_range
        self.scan_ready = True

    # Person selection --------------------------------------------------
    def select_most_relevant_person(self):
        now = self.now_s()
        fresh_people = (
            self.last_people_time_s is not None
            and now - self.last_people_time_s <= self.people_timeout
        )
        if fresh_people and self.people:
            best_person = None
            best_score = -math.inf
            for raw_person in self.people:
                px = float(raw_person.get("x", 20.0))
                py = float(raw_person.get("y", 0.0))
                pred_x = float(raw_person.get("pred_x_1s", raw_person.get("pred_x", px)))
                pred_y = float(raw_person.get("pred_y_1s", raw_person.get("pred_y", py)))
                vx = float(raw_person.get("vx", 0.0))
                vy = float(raw_person.get("vy", 0.0))
                front_score = 1.0 if pred_x > 0.0 else -1.0
                lateral_score = math.exp(-(pred_y ** 2) / (2.0 * 0.45 ** 2))
                distance_score = math.exp(
                    -(max(pred_x, 0.0) ** 2) / (2.0 * 2.5 ** 2)
                )
                score = front_score + 2.0 * lateral_score + distance_score
                if score > best_score:
                    best_score = score
                    best_person = {
                        "x": px, "y": py, "vx": vx, "vy": vy,
                        "pred_x": pred_x, "pred_y": pred_y,
                        "is_fake": False, "source": "live",
                    }
            if best_person is not None:
                self.last_real_person = dict(best_person)
                self.last_real_person_time_s = now
                return best_person

        if self.last_real_person is not None:
            age = now - self.last_real_person_time_s
            if age <= self.people_hold_time:
                held = dict(self.last_real_person)
                held["source"] = "held"
                held["is_fake"] = False
                return held
        return self.fake_far_person()

    @staticmethod
    def fake_far_person():
        return {
            "x": 20.0, "y": 0.0, "vx": 0.0, "vy": 0.0,
            "pred_x": 20.0, "pred_y": 0.0,
            "is_fake": True, "source": "fake",
        }

    # Feature computation ----------------------------------------------
    def compute_action_features_ros(self, action_wp, person_local):
        robot_xy = np.array([self.robot_forward, self.robot_lateral], dtype=np.float64)
        goal_xy = np.array([self.goal_forward, 0.0], dtype=np.float64)
        ax_local, ay_local = float(action_wp[0]), float(action_wp[1])
        action_x, action_y = robot_xy[0] + ax_local, robot_xy[1] + ay_local
        person_x = robot_xy[0] + float(person_local["x"])
        person_y = robot_xy[1] + float(person_local["y"])
        person_pred_x = robot_xy[0] + float(person_local["pred_x"])
        person_pred_y = robot_xy[1] + float(person_local["pred_y"])

        goal_distance = math.hypot(goal_xy[0] - action_x, goal_xy[1] - action_y)
        goal_distance /= max(np.linalg.norm(goal_xy), 1e-6)
        dist_to_wall = self.corridor_half_width - abs(action_y)
        wall_risk = 30.0 if dist_to_wall <= 0.0 else 8.0 * math.exp(-dist_to_wall / 0.12)
        near_wall_cost = max(0.0, abs(action_y) - 0.70) ** 2 / (0.30 ** 2)
        d_current = math.hypot(action_x - person_x, action_y - person_y)
        current_person_risk = math.exp(-(d_current ** 2) / (2.0 * 0.36 ** 2))
        d_pred = math.hypot(action_x - person_pred_x, action_y - person_pred_y)
        predicted_person_risk = math.exp(-(d_pred ** 2) / (2.0 * 0.42 ** 2))

        rel_px, rel_py = person_x - robot_xy[0], person_y - robot_xy[1]
        front_person_risk = (
            1.0 if 0.0 < rel_px < 2.6 and abs(rel_py) < 0.42 else 0.0
        )
        rel_pred_x = person_pred_x - robot_xy[0]
        rel_pred_y = person_pred_y - robot_xy[1]
        future_lateral_distance = abs(person_pred_y - action_y)
        longitudinal_risk = (
            math.exp(-((rel_pred_x - 1.05) ** 2) / (2.0 * 0.85 ** 2))
            if 0.0 < rel_pred_x < 2.6 else 0.0
        )
        lateral_risk = math.exp(-(future_lateral_distance ** 2) / (2.0 * 0.32 ** 2))
        predicted_collision_risk = longitudinal_risk * lateral_risk
        center_deviation = abs(action_y)
        lateral_motion = abs(ay_local) / 0.08
        same_side_current_risk = (
            predicted_collision_risk * 3.0
            * max(0.0, self.sign_nonzero(rel_py) * ay_local)
        )
        same_side_predicted_risk = (
            predicted_collision_risk * 3.0
            * max(0.0, self.sign_nonzero(rel_pred_y) * ay_local)
        )
        opposite_side_bonus_as_cost = (
            -predicted_collision_risk * 3.0
            * max(0.0, -self.sign_nonzero(rel_pred_y) * ay_local)
        )

        current_y = robot_xy[1]
        if abs(current_y) > 0.08 and abs(action_y) >= abs(current_y):
            return_to_center_cost = (
                (1.0 - min(predicted_collision_risk, 1.0))
                * (abs(current_y) ** 2 + 0.25)
            )
        else:
            return_to_center_cost = 0.0
        outward_motion_cost = (
            max(0.0, self.sign_nonzero(current_y) * ay_local) * abs(current_y) / 0.30
            if abs(current_y) > 0.10 else 0.0
        )

        return np.array([
            goal_distance, wall_risk, near_wall_cost, current_person_risk,
            predicted_person_risk, front_person_risk, predicted_collision_risk,
            center_deviation, lateral_motion, same_side_current_risk,
            same_side_predicted_risk, opposite_side_bonus_as_cost,
            return_to_center_cost, outward_motion_cost,
        ], dtype=np.float64)

    def compute_costs(self):
        person = self.select_most_relevant_person()
        raw_features, costs = [], []
        for action_index in range(self.num_actions):
            feature = self.compute_action_features_ros(
                self.action_waypoints[action_index], person
            )
            if len(feature) != self.num_features:
                raise ValueError(
                    "Feature length mismatch. Python computed {}, model expects {}."
                    .format(len(feature), self.num_features)
                )
            feature_norm = (feature - self.feature_mean) / self.feature_std
            costs.append(float(np.dot(self.theta, feature_norm) + self.action_bias[action_index]))
            raw_features.append(feature)

        costs = np.array(costs, dtype=np.float64)
        raw_features = np.vstack(raw_features)
        logits = -costs
        logits -= np.max(logits)
        probabilities = np.exp(logits)
        probabilities /= np.sum(probabilities)
        return costs, probabilities, raw_features, person

    # Decision and control ---------------------------------------------
    def choose_action(self):
        costs, probabilities, raw_features, person = self.compute_costs()
        if person.get("is_fake", False):
            if abs(self.robot_lateral) < self.no_person_center_deadband:
                best_index = self.action_names.index("forward")
            elif self.robot_lateral < 0.0 and "slight_left" in self.action_names:
                best_index = self.action_names.index("slight_left")
            elif self.robot_lateral > 0.0 and "slight_right" in self.action_names:
                best_index = self.action_names.index("slight_right")
            else:
                best_index = self.action_names.index("forward")
            return best_index, costs, probabilities, raw_features, person

        best_index = int(np.argmin(costs))
        best_name = self.action_names[best_index]
        if (
            self.robot_lateral > self.same_side_block_limit
            and best_name in ["slight_left", "left"]
            and "forward" in self.action_names
        ):
            best_index = self.action_names.index("forward")
        if (
            self.robot_lateral < -self.same_side_block_limit
            and best_name in ["slight_right", "right"]
            and "forward" in self.action_names
        ):
            best_index = self.action_names.index("forward")
        return best_index, costs, probabilities, raw_features, person

    def action_to_cmd(self, action_index):
        cmd = Twist()
        name = self.action_names[action_index]
        if name in ["slow_forward", "forward"]:
            cmd.linear.x = self.v_forward
        elif name == "slight_left":
            cmd.linear.x = self.v_forward
            cmd.angular.z = self.turn_sign * self.w_slight
        elif name == "slight_right":
            cmd.linear.x = self.v_forward
            cmd.angular.z = -self.turn_sign * self.w_slight
        elif name == "left":
            cmd.linear.x = self.v_forward
            cmd.angular.z = self.turn_sign * self.w_turn
        elif name == "right":
            cmd.linear.x = self.v_forward
            cmd.angular.z = -self.turn_sign * self.w_turn
        else:
            self.log_throttled(
                "warning", "unknown_action", 1.0,
                "[IRL NAV ROS2] Unknown action name: {}. Stop.".format(name),
            )
        return cmd

    def make_cmd(self, linear_x, angular_z):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        return cmd

    def goal_line_cmd(self):
        cmd = Twist()
        cmd.linear.x = self.v_forward

        if (
            abs(self.robot_lateral) <= self.goal_line_deadband
            and abs(self.robot_yaw) <= self.goal_line_deadband
        ):
            cmd.angular.z = 0.0
            return cmd

        angular = (
            -self.goal_line_k_y * self.robot_lateral
            - self.goal_line_k_yaw * self.robot_yaw
        )
        angular = max(-self.goal_line_max_w, min(self.goal_line_max_w, angular))
        cmd.angular.z = self.turn_sign * angular
        return cmd

    def person_requires_avoid(self, person):
        if person.get("is_fake", False):
            return False

        px = float(person.get("x", 20.0))
        py = float(person.get("y", 0.0))
        pred_x = float(person.get("pred_x", px))
        pred_y = float(person.get("pred_y", py))
        use_x = min(px, pred_x)
        use_y = py if abs(py) >= abs(pred_y) else pred_y

        if 0.0 < use_x < self.person_stop_x and abs(use_y) < self.person_stop_y:
            return True
        return 0.0 < use_x < self.person_avoid_x and abs(use_y) < self.person_avoid_y

    def at_goal(self):
        return (
            self.robot_forward >= self.goal_forward - self.goal_tolerance
            and abs(self.robot_lateral) <= self.goal_lateral_tolerance
        )

    def past_goal_limit(self):
        return (
            self.robot_forward
            >= self.goal_forward + self.final_centering_max_overshoot
        )

    def final_centering_cmd(self):
        cmd = Twist()
        cmd.linear.x = min(self.final_centering_speed, self.v_forward)

        remaining_x = self.goal_forward - self.robot_forward
        aim_x = max(remaining_x, self.final_centering_lookahead)
        desired_yaw = math.atan2(-self.robot_lateral, aim_x)
        yaw_error = self.wrap_angle(desired_yaw - self.robot_yaw)
        angular = self.final_centering_angular_gain * yaw_error
        angular = max(
            -self.final_centering_max_angular,
            min(self.final_centering_max_angular, angular),
        )
        cmd.angular.z = self.turn_sign * angular
        return cmd

    def apply_human_safety_override(self, cmd, person):
        if not self.use_human_safety_override or person.get("is_fake", False):
            return cmd, "none"

        px, py = float(person.get("x", 20.0)), float(person.get("y", 0.0))
        pred_x = float(person.get("pred_x", px))
        pred_y = float(person.get("pred_y", py))
        use_x = min(px, pred_x)
        use_y = py if abs(py) >= abs(pred_y) else pred_y
        robot_y = self.robot_lateral

        if 0.0 < use_x < self.person_stop_x and abs(use_y) < self.person_stop_y:
            return Twist(), "human_stop_too_close"
        if robot_y > self.return_center_y and use_y < -self.clear_lateral_y:
            return self.make_cmd(self.v_forward, -self.turn_sign * abs(self.w_slight)), "human_cleared_return_right"
        if robot_y < -self.return_center_y and use_y > self.clear_lateral_y:
            return self.make_cmd(self.v_forward, self.turn_sign * abs(self.w_slight)), "human_cleared_return_left"
        if robot_y > self.outer_limit_y:
            return self.make_cmd(self.v_forward, -self.turn_sign * abs(self.w_slight)), "limit_return_right"
        if robot_y < -self.outer_limit_y:
            return self.make_cmd(self.v_forward, self.turn_sign * abs(self.w_slight)), "limit_return_left"
        if 0.0 < use_x < self.person_avoid_x and abs(use_y) < self.person_avoid_y:
            if use_y >= 0.0:
                return self.make_cmd(self.v_forward, -self.turn_sign * abs(self.w_slight)), "human_force_slight_right"
            return self.make_cmd(self.v_forward, self.turn_sign * abs(self.w_slight)), "human_force_slight_left"
        return cmd, "none"

    def apply_emergency_wall_stop(self, cmd):
        if not self.use_emergency_wall_stop:
            return cmd, "none"
        if abs(self.robot_lateral) >= self.emergency_wall_stop_limit:
            side = "left" if self.robot_lateral > 0.0 else "right"
            return Twist(), "emergency_wall_stop_{}_side".format(side)
        return cmd, "none"

    def apply_safety_scan_override(self, cmd):
        if not self.use_safety_scan or not self.scan_ready:
            return cmd, "none"
        if self.front_min_range < self.stop_distance:
            return Twist(), "scan_stop_front_obstacle"
        if self.front_min_range < self.slow_distance:
            cmd.linear.x = min(cmd.linear.x, 0.05)
            return cmd, "scan_slow_front_obstacle"
        return cmd, "none"

    @staticmethod
    def first_safety_state(human_state, wall_state, scan_state):
        if human_state != "none":
            return human_state
        if wall_state != "none":
            return wall_state
        return scan_state

    def control_step(self):
        if not self.odom_ready:
            self.log_throttled(
                "warning", "waiting_odom", 1.0, "[IRL NAV ROS2] Waiting for /odom..."
            )
            self.stop_robot()
            return

        if self.at_goal() or self.past_goal_limit():
            if not self.goal_reached:
                self.get_logger().info(
                    "[IRL NAV ROS2] Goal reached. Stop robot. x=%.3f y=%.3f"
                    % (self.robot_forward, self.robot_lateral)
                )
                self.goal_reached = True
            self.stop_robot()
            return

        if self.robot_forward >= self.goal_forward - self.goal_tolerance:
            try:
                person = self.select_most_relevant_person()
                cmd = self.final_centering_cmd()
                cmd, human_state = self.apply_human_safety_override(cmd, person)
                cmd, wall_state = self.apply_emergency_wall_stop(cmd)
                cmd, scan_state = self.apply_safety_scan_override(cmd)
                safety_state = self.first_safety_state(
                    human_state, wall_state, scan_state
                )
                self.publish_cmd(cmd)
                self.log_throttled(
                    "info", "final_centering", 1.0,
                    "[IRL NAV ROS2] Near goal, returning to center. "
                    "x={:.2f} y={:.2f} yaw={:.2f} cmd_v={:.2f} "
                    "cmd_w={:.3f} safety={}".format(
                        self.robot_forward, self.robot_lateral, self.robot_yaw,
                        cmd.linear.x, cmd.angular.z, safety_state,
                    ),
                )
            except Exception as exc:
                self.log_throttled(
                    "error", "final_centering_error", 1.0,
                    "[IRL NAV ROS2] Error in final centering: {}".format(exc),
                )
                self.stop_robot()
            return

        try:
            action_index, costs, probabilities, _, person = self.choose_action()
            selected_name = self.action_names[action_index]
            cmd = self.action_to_cmd(action_index)

            if self.use_goal_line_controller and not self.person_requires_avoid(person):
                cmd = self.goal_line_cmd()
                selected_name = "goal_line_follow"

            cmd, human_state = self.apply_human_safety_override(cmd, person)
            cmd, wall_state = self.apply_emergency_wall_stop(cmd)
            cmd, scan_state = self.apply_safety_scan_override(cmd)
            safety_state = self.first_safety_state(human_state, wall_state, scan_state)
            self.publish_cmd(cmd)
            self.publish_debug(costs, probabilities, person, safety_state, selected_name)
        except Exception as exc:
            self.log_throttled(
                "error", "control_error", 1.0,
                "[IRL NAV ROS2] Error in control loop: {}".format(exc),
            )
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

    def publish_debug(self, costs, probabilities, person, safety_state, selected_name):
        self.loop_count += 1
        if self.loop_count % max(self.debug_every_n, 1) != 0:
            return
        debug = {
            "robot_forward": self.robot_forward,
            "robot_lateral": self.robot_lateral,
            "robot_yaw": self.robot_yaw,
            "selected_action": selected_name,
            "costs": {self.action_names[i]: float(costs[i]) for i in range(self.num_actions)},
            "probs": {
                self.action_names[i]: float(probabilities[i])
                for i in range(self.num_actions)
            },
            "selected_person": person,
            "front_min_range": float(self.front_min_range),
            "safety_state": safety_state,
        }
        self.debug_pub.publish(String(data=json.dumps(debug)))
        self.log_throttled(
            "info", "decision", 1.0,
            "[IRL NAV ROS2] x={:.2f} y={:.2f} yaw={:.2f} action={} "
            "cost={} person={} px={:.2f} py={:.2f} safety={}".format(
                self.robot_forward, self.robot_lateral, self.robot_yaw, selected_name,
                np.round(costs, 3).tolist(), person.get("source", "unknown"),
                float(person.get("x", 999.0)), float(person.get("y", 999.0)),
                safety_state,
            ),
        )

    @staticmethod
    def sign_nonzero(value):
        return 1.0 if value >= 0.0 else -1.0

    @staticmethod
    def yaw_from_quaternion(x, y, z, w):
        """Return yaw directly, avoiding ROS 1-only tf.transformations."""
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    @staticmethod
    def wrap_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = IRLStage5NavNodeSafe()
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
