function feat = compute_action_features_stage5(robot_xy, goal_xy, person, action_wp, corridor_half_width)

    if nargin < 5
        corridor_half_width = 1.0;
    end

    robot_xy = double(robot_xy(:)');
    goal_xy = double(goal_xy(:)');
    action_wp = double(action_wp(:)');

    action_x = robot_xy(1) + action_wp(1);
    action_y = robot_xy(2) + action_wp(2);

    person_x = person.x;
    person_y = person.y;

    if isfield(person, "pred_x")
        person_pred_x = person.pred_x;
    else
        person_pred_x = person.x + person.vx * 1.0;
    end

    if isfield(person, "pred_y")
        person_pred_y = person.pred_y;
    else
        person_pred_y = person.y + person.vy * 1.0;
    end

    %% 1. goal_distance
    goal_distance = sqrt((goal_xy(1) - action_x)^2 + (goal_xy(2) - action_y)^2);
    goal_distance = goal_distance / max(norm(goal_xy), 1e-6);

    %% 2. wall_risk
    % Exponential wall risk. This is active before hitting the wall.
    dist_to_wall = corridor_half_width - abs(action_y);

    if dist_to_wall <= 0
        wall_risk = 30.0;
    else
        wall_risk = 8.0 * exp(-dist_to_wall / 0.12);
    end

    %% 3. near_wall_cost
    % Direct near-wall penalty. This makes wall-awareness easier to learn.
    % No penalty near center, rapidly increases after |y| > 0.55.
    near_wall_threshold = 0.70;
    near_wall_cost = max(0.0, abs(action_y) - near_wall_threshold)^2 / (0.30^2);
    %% 4. current_person_risk
    d_current = sqrt((action_x - person_x)^2 + (action_y - person_y)^2);
    current_person_risk = exp(-(d_current^2) / (2.0 * 0.36^2));

    %% 5. predicted_person_risk
    d_pred = sqrt((action_x - person_pred_x)^2 + (action_y - person_pred_y)^2);
    predicted_person_risk = exp(-(d_pred^2) / (2.0 * 0.42^2));

    %% 6. front_person_risk
    rel_px = person_x - robot_xy(1);
    rel_py = person_y - robot_xy(2);

    if rel_px > 0 && rel_px < 2.6 && abs(rel_py) < 0.42
        front_person_risk = 1.0;
    else
        front_person_risk = 0.0;
    end

    %% 7. predicted_collision_risk
    rel_pred_x = person_pred_x - robot_xy(1);
    rel_pred_y = person_pred_y - robot_xy(2);

    future_lateral_distance = abs(person_pred_y - action_y);

    if rel_pred_x > 0 && rel_pred_x < 2.6
        longitudinal_risk = exp(-((rel_pred_x - 1.05)^2) / (2.0 * 0.85^2));
    else
        longitudinal_risk = 0.0;
    end

    lateral_risk = exp(-(future_lateral_distance^2) / (2.0 * 0.32^2));
    predicted_collision_risk = longitudinal_risk * lateral_risk;

    %% 8. center_deviation
    center_deviation = abs(action_y);

    %% 9. lateral_motion
    lateral_motion = abs(action_wp(2)) / 0.08;

    %% 10. same_side_current_risk
    same_side_current_raw = max(0.0, sign_nonzero_stage5(rel_py) * action_wp(2));
    same_side_current_risk = predicted_collision_risk * 3.0 * same_side_current_raw;

    %% 11. same_side_predicted_risk
    same_side_predicted_raw = max(0.0, sign_nonzero_stage5(rel_pred_y) * action_wp(2));
    same_side_predicted_risk = predicted_collision_risk * 3.0 * same_side_predicted_raw;

    %% 12. opposite_side_bonus_as_cost
    % Negative feature. Positive theta means the cost decreases when moving
    % opposite to the person's predicted lateral side.
    opposite_side_raw = max(0.0, -sign_nonzero_stage5(rel_pred_y) * action_wp(2));
    opposite_side_bonus_as_cost = -predicted_collision_risk * 3.0 * opposite_side_raw;

    %% 13. return_to_center_cost
    % Penalize actions that do not reduce |y| after avoidance is no longer strong.
    no_collision = 1.0 - min(predicted_collision_risk, 1.0);

    current_y = robot_xy(2);
    next_abs_y = abs(action_y);
    current_abs_y = abs(current_y);

    if current_abs_y > 0.08
        if next_abs_y < current_abs_y
            return_to_center_cost = 0.0;
        else
            return_to_center_cost = no_collision * (current_abs_y^2 + 0.5);
        end
    else
        return_to_center_cost = 0.0;
    end

    %% 14. outward_motion_cost
    % Penalize moving further away from the centerline.
    if abs(current_y) > 0.10
        outward = sign_nonzero_stage5(current_y) * action_wp(2);
        outward_motion_cost = max(0.0, outward) * abs(current_y) / 0.30;
    else
        outward_motion_cost = 0.0;
    end

    %% 15-17. Angle-independent dynamic collision features
    % Model the candidate action as a one-second motion primitive.  This is
    % deliberately based on relative velocity, not on the person's current
    % bearing, so it also detects side and diagonal crossings.
    prediction_horizon_s = 1.0;
    action_velocity = action_wp / prediction_horizon_s;
    relative_position = [person_x - robot_xy(1), person_y - robot_xy(2)];
    relative_velocity = [person.vx, person.vy] - action_velocity;

    t_ca_unclamped = -dot(relative_position, relative_velocity) / ...
        (dot(relative_velocity, relative_velocity) + 1e-6);
    t_ca = min(max(t_ca_unclamped, 0.0), prediction_horizon_s);
    d_ca = norm(relative_position + t_ca * relative_velocity);

    dca_risk = exp(-(d_ca^2) / (2.0 * 0.45^2));
    approaching = double(t_ca_unclamped > 0.0 && t_ca_unclamped <= prediction_horizon_s);
    ttc_risk = approaching * dca_risk * exp(-t_ca / 0.55);
    closing_speed = max(0.0, -dot(relative_position, relative_velocity) / ...
        max(norm(relative_position), 1e-6));

    %% 18. safety_clearance_cost
    % Give IRL an explicit, continuous penalty for any candidate action whose
    % predicted closest approach is inside the desired social safety distance.
    % Keep this value synchronized with success.py and the evaluator criteria.
    safety_clearance_m = 0.70;
    safety_clearance_cost = max(0.0, safety_clearance_m - d_ca)^2 / ...
        (safety_clearance_m^2);

    feat = [
        goal_distance, ...
        wall_risk, ...
        near_wall_cost, ...
        current_person_risk, ...
        predicted_person_risk, ...
        front_person_risk, ...
        predicted_collision_risk, ...
        center_deviation, ...
        lateral_motion, ...
        same_side_current_risk, ...
        same_side_predicted_risk, ...
        opposite_side_bonus_as_cost, ...
        return_to_center_cost, ...
        outward_motion_cost, ...
        dca_risk, ...
        ttc_risk, ...
        closing_speed, ...
        safety_clearance_cost ...
    ];
end

function s = sign_nonzero_stage5(x)
    if x >= 0
        s = 1.0;
    else
        s = -1.0;
    end
end
