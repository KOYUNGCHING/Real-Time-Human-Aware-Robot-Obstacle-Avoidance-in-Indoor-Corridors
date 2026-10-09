clear; clc; close all;

goal_xy = [8.0, 0.0];
corridor_half_width = 1.0;

action_names = {
    "slow_forward"
    "forward"
    "slight_left"
    "slight_right"
};

% Smaller lateral action than Stage 4.
action_waypoints = [
    0.35,  0.00;   % slow_forward
    0.60,  0.00;   % forward
    0.50,  0.10;   % slight_left
    0.50, -0.10    % slight_right
];
feature_names = {
    "goal_distance"
    "wall_risk"
    "near_wall_cost"
    "current_person_risk"
    "predicted_person_risk"
    "front_person_risk"
    "predicted_collision_risk"
    "center_deviation"
    "lateral_motion"
    "same_side_current_risk"
    "same_side_predicted_risk"
    "opposite_side_bonus_as_cost"
    "return_to_center_cost"
    "outward_motion_cost"
    "dca_risk"
    "ttc_risk"
    "closing_speed"
    "safety_clearance_cost"
};

A = length(action_names);
F = length(feature_names);

action_name_vec = string(action_names);
idx_slow_forward = find(action_name_vec == "slow_forward", 1);
idx_forward = find(action_name_vec == "forward", 1);
idx_slight_left = find(action_name_vec == "slight_left", 1);
idx_slight_right = find(action_name_vec == "slight_right", 1);

X_list = {};
y_list = [];
sample_weight_list = [];
source_type_list = strings(0,1);
scenario_index_list = [];
scenarios = struct([]);
scenario_id = 0;

fprintf("\nBuilding Stage 5 synthetic wall-aware scenarios...\n");

% Include corridor-parallel, transverse, and diagonal approach directions.
% x is robot-forward and y is robot-left.  These directions must match the
% coordinate frame published by /people_states in success.py.
modes = ["no_person", ...
         "static", ...
         "opposite", ...
         "same", ...
         "side_clear", ...
         "near_wall_recovery", ...
         "cross_left_to_right", "cross_right_to_left", ...
         "diag_front_left", "diag_front_right", ...
         "diag_receding_left", "diag_receding_right"];

person_y_values = [-0.55, -0.35, -0.18, 0.0, 0.18, 0.35, 0.55];

for mode = modes
    for py0 = person_y_values
        scenario_id = scenario_id + 1;

        [robot_path, person_traj, expert_actions] = make_synthetic_scenario_stage5( ...
            mode, py0, goal_xy, action_names);

        s = struct();
        s.mode = mode;
        s.person_y0 = py0;
        s.expert_path = robot_path;
        s.person_traj = person_traj;
        s.expert_actions = expert_actions;
        scenarios = [scenarios; s]; 

        T = size(robot_path, 1);

        for t = 1:T
            robot_xy = robot_path(t, :);

            p_row = person_traj(min(t, size(person_traj,1)), :);
            person = struct();
            person.x = p_row(1);
            person.y = p_row(2);
            person.vx = p_row(3);
            person.vy = p_row(4);
            person.pred_x = person.x + person.vx * 1.0;
            person.pred_y = person.y + person.vy * 1.0;

            X_i = zeros(A, F);

            for a = 1:A
                X_i(a,:) = compute_action_features_stage5( ...
                    robot_xy, goal_xy, person, action_waypoints(a,:), corridor_half_width);
            end

            y_i = expert_actions(t);

            X_list{end+1,1} = X_i; 
            y_list(end+1,1) = y_i; 

            % Sample weights.
            w = 1.0;

            if y_i == idx_slight_left || y_i == idx_slight_right
                w = 1.8;
            elseif y_i == idx_slow_forward
                w = 1.3;
            elseif y_i == idx_forward
                w = 1.0;
            end

            % Near-person decisions are more important.
            rel_px = person.x - robot_xy(1);
            rel_py = person.y - robot_xy(2);
            if rel_px > 0 && rel_px < 3.0 && abs(rel_py) < 0.65
                w = w * 1.5;
            end

            % Near-wall / recovery examples are important.
            if abs(robot_xy(2)) > 0.30
                w = w * 1.6;
            end

            if string(mode) == "near_wall_recovery"
                w = w * 2.0;
            end

            % Side and diagonal crossing cases are the new safety-critical
            % examples; give them enough influence during MaxEnt training.
            if any(string(mode) == ["cross_left_to_right", "cross_right_to_left", ...
                                    "diag_front_left", "diag_front_right"])
                w = w * 1.7;
            end

            sample_weight_list(end+1,1) = w; 
            source_type_list(end+1,1) = "synthetic"; 
            scenario_index_list(end+1,1) = scenario_id; 
        end
    end
end

%% Add explicit near-wall negative / recovery states

explicit_y_values = [-0.85, -0.70, -0.55, 0.55, 0.70, 0.85];
explicit_x_values = 0.5:0.5:7.0;

for y0 = explicit_y_values
    scenario_id = scenario_id + 1;

    for x0 = explicit_x_values
        robot_xy = [x0, y0];

        person = struct();
        person.x = 20.0;
        person.y = 0.0;
        person.vx = 0.0;
        person.vy = 0.0;
        person.pred_x = 20.0;
        person.pred_y = 0.0;

        X_i = zeros(A, F);
        for a = 1:A
            X_i(a,:) = compute_action_features_stage5( ...
                robot_xy, goal_xy, person, action_waypoints(a,:), corridor_half_width);
        end

        if y0 > 0
            y_i = idx_slight_right;
        else
            y_i = idx_slight_left;
        end

        X_list{end+1,1} = X_i; 
        y_list(end+1,1) = y_i; 
        sample_weight_list(end+1,1) = 4.0; 
        source_type_list(end+1,1) = "explicit_recovery"; 
        scenario_index_list(end+1,1) = scenario_id; 
    end
end

%% Convert to arrays
N = length(y_list);
X = zeros(N, A, F);

for i = 1:N
    X(i,:,:) = X_list{i};
end

y = y_list;
sample_weight = sample_weight_list;
source_type = source_type_list;
scenario_index = scenario_index_list;

fprintf("\nFinal Stage 5 dataset:\n");
fprintf("  N = %d samples\n", N);
fprintf("  A = %d actions\n", A);
fprintf("  F = %d features\n", F);

fprintf("\nAction counts:\n");
for a = 1:A
    fprintf("  %-14s : %d\n", action_names{a}, sum(y == a));
end

fprintf("\nWeighted action totals:\n");
for a = 1:A
    fprintf("  %-14s : %.2f\n", action_names{a}, sum(sample_weight(y == a)));
end

save("irl_dataset_stage5_4actions.mat", ...
    "X", "y", "sample_weight", "source_type", "scenario_index", ...
    "action_names", "action_waypoints", "feature_names", ...
    "goal_xy", "corridor_half_width", "scenarios");

fprintf("\nSaved dataset to irl_dataset_stage5_4actions.mat\n");

%% functions
function [robot_path, person_traj, expert_actions] = make_synthetic_scenario_stage5(mode, py0, goal_xy, action_names)

    idx_slow_forward = find(string(action_names) == "slow_forward", 1);
    idx_forward = find(string(action_names) == "forward", 1);
    idx_slight_left = find(string(action_names) == "slight_left", 1);
    idx_slight_right = find(string(action_names) == "slight_right", 1);

    % One synthetic step is 0.25 s and the expert moves at 0.8 m/s.  Human
    % positions are generated from the same time base as their vx/vy fields,
    % so the 1-second EKF prediction is physically consistent with the data.
    dt_s = 0.25;
    robot_cruise_speed = 0.80;
    x_grid = (0:robot_cruise_speed*dt_s:7.8)';
    T = length(x_grid);
    t_grid = (0:T-1)' * dt_s;
    variant = (py0 + 0.55) / 1.10; % deterministic 0..1 variation

    person_x = 20 * ones(T,1);
    person_y = zeros(T,1);
    person_vx = zeros(T,1);
    person_vy = zeros(T,1);

    y_path = zeros(T,1);
    mode = string(mode);

    switch mode
        case "no_person"
            y_path(:) = 0.0;

        case "side_clear"
            person_x(:) = 3.8;
            person_y(:) = sign_nonzero_local(py0) * 0.78;
            y_path(:) = 0.0;

        case "static"
            person_x(:) = 3.5;
            person_y(:) = py0;

            % Begin earlier and hold a wider clearance around static people.
            amp = 0.55;
            width = 1.25;
            direction = choose_avoid_direction_stage5(py0);
            y_path = direction * amp * exp(-((x_grid - 3.15).^2) / (2 * width^2));

        case "opposite"
            start_x = 4.6 + 0.5 * variant;
            person_vx(:) = -0.35;
            person_x = start_x + person_vx .* t_grid;
            person_y(:) = py0;

            % Head-on encounters need the earliest and largest lateral gap.
            amp = 0.58;
            width = 1.45;
            direction = choose_avoid_direction_stage5(py0);
            y_path = direction * amp * exp(-((x_grid - 2.65).^2) / (2 * width^2));

        case "same"
            start_x = 2.6 + 0.4 * variant;
            person_vx(:) = 0.18;
            person_x = start_x + person_vx .* t_grid;
            person_y(:) = py0;

            amp = 0.52;
            width = 1.45;
            direction = choose_avoid_direction_stage5(py0);
            y_path = direction * amp * exp(-((x_grid - 3.30).^2) / (2 * width^2));

        case "near_wall_recovery"
            % No person. Expert starts near side and returns to center.
            start_y = sign_nonzero_local(py0 + 0.01) * 0.60;
            y_path = start_y * exp(-x_grid / 1.40);

            person_x(:) = 20.0;
            person_y(:) = 0.0;

        case "cross_left_to_right"
            % Person starts on robot-left (+y) and crosses the corridor.
            start_x = 1.75 + 0.55 * variant;
            start_y = 0.72 + 0.18 * variant;
            person_vx(:) = 0.05;
            person_vy(:) = -(0.40 + 0.10 * variant);
            person_x = start_x + person_vx .* t_grid;
            person_y = start_y + person_vy .* t_grid;

            amp = 0.55;
            width = 1.10;
            y_path = -amp * exp(-((x_grid - (1.25 + 0.25 * variant)).^2) / (2 * width^2));

        case "cross_right_to_left"
            % Mirror image of cross_left_to_right.
            start_x = 1.75 + 0.55 * variant;
            start_y = -(0.72 + 0.18 * variant);
            person_vx(:) = 0.05;
            person_vy(:) = 0.40 + 0.10 * variant;
            person_x = start_x + person_vx .* t_grid;
            person_y = start_y + person_vy .* t_grid;

            amp = 0.55;
            width = 1.10;
            y_path = amp * exp(-((x_grid - (1.25 + 0.25 * variant)).^2) / (2 * width^2));

        case "diag_front_left"
            % Person approaches from the front-left toward the centreline.
            start_x = 2.55 + 0.45 * variant;
            start_y = 0.68 + 0.16 * variant;
            person_vx(:) = -0.32;
            person_vy(:) = -0.22;
            person_x = start_x + person_vx .* t_grid;
            person_y = start_y + person_vy .* t_grid;

            amp = 0.55;
            width = 1.30;
            y_path = -amp * exp(-((x_grid - (1.50 + 0.25 * variant)).^2) / (2 * width^2));

        case "diag_front_right"
            % Mirror image of diag_front_left.
            start_x = 2.55 + 0.45 * variant;
            start_y = -(0.68 + 0.16 * variant);
            person_vx(:) = -0.32;
            person_vy(:) = 0.22;
            person_x = start_x + person_vx .* t_grid;
            person_y = start_y + person_vy .* t_grid;

            amp = 0.55;
            width = 1.30;
            y_path = amp * exp(-((x_grid - (1.50 + 0.25 * variant)).^2) / (2 * width^2));

        case "diag_receding_left"
            % A front-left person moving away should not trigger needless
            % avoidance; it supplies negative examples for the new features.
            start_x = 1.70 + 0.45 * variant;
            start_y = 0.52 + 0.16 * variant;
            person_vx(:) = 0.35;
            person_vy(:) = 0.20;
            person_x = start_x + person_vx .* t_grid;
            person_y = start_y + person_vy .* t_grid;
            y_path(:) = 0.0;

        case "diag_receding_right"
            start_x = 1.70 + 0.45 * variant;
            start_y = -(0.52 + 0.16 * variant);
            person_vx(:) = 0.35;
            person_vy(:) = -0.20;
            person_x = start_x + person_vx .* t_grid;
            person_y = start_y + person_vy .* t_grid;
            y_path(:) = 0.0;

        otherwise
            y_path(:) = 0.0;
    end

    % Preserve wall clearance while allowing the expert to demonstrate a
    % meaningful social-distance offset.  At corridor_half_width = 1.0 this
    % leaves 0.45 m between the robot centre and either wall.
    y_path = max(min(y_path, 0.55), -0.55);

    robot_path = [x_grid, y_path];
    person_traj = [person_x, person_y, person_vx, person_vy];

    expert_actions = zeros(T,1);

    for t = 1:T
        current_y = robot_path(t,2);

        if t < T
            dy = robot_path(t+1,2) - robot_path(t,2);
        else
            dy = 0.0;
        end

        person_xy = person_traj(t,1:2);
        person_velocity = person_traj(t,3:4);
        robot_velocity = [robot_cruise_speed, dy / dt_s];
        [t_ca, d_ca] = closest_approach_stage5( ...
            robot_path(t,:), robot_velocity, person_xy, person_velocity, 1.0);

        % Slow down if a crossing/diagonal trajectory produces an imminent
        % close approach, even when the person is not currently straight ahead.
        collision_soon = t_ca <= 1.0 && d_ca < 0.55;

        if collision_soon && d_ca < 0.28 && t_ca < 0.35
            expert_actions(t) = idx_slow_forward;

        elseif dy > 0.010
            expert_actions(t) = idx_slight_left;

        elseif dy < -0.010
            expert_actions(t) = idx_slight_right;

        elseif collision_soon
            expert_actions(t) = idx_slow_forward;

        elseif abs(current_y) > 0.10
            % Explicit return-to-center.
            if current_y > 0
                expert_actions(t) = idx_slight_right;
            else
                expert_actions(t) = idx_slight_left;
            end

        else
            expert_actions(t) = idx_forward;
        end
    end

    % Near goal, go forward.
    expert_actions(x_grid > goal_xy(1)-0.8) = idx_forward;
end

function [t_ca, d_ca] = closest_approach_stage5(robot_xy, robot_velocity, person_xy, person_velocity, horizon_s)
    relative_position = person_xy - robot_xy;
    relative_velocity = person_velocity - robot_velocity;
    t_unclamped = -dot(relative_position, relative_velocity) / ...
        (dot(relative_velocity, relative_velocity) + 1e-6);
    t_ca = min(max(t_unclamped, 0.0), horizon_s);
    d_ca = norm(relative_position + t_ca * relative_velocity);
end

function direction = choose_avoid_direction_stage5(py)
    % If person is on left side (+y), robot avoids right (-y).
    % If person is on right side (-y), robot avoids left (+y).
    if abs(py) < 0.08
        direction = -1.0;
    elseif py > 0
        direction = -1.0;
    else
        direction = 1.0;
    end
end

function s = sign_nonzero_local(x)
    if x >= 0
        s = 1.0;
    else
        s = -1.0;
    end
end
