clear; clc;

%% =====================================================
% evaluate_stage5_policy_rollout.m
% Roll out the learned Stage 5 policy.

dataset_file = "irl_dataset_stage5_4actions.mat";
model_file = "irl_theta_stage5_4actions.mat";

if ~isfile(dataset_file)
    error("Cannot find %s.", dataset_file);
end

if ~isfile(model_file)
    error("Cannot find %s. Run train_irl_theta_stage5_4actions.m first.", model_file);
end

load(dataset_file);
load(model_file);

%% Safety criteria (set these to the physical dimensions of your robot)
robot_radius_m = 0.20;
person_radius_m = 0.30;
safety_margin_m = 0.20;

collision_distance_m = robot_radius_m + person_radius_m;
avoidance_clearance_m = collision_distance_m + safety_margin_m;

% Match the ROS node's default same_side_block_limit.  The previous 0.30 m
% rollout-only cap prevented the learned policy from reproducing the wider
% expert paths used for safe avoidance.
same_side_block_limit_m = 0.45;

num_scenarios = length(scenarios);
max_plot = min(12, num_scenarios);
plot_ids = round(linspace(1, num_scenarios, max_plot));

fprintf("\nEvaluating Stage 5 policy rollout on %d scenarios...\n", num_scenarios);
fprintf("Collision distance = %.2f m | avoidance-clearance distance = %.2f m\n", ...
    collision_distance_m, avoidance_clearance_m);
fprintf("Same-side lateral limit = %.2f m\n", same_side_block_limit_m);

summary = struct();
summary.reached_goal = false(num_scenarios,1);
summary.max_abs_y = zeros(num_scenarios,1);
summary.min_wall_margin = zeros(num_scenarios,1);
summary.min_person_distance = zeros(num_scenarios,1);
summary.num_steps = zeros(num_scenarios,1);
summary.mode = strings(num_scenarios,1);
summary.has_person = false(num_scenarios,1);
summary.collision = false(num_scenarios,1);
summary.near_miss = false(num_scenarios,1);
summary.avoidance_success = false(num_scenarios,1);
summary.robot_radius_m = robot_radius_m;
summary.person_radius_m = person_radius_m;
summary.safety_margin_m = safety_margin_m;
summary.collision_distance_m = collision_distance_m;
summary.avoidance_clearance_m = avoidance_clearance_m;
summary.same_side_block_limit_m = same_side_block_limit_m;

figure("Name", "Stage 5 Policy Rollout", "Color", "w");
set(gcf, "Position", [80 80 1450 860]);
tiledlayout(3, 4, "Padding", "compact", "TileSpacing", "compact");

for s = 1:num_scenarios
    expert_path = scenarios(s).expert_path;
    person_traj = scenarios(s).person_traj;
    summary.mode(s) = string(scenarios(s).mode);

    [robot_path, action_seq, cost_seq] = rollout_one_scenario_stage5( ...
        person_traj, theta, action_bias, feature_mean, feature_std, ...
        action_waypoints, action_names, goal_xy, corridor_half_width, ...
        same_side_block_limit_m);

    summary.num_steps(s) = size(robot_path, 1);
    summary.reached_goal(s) = robot_path(end,1) >= goal_xy(1) - 0.25;
    summary.max_abs_y(s) = max(abs(robot_path(:,2)));
    summary.min_wall_margin(s) = corridor_half_width - summary.max_abs_y(s);
    summary.min_person_distance(s) = min_distance_to_person_stage5(robot_path, person_traj);

    % The far-away placeholder person is used for no_person and recovery
    % scenarios.  Do not let those navigation-only cases inflate the
    % avoidance success rate.
    summary.has_person(s) = ~any(summary.mode(s) == ["no_person", "near_wall_recovery"]);
    if summary.has_person(s)
        summary.collision(s) = summary.min_person_distance(s) < collision_distance_m;
        summary.near_miss(s) = ~summary.collision(s) && ...
            summary.min_person_distance(s) < avoidance_clearance_m;
        summary.avoidance_success(s) = summary.reached_goal(s) && ...
            summary.min_person_distance(s) >= avoidance_clearance_m;
    end

    if any(plot_ids == s)
        nexttile;
        hold on; grid on; axis equal;

        plot([0 goal_xy(1)], [ corridor_half_width  corridor_half_width], "k--");
        plot([0 goal_xy(1)], [-corridor_half_width -corridor_half_width], "k--");

        plot(expert_path(:,1), expert_path(:,2), "m--", "LineWidth", 1.4);
        plot(robot_path(:,1), robot_path(:,2), "b-o", "LineWidth", 1.5, "MarkerSize", 3);
        plot(person_traj(:,1), person_traj(:,2), "r-x", "LineWidth", 1.2, "MarkerSize", 3);

        plot(0, 0, "go", "MarkerFaceColor", "g");
        plot(goal_xy(1), goal_xy(2), "bp", "MarkerFaceColor", "b", "MarkerSize", 10);

        if ~summary.has_person(s)
            avoidance_label = "no person";
        elseif summary.collision(s)
            avoidance_label = "COLLISION";
        elseif summary.near_miss(s)
            avoidance_label = "near miss";
        elseif summary.avoidance_success(s)
            avoidance_label = "avoid OK";
        else
            avoidance_label = "goal failed";
        end

        title(sprintf("s=%d | %s | minD=%.2f", ...
            s, char(avoidance_label), summary.min_person_distance(s)), "FontSize", 8);
        xlabel("x"); ylabel("y");
        xlim([0 8.2]);
        ylim([-1.15 1.15]);
    end
end

sgtitle("Stage 5 rollout: blue=model, magenta=expert, red=person", "FontWeight", "bold");

fprintf("\nRollout summary:\n");
fprintf("Reached goal: %d / %d\n", sum(summary.reached_goal), num_scenarios);
fprintf("Mean max |y|: %.3f\n", mean(summary.max_abs_y));
fprintf("Worst max |y|: %.3f\n", max(summary.max_abs_y));
fprintf("Mean min person distance: %.3f\n", mean(summary.min_person_distance));
fprintf("Worst min person distance: %.3f\n", min(summary.min_person_distance));
fprintf("Mean wall margin: %.3f\n", mean(summary.min_wall_margin));
fprintf("Worst wall margin: %.3f\n", min(summary.min_wall_margin));

person_idx = summary.has_person;
num_person_scenarios = sum(person_idx);
num_collisions = sum(summary.collision(person_idx));
num_near_misses = sum(summary.near_miss(person_idx));
num_avoidance_successes = sum(summary.avoidance_success(person_idx));

summary.avoidance_success_rate = num_avoidance_successes / max(num_person_scenarios, 1);
summary.collision_rate = num_collisions / max(num_person_scenarios, 1);

fprintf("\nCollision / avoidance summary (person scenarios only):\n");
fprintf("Collision: %d / %d (%.1f%%)\n", ...
    num_collisions, num_person_scenarios, summary.collision_rate * 100);
fprintf("Near miss: %d / %d (minD >= %.2f m but < %.2f m)\n", ...
    num_near_misses, num_person_scenarios, collision_distance_m, avoidance_clearance_m);
fprintf("Avoidance success: %d / %d (%.1f%%)\n", ...
    num_avoidance_successes, num_person_scenarios, summary.avoidance_success_rate * 100);

fprintf("\nPer-approach-mode summary:\n");
for mode_name = unique(summary.mode)'
    idx = summary.mode == mode_name;
    mode_person_idx = idx & summary.has_person;
    if any(mode_person_idx)
        fprintf("  %-22s n=%2d | avoid=%d/%d | collision=%d | near-miss=%d | worst minD=%.3f\n", ...
            char(mode_name), sum(idx), sum(summary.avoidance_success(mode_person_idx)), ...
            sum(mode_person_idx), sum(summary.collision(mode_person_idx)), ...
            sum(summary.near_miss(mode_person_idx)), min(summary.min_person_distance(idx)));
    else
        fprintf("  %-22s n=%2d | navigation-only | reached=%d/%d\n", ...
            char(mode_name), sum(idx), sum(summary.reached_goal(idx)), sum(idx));
    end
end

save("stage5_rollout_summary.mat", "summary");
fprintf("\nSaved rollout summary to stage5_rollout_summary.mat\n");

%% Helper functions
function [robot_path, action_seq, cost_seq] = rollout_one_scenario_stage5( ...
    person_traj, theta, action_bias, feature_mean, feature_std, ...
    action_waypoints, action_names, goal_xy, corridor_half_width, ...
    same_side_block_limit_m)

    A = size(action_waypoints, 1);
    action_names_str = string(action_names);

    idx_forward = find(action_names_str == "forward", 1);
    if isempty(idx_forward)
        error("Cannot find forward action.");
    end

    robot_xy = [0.0, 0.0];
    robot_path = robot_xy;
    action_seq = [];
    cost_seq = [];

    max_steps = 70;

    for t = 1:max_steps
        idx = min(t, size(person_traj,1));

        person = struct();
        person.x = person_traj(idx,1);
        person.y = person_traj(idx,2);
        person.vx = person_traj(idx,3);
        person.vy = person_traj(idx,4);
        person.pred_x = person.x + person.vx * 1.0;
        person.pred_y = person.y + person.vy * 1.0;

        costs = zeros(A,1);

        for a = 1:A
            feat = compute_action_features_stage5(robot_xy, goal_xy, person, action_waypoints(a,:), corridor_half_width);
            feat_norm = (feat - feature_mean) ./ feature_std;
            costs(a) = feat_norm * theta + action_bias(a);
        end

        [~, best_a] = min(costs);
        best_name = action_names_str(best_a);

        % Keep this guard aligned with success.py.  It still prevents the
        % robot from drifting toward a wall, but no longer caps it at 0.30 m.
        if robot_xy(2) > same_side_block_limit_m && best_name == "slight_left"
            best_a = idx_forward;
        elseif robot_xy(2) < -same_side_block_limit_m && best_name == "slight_right"
            best_a = idx_forward;
        end

        % Emergency near-wall correction in rollout only.
        if robot_xy(2) > 0.85 && any(action_names_str == "slight_right")
            best_a = find(action_names_str == "slight_right", 1);
        elseif robot_xy(2) < -0.85 && any(action_names_str == "slight_left")
            best_a = find(action_names_str == "slight_left", 1);
        end

        step_xy = action_waypoints(best_a, :);
        step_xy = reshape(step_xy, 1, 2);

        robot_xy = reshape(robot_xy, 1, 2);
        next_xy = robot_xy + step_xy;

        % Clamp only for numerical rollout safety.
        next_xy(2) = max(min(next_xy(2), 0.95), -0.95);

        robot_xy = next_xy;
        robot_path = [robot_path; robot_xy]; %#ok<AGROW>
        action_seq = [action_seq; best_a]; %#ok<AGROW>
        cost_seq = [cost_seq; costs']; %#ok<AGROW>

        if robot_xy(1) >= goal_xy(1) - 0.25
            break;
        end
    end
end

function dmin = min_distance_to_person_stage5(robot_path, person_traj)
    dmin = inf;
    T = min(size(robot_path,1), size(person_traj,1));
    for t = 1:T
        d = norm(robot_path(t,:) - person_traj(t,1:2));
        dmin = min(dmin, d);
    end
end
