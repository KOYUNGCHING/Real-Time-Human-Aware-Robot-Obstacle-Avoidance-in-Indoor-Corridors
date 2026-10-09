clear; clc; close all;

opts = struct();

opts.dataset_file = "irl_dataset_stage5_4actions.mat";
opts.output_model_file = "irl_theta_stage5_4actions.mat";

opts.random_seed = 42;
opts.val_ratio = 0.20;

opts.max_epochs = 1500;
opts.learning_rate = 0.018;
opts.l2_theta = 0.003;
opts.l2_bias = 0.001;

opts.use_class_weights = false;
opts.use_sample_weights = true;
opts.use_sign_projection = true;

opts.show_training_plots = true;
opts.plot_every = 1;
opts.print_every = 25;
opts.show_rollout_plots = false;
opts.rollout_plot_every = 100;
opts.rollout_plot_scenarios = 2;
opts.checkpoint_epochs = [1 50 100 250 500 1000 1500];
opts.save_training_video = true;
opts.training_video_file = "stage5_training_dashboard.mp4";
opts.video_frame_rate = 5;

opts.patience = 250;
opts.min_delta = 1e-4;

rng(opts.random_seed);

%% Load dataset
if ~isfile(opts.dataset_file)
    error("Cannot find %s. Please run build_irl_dataset_stage5_4actions.m first.", opts.dataset_file);
end

load(opts.dataset_file);

[N, A, F] = size(X);

fprintf("\nLoaded dataset: %s\n", opts.dataset_file);
fprintf("N = %d samples, A = %d actions, F = %d features\n", N, A, F);

fprintf("\nAction counts:\n");
for a = 1:A
    fprintf("  %-14s : %d\n", action_names{a}, sum(y == a));
end

%% Scenario-level split
unique_groups = unique(scenario_index);
num_groups = length(unique_groups);
perm = unique_groups(randperm(num_groups));
num_val = max(1, round(opts.val_ratio * num_groups));

val_groups = perm(1:num_val);
train_groups = perm(num_val+1:end);

train_mask = ismember(scenario_index, train_groups);
val_mask = ismember(scenario_index, val_groups);

X_train = X(train_mask,:,:);
y_train = y(train_mask);
w_train = sample_weight(train_mask);

X_val = X(val_mask,:,:);
y_val = y(val_mask);
w_val = sample_weight(val_mask);

N_train = length(y_train);
N_val = length(y_val);

fprintf("\nTrain samples = %d, Val samples = %d\n", N_train, N_val);

%% Normalize features using train data only
feature_mean = zeros(1, F);
feature_std = zeros(1, F);

X_train_norm = X_train;
X_val_norm = X_val;

for f = 1:F
    values = X_train(:,:,f);
    feature_mean(f) = mean(values(:));
    feature_std(f) = std(values(:));

    if feature_std(f) < 1e-8
        feature_std(f) = 1.0;
    end

    X_train_norm(:,:,f) = (X_train(:,:,f) - feature_mean(f)) / feature_std(f);
    X_val_norm(:,:,f) = (X_val(:,:,f) - feature_mean(f)) / feature_std(f);
end

fprintf("Feature normalization finished.\n");

%% Class weights
class_counts = zeros(A,1);
for a = 1:A
    class_counts(a) = sum(y_train == a);
end

class_weights = ones(A,1);

fprintf("\nTraining class counts:\n");
for a = 1:A
    fprintf("  %-14s : count=%5d\n", action_names{a}, class_counts(a));
end

%% Initialize model
theta = zeros(F,1);

weighted_counts = zeros(A,1);
for a = 1:A
    weighted_counts(a) = sum(w_train(y_train == a));
end

prior = weighted_counts / max(sum(weighted_counts), 1e-9);
prior(prior < 1e-6) = 1e-6;

action_bias = -log(prior);
action_bias = action_bias - mean(action_bias);

initial_theta = theta;
initial_action_bias = action_bias;

% Adam state
m_theta = zeros(F,1);
v_theta = zeros(F,1);
m_bias = zeros(A,1);
v_bias = zeros(A,1);

beta1 = 0.9;
beta2 = 0.999;
adam_eps = 1e-8;

train_loss_history = zeros(opts.max_epochs,1);
train_acc_history = zeros(opts.max_epochs,1);
val_loss_history = zeros(opts.max_epochs,1);
val_acc_history = zeros(opts.max_epochs,1);
grad_norm_history = zeros(opts.max_epochs,1);
theta_history = zeros(opts.max_epochs,F);
action_bias_history = zeros(opts.max_epochs,A);
train_cost_history = zeros(opts.max_epochs,1);
val_cost_history = zeros(opts.max_epochs,1);
train_reward_history = zeros(opts.max_epochs,1);
val_reward_history = zeros(opts.max_epochs,1);

best_val_loss = inf;
best_val_acc = -inf;
best_theta = theta;
best_action_bias = action_bias;
best_epoch = 1;
epochs_without_improvement = 0;

if opts.show_training_plots
    figTrain = figure("Name", "Stage 5 Training Dashboard", "Color", "w");
    set(figTrain, "Position", [80 80 1180 760]);
end

if opts.save_training_video
    if ~opts.show_training_plots
        error("opts.save_training_video requires opts.show_training_plots = true.");
    end

    videoDashboard = VideoWriter(opts.training_video_file, "MPEG-4");
    videoDashboard.FrameRate = opts.video_frame_rate;
    open(videoDashboard);
end

rollout_scenario_ids = [];
if opts.show_rollout_plots
    rollout_scenario_ids = val_groups(val_groups >= 1 & val_groups <= numel(scenarios));
    rollout_scenario_ids = rollout_scenario_ids(:)';
    if isempty(rollout_scenario_ids)
        rollout_scenario_ids = 1:min(opts.rollout_plot_scenarios, numel(scenarios));
    else
        rollout_scenario_ids = rollout_scenario_ids(1:min(opts.rollout_plot_scenarios, numel(rollout_scenario_ids)));
    end
end
checkpoint_records = struct("epoch", {}, "theta", {}, "action_bias", {});
checkpoint_records(end+1) = struct( ...
    "epoch", 0, ...
    "theta", initial_theta, ...
    "action_bias", initial_action_bias);

%% Training loop
for epoch = 1:opts.max_epochs

    [train_loss, train_acc, grad_theta, grad_bias, y_pred_train, prob_train] = ...
        loss_grad_stage5(X_train_norm, y_train, w_train, theta, action_bias, class_weights, opts);

    grad_norm = norm([grad_theta; grad_bias]);

    % Adam theta
    m_theta = beta1*m_theta + (1-beta1)*grad_theta;
    v_theta = beta2*v_theta + (1-beta2)*(grad_theta.^2);
    m_theta_hat = m_theta / (1 - beta1^epoch);
    v_theta_hat = v_theta / (1 - beta2^epoch);
    theta = theta - opts.learning_rate * m_theta_hat ./ (sqrt(v_theta_hat) + adam_eps);

    % Adam bias
    m_bias = beta1*m_bias + (1-beta1)*grad_bias;
    v_bias = beta2*v_bias + (1-beta2)*(grad_bias.^2);
    m_bias_hat = m_bias / (1 - beta1^epoch);
    v_bias_hat = v_bias / (1 - beta2^epoch);
    action_bias = action_bias - opts.learning_rate * m_bias_hat ./ (sqrt(v_bias_hat) + adam_eps);

    action_bias = action_bias - mean(action_bias);

    if opts.use_sign_projection
        theta = project_theta_stage5(theta, feature_names);
    end

    [val_loss, val_acc, ~, ~, y_pred_val, prob_val] = ...
        loss_grad_stage5(X_val_norm, y_val, w_val, theta, action_bias, class_weights, opts);

    train_loss_history(epoch) = train_loss;
    train_acc_history(epoch) = train_acc;
    val_loss_history(epoch) = val_loss;
    val_acc_history(epoch) = val_acc;
    grad_norm_history(epoch) = grad_norm;
    theta_history(epoch,:) = theta';
    action_bias_history(epoch,:) = action_bias';

    [train_cost, train_reward] = policy_cost_reward_stage5(X_train_norm, w_train, theta, action_bias, opts);
    [val_cost, val_reward] = policy_cost_reward_stage5(X_val_norm, w_val, theta, action_bias, opts);
    train_cost_history(epoch) = train_cost;
    val_cost_history(epoch) = val_cost;
    train_reward_history(epoch) = train_reward;
    val_reward_history(epoch) = val_reward;

    improved = false;
    if val_loss < best_val_loss - opts.min_delta
        improved = true;
    elseif abs(val_loss - best_val_loss) <= opts.min_delta && val_acc > best_val_acc
        improved = true;
    end

    if improved
        best_val_loss = val_loss;
        best_val_acc = val_acc;
        best_theta = theta;
        best_action_bias = action_bias;
        best_epoch = epoch;
        epochs_without_improvement = 0;
    else
        epochs_without_improvement = epochs_without_improvement + 1;
    end

    if epoch == 1 || mod(epoch, opts.print_every) == 0 || epoch == opts.max_epochs
        fprintf("epoch=%4d | train loss=%.4f acc=%.2f%% | val loss=%.4f acc=%.2f%% | grad=%.4f\n", ...
            epoch, train_loss, train_acc*100, val_loss, val_acc*100, grad_norm);
    end

    if opts.show_training_plots && (epoch == 1 || mod(epoch, opts.plot_every) == 0 || epoch == opts.max_epochs)
        update_training_plot_stage5(figTrain, epoch, train_loss_history, val_loss_history, ...
            train_acc_history, val_acc_history, train_cost_history, val_cost_history, ...
            train_reward_history, val_reward_history, grad_norm_history, theta_history, ...
            action_bias_history, feature_names, y_val, y_pred_val, action_names, ...
            checkpoint_records, scenarios, rollout_scenario_ids, theta, action_bias, ...
            feature_mean, feature_std, action_waypoints, goal_xy, corridor_half_width);
        if opts.save_training_video
            writeVideo(videoDashboard, getframe(figTrain));
        end
    end

    is_checkpoint = any(opts.checkpoint_epochs == epoch);
    if opts.show_rollout_plots && (is_checkpoint || mod(epoch, opts.rollout_plot_every) == 0 || epoch == opts.max_epochs)
        if isempty(checkpoint_records) || checkpoint_records(end).epoch ~= epoch
            checkpoint_records(end+1) = struct( ...
                "epoch", epoch, ...
                "theta", theta, ...
                "action_bias", action_bias); %#ok<SAGROW>
        end
        if opts.show_training_plots
            update_training_plot_stage5(figTrain, epoch, train_loss_history, val_loss_history, ...
                train_acc_history, val_acc_history, train_cost_history, val_cost_history, ...
                train_reward_history, val_reward_history, grad_norm_history, theta_history, ...
                action_bias_history, feature_names, y_val, y_pred_val, action_names, ...
                checkpoint_records, scenarios, rollout_scenario_ids, theta, action_bias, ...
                feature_mean, feature_std, action_waypoints, goal_xy, corridor_half_width);
            update_rollout_progress_windows_stage5(getappdata(figTrain, "stage5_latest_data"));
            if opts.save_training_video
                writeVideo(videoDashboard, getframe(figTrain));
            end
        end
    end

    if epochs_without_improvement >= opts.patience
        fprintf("\nEarly stopping at epoch %d. Best epoch = %d.\n", epoch, best_epoch);
        train_loss_history = train_loss_history(1:epoch);
        train_acc_history = train_acc_history(1:epoch);
        val_loss_history = val_loss_history(1:epoch);
        val_acc_history = val_acc_history(1:epoch);
        grad_norm_history = grad_norm_history(1:epoch);
        theta_history = theta_history(1:epoch,:);
        action_bias_history = action_bias_history(1:epoch,:);
        train_cost_history = train_cost_history(1:epoch);
        val_cost_history = val_cost_history(1:epoch);
        train_reward_history = train_reward_history(1:epoch);
        val_reward_history = val_reward_history(1:epoch);
        break;
    end
end

if opts.save_training_video
    close(videoDashboard);
    fprintf("\nSaved training dashboard video to %s\n", opts.training_video_file);
end

%% Use best model
theta = best_theta;
action_bias = best_action_bias;

[final_train_loss, final_train_acc, ~, ~, y_pred_train, prob_train] = ...
    loss_grad_stage5(X_train_norm, y_train, w_train, theta, action_bias, class_weights, opts);

[final_val_loss, final_val_acc, ~, ~, y_pred_val, prob_val] = ...
    loss_grad_stage5(X_val_norm, y_val, w_val, theta, action_bias, class_weights, opts);

conf_train = confusion_matrix_manual_stage5(y_train, y_pred_train, A);
conf_val = confusion_matrix_manual_stage5(y_val, y_pred_val, A);

fprintf("\n=====================================================\n");
fprintf("Stage 5 wall-aware 4-action training finished.\n");
fprintf("Best epoch = %d\n", best_epoch);
fprintf("Final train loss = %.4f, acc = %.2f%%\n", final_train_loss, final_train_acc*100);
fprintf("Final val loss   = %.4f, acc = %.2f%%\n", final_val_loss, final_val_acc*100);
fprintf("=====================================================\n\n");

fprintf("Learned theta cost weights:\n");
for f = 1:F
    fprintf("  %-32s : %+9.5f\n", feature_names{f}, theta(f));
end

fprintf("\nLearned action bias:\n");
for a = 1:A
    fprintf("  %-14s : %+9.5f\n", action_names{a}, action_bias(a));
end

fprintf("\nValidation confusion matrix: rows=true, cols=pred\n");
disp(conf_val);

fprintf("Per-action validation accuracy:\n");
for a = 1:A
    idx = (y_val == a);
    if sum(idx) == 0
        fprintf("  %-14s : no validation samples\n", action_names{a});
    else
        fprintf("  %-14s : %.2f%% (%d samples)\n", ...
            action_names{a}, mean(y_pred_val(idx) == y_val(idx))*100, sum(idx));
    end
end

%% Final diagnostic figures for non-interactive runs
if ~opts.show_training_plots
    figure("Name", "Stage 5 Validation Confusion Matrix", "Color", "w");
    imagesc(conf_val);
    axis equal tight;
    colorbar;
    title("Stage 5 Validation Confusion Matrix");
    xlabel("Predicted action");
    ylabel("Expert action");
    xticks(1:A); yticks(1:A);
    xticklabels(action_names); yticklabels(action_names);
    xtickangle(45);
    for r = 1:A
        for c = 1:A
            text(c, r, num2str(conf_val(r,c)), ...
                "HorizontalAlignment", "center", ...
                "Color", "w", ...
                "FontWeight", "bold");
        end
    end

    figure("Name", "Stage 5 Learned Theta", "Color", "w");
    bar(theta);
    grid on;
    title("Stage 5 Learned Cost Weights \theta");
    xticks(1:F);
    xticklabels(feature_names);
    xtickangle(45);
    ylabel("theta value");
end

if opts.show_rollout_plots
    if isempty(checkpoint_records) || checkpoint_records(end).epoch ~= best_epoch
        checkpoint_records(end+1) = struct( ...
            "epoch", best_epoch, ...
            "theta", theta, ...
            "action_bias", action_bias);
    end
    if opts.show_training_plots
        update_training_plot_stage5(figTrain, numel(train_loss_history), train_loss_history, val_loss_history, ...
            train_acc_history, val_acc_history, train_cost_history, val_cost_history, ...
            train_reward_history, val_reward_history, grad_norm_history, theta_history, ...
            action_bias_history, feature_names, y_val, y_pred_val, action_names, ...
            checkpoint_records, scenarios, rollout_scenario_ids, theta, action_bias, ...
            feature_mean, feature_std, action_waypoints, goal_xy, corridor_half_width);
        update_rollout_progress_windows_stage5(getappdata(figTrain, "stage5_latest_data"));
    end
end

%% Save model
final_train_acc_stage5 = final_train_acc;
final_val_acc_stage5 = final_val_acc;
final_train_loss_stage5 = final_train_loss;
final_val_loss_stage5 = final_val_loss;

save(opts.output_model_file, ...
    "theta", ...
    "action_bias", ...
    "feature_mean", ...
    "feature_std", ...
    "feature_names", ...
    "action_names", ...
    "action_waypoints", ...
    "class_counts", ...
    "class_weights", ...
    "train_loss_history", ...
    "train_acc_history", ...
    "val_loss_history", ...
    "val_acc_history", ...
    "grad_norm_history", ...
    "theta_history", ...
    "action_bias_history", ...
    "train_cost_history", ...
    "val_cost_history", ...
    "train_reward_history", ...
    "val_reward_history", ...
    "checkpoint_records", ...
    "final_train_acc_stage5", ...
    "final_val_acc_stage5", ...
    "final_train_loss_stage5", ...
    "final_val_loss_stage5", ...
    "conf_train", ...
    "conf_val", ...
    "y_pred_train", ...
    "y_pred_val", ...
    "prob_train", ...
    "prob_val", ...
    "opts", ...
    "goal_xy", ...
    "corridor_half_width");

fprintf("\nSaved Stage 5 model to %s\n", opts.output_model_file);

%% Helper functions
function [loss, acc, grad_theta, grad_bias, y_pred, prob_all] = ...
    loss_grad_stage5(X_data, y_data, sample_w, theta, action_bias, class_weights, opts)

    [N, A, F] = size(X_data);

    grad_theta = zeros(F,1);
    grad_bias = zeros(A,1);

    total_loss = 0.0;
    total_weight = 0.0;

    y_pred = zeros(N,1);
    prob_all = zeros(N,A);

    for i = 1:N
        features = squeeze(X_data(i,:,:)); % A x F
        expert_a = y_data(i);

        if opts.use_sample_weights
            sw = sample_w(i);
        else
            sw = 1.0;
        end

        if opts.use_class_weights
            cw = class_weights(expert_a);
        else
            cw = 1.0;
        end

        weight = sw * cw;

        costs = features * theta + action_bias;
        logits = -costs;
        logits = logits - max(logits);

        probs = exp(logits);
        probs = probs / sum(probs);

        prob_all(i,:) = probs;
        [~, y_pred(i)] = max(probs);

        total_loss = total_loss + weight * (-log(probs(expert_a) + 1e-12));
        total_weight = total_weight + weight;

        expert_feat = features(expert_a,:)';
        expected_feat = zeros(F,1);
        for a = 1:A
            expected_feat = expected_feat + probs(a) * features(a,:)';
        end

        grad_theta = grad_theta + weight * (expert_feat - expected_feat);

        for a = 1:A
            indicator = double(a == expert_a);
            grad_bias(a) = grad_bias(a) + weight * (indicator - probs(a));
        end
    end

    loss = total_loss / max(total_weight, 1e-12) + ...
        opts.l2_theta * sum(theta.^2) + opts.l2_bias * sum(action_bias.^2);

    grad_theta = grad_theta / max(total_weight, 1e-12) + 2 * opts.l2_theta * theta;
    grad_bias = grad_bias / max(total_weight, 1e-12) + 2 * opts.l2_bias * action_bias;

    acc = mean(y_pred == y_data);
end

function [mean_cost, mean_reward] = policy_cost_reward_stage5(X_data, sample_w, theta, action_bias, opts)
    [N, ~, ~] = size(X_data);
    total_cost = 0.0;
    total_weight = 0.0;

    for i = 1:N
        features = squeeze(X_data(i,:,:));
        costs = features * theta + action_bias;
        chosen_cost = min(costs);

        if opts.use_sample_weights
            weight = sample_w(i);
        else
            weight = 1.0;
        end

        total_cost = total_cost + weight * chosen_cost;
        total_weight = total_weight + weight;
    end

    mean_cost = total_cost / max(total_weight, 1e-12);
    mean_reward = -mean_cost;
end

function theta = project_theta_stage5(theta, feature_names)
    names = string(feature_names(:));

    nonnegative_features = [
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
        "return_to_center_cost"
        "outward_motion_cost"
        "dca_risk"
        "ttc_risk"
        "closing_speed"
        "safety_clearance_cost"
    ];

    for k = 1:numel(nonnegative_features)
        idx = find(names == nonnegative_features(k), 1);
        if ~isempty(idx)
            theta(idx) = max(theta(idx), 0.0);
        end
    end

    % opposite_side_bonus_as_cost is allowed to be positive theta with negative feature value.
end

function conf_mat = confusion_matrix_manual_stage5(y_true, y_pred, A)
    conf_mat = zeros(A,A);
    for i = 1:length(y_true)
        conf_mat(y_true(i), y_pred(i)) = conf_mat(y_true(i), y_pred(i)) + 1;
    end
end

function update_training_plot_stage5(figTrain, epoch, train_loss_history, val_loss_history, ...
    train_acc_history, val_acc_history, train_cost_history, val_cost_history, ...
    train_reward_history, val_reward_history, grad_norm_history, theta_history, ...
    action_bias_history, feature_names, y_val, y_pred_val, action_names, ...
    checkpoint_records, scenarios, rollout_scenario_ids, current_theta, current_action_bias, ...
    feature_mean, feature_std, action_waypoints, goal_xy, corridor_half_width)

    data = struct();
    data.epoch = epoch;
    data.train_loss_history = train_loss_history;
    data.val_loss_history = val_loss_history;
    data.train_acc_history = train_acc_history;
    data.val_acc_history = val_acc_history;
    data.train_cost_history = train_cost_history;
    data.val_cost_history = val_cost_history;
    data.train_reward_history = train_reward_history;
    data.val_reward_history = val_reward_history;
    data.grad_norm_history = grad_norm_history;
    data.theta_history = theta_history;
    data.action_bias_history = action_bias_history;
    data.feature_names = feature_names;
    data.y_val = y_val;
    data.y_pred_val = y_pred_val;
    data.action_names = action_names;
    data.checkpoint_records = checkpoint_records;
    data.scenarios = scenarios;
    data.rollout_scenario_ids = rollout_scenario_ids;
    data.current_theta = current_theta;
    data.current_action_bias = current_action_bias;
    data.feature_mean = feature_mean;
    data.feature_std = feature_std;
    data.action_waypoints = action_waypoints;
    data.goal_xy = goal_xy;
    data.corridor_half_width = corridor_half_width;

    setappdata(figTrain, "stage5_latest_data", data);
    render_training_viewer_stage5(figTrain);
end

function render_training_viewer_stage5(figTrain)
    if ~ishandle(figTrain) || ~isappdata(figTrain, "stage5_latest_data")
        return;
    end

    data = getappdata(figTrain, "stage5_latest_data");

    figure(figTrain);
    clf(figTrain);
    render_dashboard_view_stage5(data);
    drawnow;
end

function set_view_layout_position_stage5(tl)
    try
        tl.Units = "normalized";
        tl.Position = [0.05, 0.07, 0.90, 0.84];
    catch
        % Older MATLAB versions may not expose tiledlayout Position.
    end
end

function render_dashboard_view_stage5(data)
    epoch = data.epoch;
    t = 1:epoch;

    tl = tiledlayout(2, 3, "Padding", "compact", "TileSpacing", "compact");
    set_view_layout_position_stage5(tl);

    nexttile;
    plot(t, data.train_loss_history(t), "LineWidth", 1.8); hold on;
    plot(t, data.val_loss_history(t), "LineWidth", 1.8);
    grid on; xlabel("episode / epoch"); ylabel("NLL loss");
    legend("train", "val", "Location", "best"); title("Loss");

    nexttile;
    plot(t, data.train_cost_history(t), "LineWidth", 1.8); hold on;
    plot(t, data.val_cost_history(t), "LineWidth", 1.8);
    grid on; xlabel("episode / epoch"); ylabel("mean selected cost");
    legend("train", "val", "Location", "best"); title("Policy Cost");

    nexttile;
    plot(t, data.train_acc_history(t)*100, "LineWidth", 1.8); hold on;
    plot(t, data.val_acc_history(t)*100, "LineWidth", 1.8);
    grid on; xlabel("episode / epoch"); ylabel("accuracy (%)");
    legend("train", "val", "Location", "best"); title("Accuracy");

    nexttile;
    plot(t, data.train_reward_history(t), "LineWidth", 1.8); hold on;
    plot(t, data.val_reward_history(t), "LineWidth", 1.8);
    grid on; xlabel("episode / epoch"); ylabel("mean reward = -cost");
    legend("train", "val", "Location", "best"); title("Reward");

    nexttile;
    semilogy(t, data.grad_norm_history(t), "LineWidth", 1.8);
    grid on; xlabel("episode / epoch"); ylabel("grad norm"); title("Gradient Norm");

    nexttile;
    status_lines = {
        sprintf("epoch             %d", epoch)
        sprintf("train acc         %.2f%%", data.train_acc_history(epoch)*100)
        sprintf("val acc           %.2f%%", data.val_acc_history(epoch)*100)
        sprintf("val loss          %.4f", data.val_loss_history(epoch))
        sprintf("val cost          %.4f", data.val_cost_history(epoch))
        sprintf("val reward        %.4f", data.val_reward_history(epoch))
        sprintf("grad norm         %.4f", data.grad_norm_history(epoch))
    };
    text(0.05, 0.95, status_lines, "FontSize", 12, ...
        "FontName", "Monospaced", "VerticalAlignment", "top");
    axis off; title("Training Status");
    sgtitle("Stage 5 Training Dashboard", "FontWeight", "bold");
end

function render_weights_view_stage5(data)
    epoch = data.epoch;
    t = 1:epoch;

    tl = tiledlayout(2, 2, "Padding", "compact", "TileSpacing", "compact");
    set_view_layout_position_stage5(tl);

    nexttile;
    imagesc(t, 1:numel(data.feature_names), data.theta_history(t,:)');
    colorbar; xlabel("episode / epoch");
    yticks(1:numel(data.feature_names)); yticklabels(data.feature_names);
    title("\theta History Heatmap");

    nexttile;
    barh(data.theta_history(epoch,:));
    grid on; yticks(1:numel(data.feature_names)); yticklabels(data.feature_names);
    xlabel("theta value"); title("Current \theta");

    nexttile;
    plot(t, data.action_bias_history(t,:), "LineWidth", 1.6);
    grid on; xlabel("episode / epoch"); ylabel("bias");
    legend(data.action_names, "Location", "best", "Interpreter", "none");
    title("Action Bias History");

    nexttile;
    current_theta = data.theta_history(epoch,:);
    [~, theta_order] = sort(abs(current_theta), "descend");
    num_top_theta = min(8, numel(theta_order));
    top_lines = cell(num_top_theta + 1, 1);
    top_lines{1} = sprintf("Top |theta| at epoch %d", epoch);
    for k = 1:num_top_theta
        idx = theta_order(k);
        top_lines{k+1} = sprintf("%+8.4f  %s", current_theta(idx), data.feature_names{idx});
    end
    text(0.05, 0.95, top_lines, "FontSize", 11, ...
        "FontName", "Monospaced", "VerticalAlignment", "top", "Interpreter", "none");
    axis off; title("Largest Weights");
    sgtitle("Weights and Action Bias", "FontWeight", "bold");
end

function render_weights_diagnostics_view_stage5(data)
    epoch = data.epoch;
    t = 1:epoch;

    tl = tiledlayout(2, 3, "Padding", "compact", "TileSpacing", "compact");
    set_view_layout_position_stage5(tl);

    nexttile;
    imagesc(t, 1:numel(data.feature_names), data.theta_history(t,:)');
    colorbar;
    xlabel("episode / epoch");
    yticks(1:numel(data.feature_names));
    yticklabels(data.feature_names);
    title("\theta History Heatmap");

    nexttile;
    barh(data.theta_history(epoch,:));
    grid on;
    yticks(1:numel(data.feature_names));
    yticklabels(data.feature_names);
    xlabel("theta value");
    title("Current \theta");

    nexttile;
    plot(t, data.action_bias_history(t,:), "LineWidth", 1.6);
    grid on;
    xlabel("episode / epoch");
    ylabel("bias");
    legend(data.action_names, "Location", "best", "Interpreter", "none");
    title("Action Bias History");

    nexttile;
    bar(data.action_bias_history(epoch,:));
    grid on;
    xticks(1:numel(data.action_names));
    xticklabels(data.action_names);
    xtickangle(25);
    ylabel("bias");
    title("Current Action Bias");

    nexttile;
    A = numel(data.action_names);
    conf = confusion_matrix_manual_stage5(data.y_val, data.y_pred_val, A);
    imagesc(conf);
    axis equal tight;
    colorbar;
    xticks(1:A);
    yticks(1:A);
    xticklabels(data.action_names);
    yticklabels(data.action_names);
    xtickangle(35);
    xlabel("Predicted action");
    ylabel("Expert action");
    title(sprintf("Validation Confusion, epoch %d", epoch));

    for r = 1:A
        for c = 1:A
            text(c, r, num2str(conf(r,c)), ...
                "HorizontalAlignment", "center", ...
                "Color", "w", ...
                "FontWeight", "bold");
        end
    end

    nexttile;
    axis off;
    current_theta = data.theta_history(epoch,:);
    [~, theta_order] = sort(abs(current_theta), "descend");
    num_top_theta = min(6, numel(theta_order));
    top_lines = cell(num_top_theta + 1, 1);
    top_lines{1} = sprintf("Top |theta| at epoch %d", epoch);
    for k = 1:num_top_theta
        idx = theta_order(k);
        top_lines{k+1} = sprintf("%+8.4f  %s", current_theta(idx), data.feature_names{idx});
    end
    text(0.05, 0.95, top_lines, "FontSize", 10.5, ...
        "FontName", "Monospaced", "VerticalAlignment", "top", ...
        "Interpreter", "none");
    title("Largest Weights");

    sgtitle("Weights and Diagnostics", "FontWeight", "bold");
end

function render_confusion_view_stage5(data)
    tl = tiledlayout(1, 1, "Padding", "compact", "TileSpacing", "compact");
    set_view_layout_position_stage5(tl);
    nexttile;
    A = numel(data.action_names);
    conf = confusion_matrix_manual_stage5(data.y_val, data.y_pred_val, A);
    imagesc(conf);
    axis equal tight; colorbar;
    xticks(1:A); yticks(1:A);
    xticklabels(data.action_names); yticklabels(data.action_names);
    xtickangle(35);
    xlabel("Predicted action"); ylabel("Expert action");
    title(sprintf("Validation Confusion, epoch %d", data.epoch));

    for r = 1:A
        for c = 1:A
            text(c, r, num2str(conf(r,c)), ...
                "HorizontalAlignment", "center", ...
                "Color", "w", "FontWeight", "bold");
        end
    end
end

function render_rollout_view_stage5(data)
    scenario_ids = valid_rollout_scenario_ids_stage5(data);
    records = data.checkpoint_records;
    max_rows = min(4, numel(records));
    records = records(max(1, numel(records)-max_rows+1):end);
    max_cols = min(2, numel(scenario_ids));

    tl = tiledlayout(numel(records), max_cols, "Padding", "compact", "TileSpacing", "compact");
    set_view_layout_position_stage5(tl);

    for r = 1:numel(records)
        rec = records(r);
        for c = 1:max_cols
            s = scenario_ids(c);
            expert_path = data.scenarios(s).expert_path;
            person_traj = data.scenarios(s).person_traj;
            [robot_path, ~, ~] = rollout_policy_stage5( ...
                person_traj, rec.theta, rec.action_bias, data.feature_mean, data.feature_std, ...
                data.action_waypoints, data.action_names, data.goal_xy, data.corridor_half_width);
            nexttile;
            plot_one_rollout_stage5(robot_path, expert_path, person_traj, data.goal_xy, data.corridor_half_width);
            title(sprintf("epoch %d | scenario %d", rec.epoch, s), "FontSize", 9);
        end
    end
    sgtitle("Rollout Progress: blue=learned, magenta=expert, red=person", "FontWeight", "bold");
end

function render_before_after_view_stage5(data)
    scenario_ids = valid_rollout_scenario_ids_stage5(data);
    s = scenario_ids(1);
    expert_path = data.scenarios(s).expert_path;
    person_traj = data.scenarios(s).person_traj;
    before = data.checkpoint_records(1);
    mid = data.checkpoint_records(max(1, round(numel(data.checkpoint_records)/2)));

    tl = tiledlayout(1, 3, "Padding", "compact", "TileSpacing", "compact");
    set_view_layout_position_stage5(tl);

    [robot_before, ~, ~] = rollout_policy_stage5( ...
        person_traj, before.theta, before.action_bias, data.feature_mean, data.feature_std, ...
        data.action_waypoints, data.action_names, data.goal_xy, data.corridor_half_width);
    nexttile;
    plot_one_rollout_stage5(robot_before, expert_path, person_traj, data.goal_xy, data.corridor_half_width);
    title(sprintf("before training | scenario %d", s));

    [robot_mid, ~, ~] = rollout_policy_stage5( ...
        person_traj, mid.theta, mid.action_bias, data.feature_mean, data.feature_std, ...
        data.action_waypoints, data.action_names, data.goal_xy, data.corridor_half_width);
    nexttile;
    plot_one_rollout_stage5(robot_mid, expert_path, person_traj, data.goal_xy, data.corridor_half_width);
    title(sprintf("during training | epoch %d", mid.epoch));

    [robot_after, ~, ~] = rollout_policy_stage5( ...
        person_traj, data.current_theta, data.current_action_bias, data.feature_mean, data.feature_std, ...
        data.action_waypoints, data.action_names, data.goal_xy, data.corridor_half_width);
    nexttile;
    plot_one_rollout_stage5(robot_after, expert_path, person_traj, data.goal_xy, data.corridor_half_width);
    title("after / current training");

    sgtitle("Before, During, After Rollout vs Expert Path", "FontWeight", "bold");
end

function update_rollout_progress_windows_stage5(data)
    persistent figRollout figBeforeDuringAfter;

    scenario_ids = valid_rollout_scenario_ids_stage5(data);
    records = data.checkpoint_records;

    if isempty(figRollout) || ~ishandle(figRollout)
        figRollout = figure("Name", "Stage 5 Rollout Learning Progress", "Color", "w");
        set(figRollout, "Position", [120 120 1180 720]);
    else
        figure(figRollout);
        clf(figRollout);
    end

    max_rows = min(4, numel(records));
    records_to_plot = records(max(1, numel(records)-max_rows+1):end);
    max_cols = min(2, numel(scenario_ids));

    tiledlayout(numel(records_to_plot), max_cols, "Padding", "compact", "TileSpacing", "compact");

    for r = 1:numel(records_to_plot)
        rec = records_to_plot(r);
        for c = 1:max_cols
            s = scenario_ids(c);
            expert_path = data.scenarios(s).expert_path;
            person_traj = data.scenarios(s).person_traj;

            [robot_path, ~, ~] = rollout_policy_stage5( ...
                person_traj, rec.theta, rec.action_bias, data.feature_mean, data.feature_std, ...
                data.action_waypoints, data.action_names, data.goal_xy, data.corridor_half_width);

            nexttile;
            plot_one_rollout_stage5(robot_path, expert_path, person_traj, data.goal_xy, data.corridor_half_width);
            title(sprintf("epoch %d | scenario %d", rec.epoch, s), "FontSize", 9);
        end
    end

    sgtitle("Rollout Progress: blue=learned, magenta=expert, red=person", "FontWeight", "bold");
    drawnow;

    if isempty(figBeforeDuringAfter) || ~ishandle(figBeforeDuringAfter)
        figBeforeDuringAfter = figure("Name", "Stage 5 Before During After Rollout", "Color", "w");
        set(figBeforeDuringAfter, "Position", [100 100 1450 520]);
    else
        figure(figBeforeDuringAfter);
        clf(figBeforeDuringAfter);
    end

    render_before_after_view_stage5(data);
    drawnow;
end

function scenario_ids = valid_rollout_scenario_ids_stage5(data)
    scenario_ids = data.rollout_scenario_ids;
    scenario_ids = scenario_ids(scenario_ids >= 1 & scenario_ids <= numel(data.scenarios));
    if isempty(scenario_ids)
        scenario_ids = 1:min(2, numel(data.scenarios));
    end
end

function plot_one_rollout_stage5(robot_path, expert_path, person_traj, goal_xy, corridor_half_width)
    hold on; grid on; axis equal;

    plot([0 goal_xy(1)], [ corridor_half_width  corridor_half_width], "k--");
    plot([0 goal_xy(1)], [-corridor_half_width -corridor_half_width], "k--");
    plot(expert_path(:,1), expert_path(:,2), "m--", "LineWidth", 1.4);
    plot(robot_path(:,1), robot_path(:,2), "b-o", "LineWidth", 1.4, "MarkerSize", 2.8);
    plot(person_traj(:,1), person_traj(:,2), "r-x", "LineWidth", 1.1, "MarkerSize", 2.8);
    plot(0, 0, "go", "MarkerFaceColor", "g");
    plot(goal_xy(1), goal_xy(2), "bp", "MarkerFaceColor", "b", "MarkerSize", 9);

    xlabel("x");
    ylabel("y");
    xlim([0 8.2]);
    ylim([-1.15 1.15]);
end

function [robot_path, action_seq, cost_seq] = rollout_policy_stage5( ...
    person_traj, theta, action_bias, feature_mean, feature_std, ...
    action_waypoints, action_names, goal_xy, corridor_half_width)

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
    % Keep dashboard rollouts consistent with the ROS node and the final
    % evaluator; the old 0.30 m cap hid wider learned avoidance paths.
    same_side_block_limit_m = 0.45;

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

        if robot_xy(2) > same_side_block_limit_m && best_name == "slight_left"
            best_a = idx_forward;
        elseif robot_xy(2) < -same_side_block_limit_m && best_name == "slight_right"
            best_a = idx_forward;
        end

        if robot_xy(2) > 0.85 && any(action_names_str == "slight_right")
            best_a = find(action_names_str == "slight_right", 1);
        elseif robot_xy(2) < -0.85 && any(action_names_str == "slight_left")
            best_a = find(action_names_str == "slight_left", 1);
        end

        step_xy = action_waypoints(best_a, :);
        step_xy = reshape(step_xy, 1, 2);

        robot_xy = reshape(robot_xy, 1, 2);
        next_xy = robot_xy + step_xy;
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
