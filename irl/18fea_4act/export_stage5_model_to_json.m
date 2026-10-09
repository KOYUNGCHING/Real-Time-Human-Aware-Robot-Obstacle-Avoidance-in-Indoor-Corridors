clear; clc;

%% =====================================================
% export_stage5_model_to_json.m
% Export Stage 5 MATLAB model to ROS-compatible JSON.

model_file = "irl_theta_stage5_4actions.mat";

if ~isfile(model_file)
    error("Cannot find %s. Please run train_irl_theta_stage5_4actions.m first.", model_file);
end

load(model_file);

model = struct();

model.model_name = "stage5_4actions_all_angle_safety_clearance_maxent_cost_irl";
model.coordinate_frame = "x_forward_y_lateral";
model.corridor_half_width = corridor_half_width;

model.theta = theta(:)';
model.action_bias = action_bias(:)';

model.feature_mean = feature_mean(:)';
model.feature_std = feature_std(:)';

model.feature_names = cellstr(feature_names);
model.action_names = cellstr(action_names);
model.action_waypoints = action_waypoints;

if exist("final_train_acc_stage5", "var")
    model.final_train_acc = final_train_acc_stage5;
end

if exist("final_val_acc_stage5", "var")
    model.final_val_acc = final_val_acc_stage5;
end

if exist("class_counts", "var")
    model.class_counts = class_counts(:)';
end

if exist("class_weights", "var")
    model.class_weights = class_weights(:)';
end

if exist("conf_val", "var")
    model.validation_confusion_matrix = conf_val;
end

json_text = jsonencode(model, "PrettyPrint", true);

output_file = "stage5_irl_model_4actions.json";

fid = fopen(output_file, "w");
if fid == -1
    error("Cannot open output JSON file.");
end

fprintf(fid, "%s", json_text);
fclose(fid);

fprintf("Exported Stage 5 model to %s\n", output_file);

fprintf("\nModel summary:\n");
fprintf("Number of actions: %d\n", length(model.action_names));
fprintf("Number of features: %d\n", length(model.feature_names));

if isfield(model, "final_train_acc")
    fprintf("Final train accuracy: %.2f%%\n", model.final_train_acc * 100);
end

if isfield(model, "final_val_acc")
    fprintf("Final validation accuracy: %.2f%%\n", model.final_val_acc * 100);
end

fprintf("\nActions:\n");
for i = 1:length(model.action_names)
    fprintf("%d. %-14s -> waypoint = [%.2f, %.2f], bias = %+8.4f\n", ...
        i, model.action_names{i}, ...
        model.action_waypoints(i,1), model.action_waypoints(i,2), ...
        model.action_bias(i));
end

fprintf("\nTheta:\n");
for i = 1:length(model.feature_names)
    fprintf("%32s : %+9.5f\n", model.feature_names{i}, model.theta(i));
end
