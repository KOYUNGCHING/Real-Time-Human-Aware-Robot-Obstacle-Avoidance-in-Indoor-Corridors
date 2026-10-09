clear; clc; close all;

% run_stage5_training_4actions.m

fprintf("\n==============================\n");
fprintf("Step 1: Build Stage 5 dataset\n");
fprintf("==============================\n");
build_irl_dataset_stage5_4actions;

fprintf("\n==============================\n");
fprintf("Step 2: Train Stage 5 IRL model\n");
fprintf("==============================\n");
train_irl_theta_stage5_4actions;

fprintf("\n==============================\n");
fprintf("Step 3: Evaluate Stage 5 rollout\n");
fprintf("==============================\n");
evaluate_stage5_policy_rollout;

fprintf("\n==============================\n");
fprintf("Step 4: Export Stage 5 JSON\n");
fprintf("==============================\n");
export_stage5_model_to_json;

fprintf("\nAll done.\n");
fprintf("Generated files:\n");
fprintf("  irl_dataset_stage5_4actions.mat\n");
fprintf("  irl_theta_stage5_4actions.mat\n");
fprintf("  stage5_rollout_summary.mat\n");
fprintf("  stage5_irl_model_4actions.json\n");
fprintf("  stage5_training_dashboard.mp4\n");
