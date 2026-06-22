clear; clc; close all;

%% Load data
load('10s_2p_2.mat');          % 載入 LiDAR xy_data
load('10s_2p_2_label.mat');    % 載入人工標記的 leg_labels

val = xy_data;                      % val: N x 2 x T，每一幀的雷射點
T = size(val, 3);                   % 總 frame 數
dt = 0.1;                           % 每一幀間隔時間，假設 0.1 秒

%% EKF / tracking parameters
Q = diag([0.01, 0.01, 0.2, 0.1, 0.1]);  % process noise，控制預測不確定性
R = eye(2) * 0.05;                       % measurement noise，只量測 x,y

predict_time = 0.3;                      % 顯示未來預測 0.1 秒
steps_future = round(predict_time / dt); % 預測幾步

dist_gate = 0.8;          % 配對最大距離，超過就不配
merge_dist = 0.8;        % 兩隻腳合併成人中心的距離門檻
max_targets = 2;          % 最多追蹤兩個人
max_missed = 5;           % track 連續沒配到 measurement 幾次後刪除

% Collision risk parameters
risk_horizon = 1.0;                 % 看未來 1 秒內會不會撞
risk_steps = round(risk_horizon / dt);

robot_radius = 0.25;                % 機器人半徑，可依實際改
human_radius = 0.25;                % 人的半徑，可依實際改
safety_margin = 0.20;               % 安全距離
collision_radius = robot_radius + human_radius + safety_margin;

num_mc = 300;                       % Monte Carlo sample 數
stop_threshold = 0.7;               % >70% 停止
slow_threshold = 0.4;               % >40% 減速
nextTrackID = 1;          % 下一個 track ID

% 每一個 Track 都有：
% id      : 身分編號
% X       : EKF state = [x; y; v; theta; omega]
% P       : covariance
% age     : 存活多久
% miss    : 連續幾幀沒配到 detection
% history : 軌跡紀錄
Tracks = struct('id', {}, 'X', {}, 'P', {}, 'age', {}, 'miss', {}, 'history', {});

%% Plot range
all_x = val(:,1,:);
all_y = val(:,2,:);
all_x = all_x(:);
all_y = all_y(:);

good = isfinite(all_x) & isfinite(all_y) & ~(all_x==0 & all_y==0);

xmin = min(all_x(good)) - 0.5;
xmax = max(all_x(good)) + 0.5;
ymin = min(all_y(good)) - 0.5;
ymax = max(all_y(good)) + 0.5;

%% main loop
fig = figure('Position', [100 100 1300 800]);

for t = 1:T
    clf(fig);
    hold on; grid on;
    axis equal;
    xlim([xmin xmax]);
    ylim([ymin ymax]);

    % 取出當前 frame 的 LiDAR 點
    xy = squeeze(val(:,:,t));

    % 過濾無效點：
    % 1. 不能是 NaN 或 Inf
    % 2. 不能是 (0,0)
    valid = isfinite(xy(:,1)) & isfinite(xy(:,2)) & ...
            ~(xy(:,1)==0 & xy(:,2)==0);
    xy_valid = xy(valid,:);

    % 如果這一幀沒有有效點，就直接跳過
    if isempty(xy_valid)
        plot(0,0,'ks','MarkerSize',10,'MarkerFaceColor','k');
        text(0.1,-0.2,'Robot','FontWeight','bold');
        title(sprintf('EKF + Hungarian Multi-Target Tracking | Frame %d / %d', t, T), ...
            'FontSize', 14);
        drawnow;
        pause(0.1);
        continue;
    end

    % 畫出所有 LiDAR 點
    plot(xy_valid(:,1), xy_valid(:,2), '.', 'Color', [0.85 0.85 0.85]);

    % Segmentation
    % Segment() 會把 LiDAR 點切成一段一段
    [Seg, Si_n, S_n] = Segment(xy_valid);

    % 讀取人工標記的人腳 segment
    trueFootIDs = [];

    if t <= numel(leg_labels) && ~isempty(leg_labels{t})
        tmp = leg_labels{t};

        % 只保留合法 segment index
        tmp = tmp(tmp >= 1 & tmp <= S_n);

        trueFootIDs = tmp(:)';
    end

    % 將 labeled leg segment 轉成 centroid
    seg_centers = [];      % 每個 leg segment 的中心點
    seg_pts_cell = {};     % 每個 leg segment 的原始點，之後畫圖用

    if ~isempty(trueFootIDs)
        for j = trueFootIDs
            idxs = Seg(:,j);
            idxs = idxs(idxs > 0);

            if isempty(idxs)
                continue;
            end

            pts = xy_valid(idxs, :);   % 第 j 個 leg segment 的所有點
            c = mean(pts, 1);          % segment centroid

            seg_centers = [seg_centers; c]; 
            seg_pts_cell{end+1} = pts;
        end
    end

    % 將兩隻腳合併成人中心
    [person_meas, group_ids] = merge_close_points(seg_centers, merge_dist); 

    % 如果偵測到超過 max_targets 個人，只保留離 robot 最近的幾個
    if size(person_meas,1) > max_targets
        d_robot = vecnorm(person_meas, 2, 2);
        [~, ord] = sort(d_robot, 'ascend');
        keep_idx = ord(1:max_targets);
        person_meas = person_meas(keep_idx, :);
    end

    % EKF Prediction
    % 這一步是：每個舊 track 先預測自己下一幀會在哪裡
    for i = 1:numel(Tracks)
        [Tracks(i).X, Tracks(i).P] = ekf_predict_ctrv(Tracks(i).X, Tracks(i).P, Q, dt);
    end

    %% Data Association using Hungarian Algorithm

    % 原本你用 greedy nearest-neighbor
    % 現在改成 Hungarian algorithm 做全域最佳配對

    N = numel(Tracks);             % track 數量
    M = size(person_meas, 1);      % measurement 數量

    matched_pairs = [];            % 每列是 [track_index, measurement_index]
    unmatched_tracks = 1:N;        % 預設所有 track 都還沒配到
    unmatched_meas = 1:M;          % 預設所有 measurement 都還沒配到

    if N > 0 && M > 0

        % 建立 cost matrix
        % cost(i,j) = 第 i 個 track 預測位置 到 第 j 個 measurement 的距離
        cost = inf(N, M);

        for i = 1:N
            pred_pos = Tracks(i).X(1:2)';

            for j = 1:M
                z = person_meas(j, :);

                % 用 Euclidean distance 當配對成本
                d = norm(z - pred_pos);

                % 如果距離小於 gate，才允許配對
                if d < dist_gate
                    cost(i,j) = d;
                end
            end
        end

        % 使用 Hungarian algorithm 找最佳配對
        matched_pairs = hungarian_match(cost);

        % 如果有配對成功，更新 unmatched list
        if ~isempty(matched_pairs)
            matched_tracks = matched_pairs(:,1)';
            matched_meas_ids = matched_pairs(:,2)';

            unmatched_tracks = setdiff(1:N, matched_tracks);
            unmatched_meas = setdiff(1:M, matched_meas_ids);
        end
    end

    %% EKF Update for matched tracks
    % 對成功配對的 track，用 measurement 修正 EKF 狀態
    for k = 1:size(matched_pairs,1)
        ti = matched_pairs(k,1);   % track index
        mj = matched_pairs(k,2);   % measurement index

        z = person_meas(mj, :)';

        [Tracks(ti).X, Tracks(ti).P] = ekf_update_pos(Tracks(ti).X, Tracks(ti).P, z, R);

        Tracks(ti).age = Tracks(ti).age + 1;
        Tracks(ti).miss = 0;
        Tracks(ti).history = [Tracks(ti).history; Tracks(ti).X(1:2)'];
    end

    % Handle unmatched tracks
    % 沒有配到 measurement 的 track：
    % 只保留 EKF prediction，不做 update
    % miss + 1
    for idx = unmatched_tracks
        Tracks(idx).age = Tracks(idx).age + 1;
        Tracks(idx).miss = Tracks(idx).miss + 1;
        Tracks(idx).history = [Tracks(idx).history; Tracks(idx).X(1:2)'];
    end

    % Birth new tracks
    % 沒有配到任何舊 track 的 measurement：
    % 視為新出現的人，建立新 track
    for mj = unmatched_meas
        z = person_meas(mj, :)';

        X0 = [z(1); z(2); 0; 0; 1e-4];  % 初始速度設 0，角速度給很小值
        P0 = eye(5) * 0.1;

        Tracks(end+1).id = nextTrackID; 
        Tracks(end).X = X0;
        Tracks(end).P = P0;
        Tracks(end).age = 1;
        Tracks(end).miss = 0;
        Tracks(end).history = z';

        nextTrackID = nextTrackID + 1;
    end

    % Delete dead tracks
    % 如果 track 連續太多幀沒配到 measurement，就刪掉
    keep = true(1, numel(Tracks));

    for i = 1:numel(Tracks)
        if Tracks(i).miss > max_missed
            keep(i) = false;
        end
    end

    Tracks = Tracks(keep);

    %% Plot results

    % 畫 labeled foot points
    for j = 1:numel(seg_pts_cell)
        pts = seg_pts_cell{j};
        plot(pts(:,1), pts(:,2), 'r.', 'MarkerSize', 8);
    end

    % 畫 leg segment centroid
    if ~isempty(seg_centers)
        plot(seg_centers(:,1), seg_centers(:,2), 'mo', ...
            'MarkerSize', 6, 'LineWidth', 1.0);
    end

    % 畫合併後的人中心 measurement
    if ~isempty(person_meas)
        plot(person_meas(:,1), person_meas(:,2), 'ko', ...
            'MarkerSize', 10, 'LineWidth', 1.8);
    end

    % 畫 EKF tracks
    cmap = lines(max(10, numel(Tracks)));

    for i = 1:numel(Tracks)
        Xnow = Tracks(i).X;
        Pnow = Tracks(i).P;
        color_i = cmap(mod(Tracks(i).id-1, size(cmap,1))+1, :);

        % 目前 track 位置
        plot(Xnow(1), Xnow(2), 'o', ...
            'Color', color_i, ...
            'MarkerSize', 8, ...
            'MarkerFaceColor', color_i);

        % 目前 covariance ellipse
        [xe_now, ye_now] = ellipse_points(Xnow(1:2), Pnow(1:2,1:2));
        %plot(xe_now, ye_now, '-', 'Color', color_i, 'LineWidth', 1.5);

        % 未來位置預測
        [Xfut, Pfut] = predict_future(Xnow, Pnow, Q, dt, steps_future);
        %% Collision probability for next 1 second
        [Xrisk, Prisk] = predict_future(Xnow, Pnow, Q, dt, risk_steps);
        
        [collision_prob, tcpa, dcpa] = estimate_collision_probability_tcpa( ...
            Xnow, Pnow, collision_radius, risk_horizon, num_mc);
        
        if collision_prob >= stop_threshold
            decision = 'STOP';
            risk_color = [1 0 0];       % red
        elseif collision_prob >= slow_threshold
            decision = 'SLOW';
            risk_color = [1 0.5 0];     % orange
        else
            decision = 'GO';
            risk_color = [0 0.7 0];     % green
        end
        % 用星星標出 0.5 秒後的位置

        plot(Xfut(1), Xfut(2), '*', ...
            'Color', color_i, ...
            'MarkerSize', 12, ...
            'LineWidth', 1.8);
        % 未來 covariance ellipse
        [xe_f, ye_f] = ellipse_points(Xfut(1:2), Pfut(1:2,1:2));
        plot(xe_f, ye_f, '--', 'Color', color_i, 'LineWidth', 1.2);

        % 目前位置到未來預測位置的線
        plot([Xnow(1) Xfut(1)], [Xnow(2) Xfut(2)], '--', ...
            'Color', color_i, 'LineWidth', 1.2);

        % ID label
        text(Xnow(1)+0.05, Xnow(2)+0.05, ...
            sprintf('ID %d | Risk %.0f%% | %s', Tracks(i).id, collision_prob*100, decision), ...
            'Color', risk_color, 'FontWeight', 'bold');
    end

    %% Legend
    h_laser   = plot(nan,nan,'.','Color',[0.85 0.85 0.85],'MarkerSize',10);
    h_foot    = plot(nan,nan,'r.','MarkerSize',8);
    h_seg     = plot(nan,nan,'mo','MarkerSize',6,'LineWidth',1);
    h_person  = plot(nan,nan,'ko','MarkerSize',10,'LineWidth',1.8);
    h_track   = plot(nan,nan,'b-','LineWidth',1.5);

    legend([h_laser, h_foot, h_seg, h_person, h_track], ...
        {'Laser points', ...
         'Labeled foot points', ...
         'Segment centroids', ...
         'Merged person center', ...
         'EKF track'}, ...
        'Location', 'southoutside', ...
        'Orientation', 'horizontal', ...
        'NumColumns', 3);

    %% Robot origin
    plot(0,0,'ks','MarkerSize',10,'MarkerFaceColor','k');
    text(0.1,-0.2,'Robot','FontWeight','bold');
    theta_circle = linspace(0, 2*pi, 100);
    plot(collision_radius*cos(theta_circle), ...
         collision_radius*sin(theta_circle), ...
         'k--', 'LineWidth', 1.2);
    title(sprintf('EKF + Hungarian Multi-Target Tracking | Frame %d / %d | Tracks = %d', ...
        t, T, numel(Tracks)), 'FontSize', 14);

    drawnow;
    pause(0.5);
end

%% Functions
function matched_pairs = hungarian_match(cost)

    matched_pairs = [];

    if isempty(cost)
        return;
    end

    % 如果 MATLAB 有 matchpairs，就直接使用
    % matchpairs 是 MATLAB Statistics and Machine Learning Toolbox 的函式
    if exist('matchpairs', 'file') == 2

        % matchpairs(cost, maxCost)
        % maxCost 設為很大的數，因為我們已經用 inf 做 gate
        pairs = matchpairs(cost, 1e9);

        % 移除 inf 配對
        for k = 1:size(pairs,1)
            i = pairs(k,1);
            j = pairs(k,2);

            if isfinite(cost(i,j))
                matched_pairs = [matched_pairs; i, j];
            end
        end

    else
        % 如果沒有 matchpairs，就用簡化版 assignment
        % 因為你的 max_targets = 2，目標數很少，這個版本夠用
        matched_pairs = simple_assignment(cost);
    end
end

function matched_pairs = simple_assignment(cost)

    [N, M] = size(cost);
    matched_pairs = [];

    K = min(N, M);

    best_cost = inf;
    best_pairs = [];

    % 如果 track 比 measurement 少或相等
    if N <= M
        meas_perm_all = nchoosek(1:M, N);

        for r = 1:size(meas_perm_all,1)
            meas_set = meas_perm_all(r,:);
            perms_meas = perms(meas_set);

            for p = 1:size(perms_meas,1)
                pairs = [(1:N)', perms_meas(p,:)'];
                total = 0;
                valid = true;

                for k = 1:size(pairs,1)
                    c = cost(pairs(k,1), pairs(k,2));
                    if isinf(c)
                        valid = false;
                        break;
                    end
                    total = total + c;
                end

                if valid && total < best_cost
                    best_cost = total;
                    best_pairs = pairs;
                end
            end
        end

    else
        track_set_all = nchoosek(1:N, M);

        for r = 1:size(track_set_all,1)
            track_set = track_set_all(r,:);
            perms_track = perms(track_set);

            for p = 1:size(perms_track,1)
                pairs = [perms_track(p,:)', (1:M)'];
                total = 0;
                valid = true;

                for k = 1:size(pairs,1)
                    c = cost(pairs(k,1), pairs(k,2));
                    if isinf(c)
                        valid = false;
                        break;
                    end
                    total = total + c;
                end

                if valid && total < best_cost
                    best_cost = total;
                    best_pairs = pairs;
                end
            end
        end
    end

    matched_pairs = best_pairs;
end

function [merged_pts, group_ids] = merge_close_points(pts, merge_dist)

    if isempty(pts)
        merged_pts = [];
        group_ids = [];
        return;
    end

    N = size(pts,1);
    used = false(N,1);
    merged_pts = [];
    group_ids = zeros(N,1);
    gid = 0;

    for i = 1:N
        if used(i)
            continue;
        end

        gid = gid + 1;
        members = i;
        used(i) = true;

        changed = true;

        while changed
            changed = false;

            for j = 1:N
                if used(j)
                    continue;
                end

                d = vecnorm(pts(j,:) - pts(members,:), 2, 2);

                if any(d < merge_dist)
                    members(end+1) = j; 
                    used(j) = true;
                    changed = true;
                end
            end
        end

        merged_pts(gid, :) = mean(pts(members,:), 1); 
        group_ids(members) = gid;
    end
end

function [X_pred, P_pred] = ekf_predict_ctrv(X, P, Q, dt)

    v  = X(3);
    th = X(4);
    om = X(5);

    if abs(om) < 1e-3
        % 如果 omega 很小，近似成直線運動
        X_pred = [X(1) + v*cos(th)*dt;
                  X(2) + v*sin(th)*dt;
                  X(3);
                  X(4);
                  X(5)];

        F = eye(5);
        F(1,3) =  cos(th)*dt;
        F(1,4) = -v*sin(th)*dt;
        F(2,3) =  sin(th)*dt;
        F(2,4) =  v*cos(th)*dt;

    else
        % 如果 omega 不小，使用 CTRV nonlinear motion model
        X_pred = [X(1) + (v/om)*(sin(th+om*dt)-sin(th));
                  X(2) + (v/om)*(-cos(th+om*dt)+cos(th));
                  X(3);
                  X(4) + om*dt;
                  X(5)];

        F = eye(5);

        F(1,3) = (sin(th+om*dt)-sin(th))/om;
        F(1,4) = (v/om)*(cos(th+om*dt)-cos(th));
        F(1,5) = (v*dt*cos(th+om*dt)/om) ...
               - (v*(sin(th+om*dt)-sin(th))/om^2);

        F(2,3) = (-cos(th+om*dt)+cos(th))/om;
        F(2,4) = (v/om)*(sin(th+om*dt)-sin(th));
        F(2,5) = (v*dt*sin(th+om*dt)/om) ...
               - (v*(-cos(th+om*dt)+cos(th))/om^2);

        F(4,5) = dt;
    end

    % covariance prediction
    P_pred = F * P * F' + Q;

    % 將角度限制在 -pi 到 pi
    X_pred(4) = atan2(sin(X_pred(4)), cos(X_pred(4)));
end

function [X_upd, P_upd] = ekf_update_pos(X, P, z, R)
    % EKF update using position measurement z = [x; y]

    H = [1 0 0 0 0;
         0 1 0 0 0];

    z_hat = X(1:2);              % 預測量測位置
    S = H * P * H' + R;          % innovation covariance
    K = P * H' / S;              % Kalman gain

    innovation = z - z_hat;      % measurement residual

    X_upd = X + K * innovation;  % state update
    P_upd = (eye(5) - K * H) * P;

    X_upd(4) = atan2(sin(X_upd(4)), cos(X_upd(4)));
end

function [X_fut, P_fut] = predict_future(X, P, Q, dt, steps_future)
    % 往未來多預測幾步，只用來畫圖，不影響真正 tracking

    X_fut = X;
    P_fut = P;

    for s = 1:steps_future
        [X_fut, P_fut] = ekf_predict_ctrv(X_fut, P_fut, Q, dt);
    end
end

function [x_plot, y_plot] = ellipse_points(mu, Sigma)
    % 畫 2D covariance ellipse

    s = 2.279; % 95% confidence interval in 2D
    Sigma = (Sigma + Sigma') / 2;

    [V, D] = eig(Sigma * s);
    t = linspace(0, 2*pi, 60);

    a = sqrt(max(D(1,1), 0));
    b = sqrt(max(D(2,2), 0));

    xy = [a*cos(t); b*sin(t)];
    xy_rot = V * xy + mu(:);

    x_plot = xy_rot(1,:);
    y_plot = xy_rot(2,:);
end

function vxy = velocity_from_state(X)
    % CTRV state:
    % X = [x; y; v; theta; omega]
    speed = X(3);
    theta = X(4);

    vx = speed * cos(theta);
    vy = speed * sin(theta);

    vxy = [vx; vy];
end

function [tcpa, dcpa] = compute_tcpa_dcpa(pos, vel, horizon)
    % TCPA: Time to Closest Point of Approach
    % DCPA: Distance at Closest Point of Approach
    %
    % pos: 人相對機器人的位置 [x; y]
    % vel: 人相對機器人的速度 [vx; vy]

    EPS = 1e-9;

    v_norm2 = vel' * vel;

    if v_norm2 < EPS
        tcpa = 0;
        dcpa = norm(pos);
        return;
    end

    % 最近接近時間
    tcpa = - (pos' * vel) / v_norm2;

    % 只看未來 horizon 秒內
    tcpa = max(0, min(horizon, tcpa));

    % 最近接近點
    closest_point = pos + vel * tcpa;

    % 最近距離
    dcpa = norm(closest_point);
end

function [prob, tcpa_mean, dcpa_mean] = estimate_collision_probability_tcpa( ...
    X, P, collision_radius, horizon, num_mc)

    % 用目前 EKF mean state 先算一次 TCPA / DCPA
    pos_mean = X(1:2);
    vel_mean = velocity_from_state(X);

    [tcpa_mean, dcpa_mean] = compute_tcpa_dcpa(pos_mean, vel_mean, horizon);

    % covariance 數值穩定化
    P_safe = (P + P') / 2;
    P_safe = P_safe + eye(size(P_safe)) * 1e-6;

    % 不使用 mvnrnd，自己產生 Gaussian samples
    % X_sample = X + A * randn(5,1)，其中 A*A' = P_safe
    [V, D] = eig(P_safe);
    D = max(D, 0);
    A = V * sqrt(D);

    collision_count = 0;

    for n = 1:num_mc
        % 從 N(X, P) 抽一個樣本，不需要 Statistics Toolbox
        X_sample = X(:) + A * randn(5, 1);

        pos = X_sample(1:2);
        vel = velocity_from_state(X_sample);

        [~, dcpa] = compute_tcpa_dcpa(pos, vel, horizon);

        % approaching_speed > 0 代表人正在往機器人靠近
        approaching_speed = - dot(pos, vel) / (norm(pos) + 1e-9);

        % 未來 horizon 秒內軌跡進入碰撞區，就算一次 collision sample
        if dcpa <= collision_radius && approaching_speed > 0
            collision_count = collision_count + 1;
        end
    end

    prob = collision_count / num_mc;
end