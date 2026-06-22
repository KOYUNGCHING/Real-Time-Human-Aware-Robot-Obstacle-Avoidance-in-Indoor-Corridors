function run_full_nfold()
clc; clear; close all;

%% 載入 train / test 資料
S = load('train_data_60s_2p.mat');
xy_data_train = S.xy_data;

S = load('train_data_60s_2p＿label.mat');
leg_labels_train = S.leg_labels;

S = load('test_data_60s_2p.mat');
xy_data_test = S.xy_data;

S = load('test_data_60s_2p＿label.mat');
leg_labels_test = S.leg_labels;

%% 合併成完整資料集
xy_data_all = cat(3, xy_data_train, xy_data_test);
leg_labels_all = [leg_labels_train; leg_labels_test];

fprintf('Merged total frames: %d\n', size(xy_data_all, 3));

%% 參數設定
cfg.reviewed_frames_all = 1:10:size(xy_data_all, 3);   % 所有有標記的幀
cfg.T = 50;               % AdaBoost weak classifiers 數量
cfg.neg_ratio = 5;        % 負樣本保留比例
cfg.K = 5;                % K-fold
rng(0);

feature_names = { ...
    'point_count', ...
    'std_dev_to_centroid', ...
    'segment_width', ...
    'circle_fit_radius', ...
    'boundary_std_dev', ...
    'mean_curvature', ...
    'mean_angular_difference', ...
    'min_line_fitting_error', ...
    'max_line_fitting_error', ...
    'ransac_inlier_ratio' ...
};

%% 建立完整資料集
[X_all, Y_all, frame_ids_all, seg_ids_all] = ...
    build_leg_dataset(xy_data_all, leg_labels_all, cfg.reviewed_frames_all);

fprintf('Total feature matrix size: %d x %d\n', size(X_all,1), size(X_all,2));
fprintf('Total positive samples: %d\n', sum(Y_all==1));
fprintf('Total negative samples: %d\n', sum(Y_all==0));

if size(X_all,2) ~= numel(feature_names)
    error('Feature dimension mismatch: expected %d, got %d', ...
        numel(feature_names), size(X_all,2));
end

%% 取得所有 frame，做 K-fold
frames = unique(frame_ids_all);
num_frames = numel(frames);

if cfg.K > num_frames
    error('K=%d 大於可用 frame 數量=%d', cfg.K, num_frames);
end

permF = frames(randperm(num_frames));   % 打亂 frame 順序
fold_id_of_frame = mod(0:num_frames-1, cfg.K) + 1;
fold_id_of_frame = fold_id_of_frame(:);
frames_shuffled = permF(:);

%% 儲存每一 fold 結果
all_results = struct();

best_f1 = -inf;
best_fold = -1;
best_model = [];
best_thr = 0;
best_metrics = [];

%% K-fold cross validation
for k = 1:cfg.K
    fprintf('\n========================================\n');
    fprintf('Fold %d / %d\n', k, cfg.K);
    fprintf('========================================\n');

    % test frames = 第 k fold
    test_f = frames_shuffled(fold_id_of_frame == k);
    
    % train_pool frames = 其他 folds
    train_pool_f = frames_shuffled(fold_id_of_frame ~= k);
    
    % 在 train_pool 裡再切 train / validation
    perm_pool = train_pool_f(randperm(numel(train_pool_f)));
    
    nTrainInner = round(0.8 * numel(perm_pool));
    
    train_f = perm_pool(1:nTrainInner);
    val_f   = perm_pool(nTrainInner+1:end);
    
    % segment level mask
    tr = ismember(frame_ids_all, train_f);
    va = ismember(frame_ids_all, val_f);
    te = ismember(frame_ids_all, test_f);
    
    Xtr = X_all(tr,:);
    ytr = Y_all(tr);
    
    Xva = X_all(va,:);
    yva = Y_all(va);
    
    Xte = X_all(te,:);
    yte = Y_all(te);
    
    fprintf('Test  samples: %d (pos=%d, neg=%d)\n', ...
        numel(yte), sum(yte==1), sum(yte==0));
    if isempty(Xtr) || isempty(Xva)
        error('Fold %d 出現空的 train/val 集合', k);
    end

    fprintf('Train samples: %d (pos=%d, neg=%d)\n', ...
        numel(ytr), sum(ytr==1), sum(ytr==0));
    fprintf('Val   samples: %d (pos=%d, neg=%d)\n', ...
        numel(yva), sum(yva==1), sum(yva==0));

    % 平衡資料
    pos = find(ytr==1);
    neg = find(ytr==0);

    if isempty(pos)
        error('Fold %d 訓練集中沒有正樣本', k);
    end

    neg_keep = min(cfg.neg_ratio * numel(pos), numel(neg));
    neg = neg(randperm(numel(neg), neg_keep));

    idx = [pos; neg];
    idx = idx(randperm(numel(idx)));

    Xtr_bal = Xtr(idx,:);
    ytr_bal = ytr(idx);

    fprintf('Balanced train samples: %d (pos=%d, neg=%d)\n', ...
        numel(ytr_bal), sum(ytr_bal==1), sum(ytr_bal==0));

    %% 訓練 AdaBoost
    [~,~,~,~,~,~,model] = Adaboost(Xtr_bal, ytr_bal, Xva, yva, cfg.T);

    %% Training 評估（balanced training set）
    scores_tr = adaboost_score(model, Xtr_bal);
    yhat_tr = scores_tr > 0;
    metrics_tr = evaluate_binary(ytr_bal, yhat_tr);

    fprintf('\n[Fold %d] Training Result\n', k);
    fprintf('Accuracy : %.4f\n', metrics_tr.acc);
    fprintf('Precision: %.4f\n', metrics_tr.precision);
    fprintf('Recall   : %.4f\n', metrics_tr.recall);
    fprintf('F1       : %.4f\n', metrics_tr.f1);
    disp('Confusion Matrix [TN FP; FN TP]:');
    disp(metrics_tr.CM);

    %% Validation 找最佳 threshold
    scores_va = adaboost_score(model, Xva);
    [thr, info_val] = choose_best_threshold(scores_va, yva);

    fprintf('\n[Fold %d] Validation Result\n', k);
    fprintf('Best threshold: %.4f\n', thr);
    fprintf('Accuracy : %.4f\n', info_val.acc);
    fprintf('Precision: %.4f\n', info_val.precision);
    fprintf('Recall   : %.4f\n', info_val.recall);
    fprintf('F1       : %.4f\n', info_val.f1);
    disp('Confusion Matrix [TN FP; FN TP]:');
    disp(info_val.CM);

    scores_te = adaboost_score(model, Xte);
    yhat_te = scores_te > thr;
    metrics_test = evaluate_binary(yte, yhat_te);
    
    fprintf('\n[Fold %d] Test Result\n', k);
    fprintf('Threshold from validation: %.4f\n', thr);
    fprintf('Accuracy : %.4f\n', metrics_test.acc);
    fprintf('Precision: %.4f\n', metrics_test.precision);
    fprintf('Recall   : %.4f\n', metrics_test.recall);
    fprintf('F1       : %.4f\n', metrics_test.f1);
    disp('Confusion Matrix [TN FP; FN TP]:');
    disp(metrics_test.CM);
    %% Feature importance
    D = size(Xtr_bal, 2);
    feat_importance = zeros(D,1);

    for t = 1:numel(model.alpha)
        j = model.stump(t).j;
        feat_importance(j) = feat_importance(j) + model.alpha(t);
    end

    feat_importance_norm = feat_importance / max(sum(feat_importance), eps);

    %% 存結果
    all_results(k).fold = k;

    all_results(k).train_frames = train_f;
    all_results(k).val_frames = val_f;
    all_results(k).test_frames = test_f;
    
    all_results(k).model = model;
    all_results(k).thr = thr;
    
    all_results(k).metrics_train = metrics_tr;
    all_results(k).metrics_val = info_val;
    all_results(k).metrics_test = metrics_test;
    
    all_results(k).feat_importance = feat_importance;
    all_results(k).feat_importance_norm = feat_importance_norm;
    %% 更新最佳 fold（用 validation F1）
    if metrics_test.f1 > best_f1
        best_f1 = metrics_test.f1;
        best_fold = k;
        best_model = model;
        best_thr = thr;
        best_metrics = metrics_test;
    end
end

%% 印出所有 fold 總表
fprintf('\n========================================\n');
fprintf('K-FOLD SUMMARY\n');
fprintf('========================================\n');

for k = 1:cfg.K
    m = all_results(k).metrics_test;
    fprintf('Fold %d Test -> Acc=%.4f, Prec=%.4f, Recall=%.4f, F1=%.4f, Thr=%.4f\n', ...
        k, m.acc, m.precision, m.recall, m.f1, all_results(k).thr);
end

fprintf('\n========================================\n');
fprintf('BEST FOLD = %d\n', best_fold);
fprintf('Best Test F1 = %.4f\n', best_f1);
fprintf('Accuracy : %.4f\n', best_metrics.acc);
fprintf('Precision: %.4f\n', best_metrics.precision);
fprintf('Recall   : %.4f\n', best_metrics.recall);
fprintf('F1       : %.4f\n', best_metrics.f1);
disp('Best Fold Confusion Matrix [TN FP; FN TP]:');
disp(best_metrics.CM);

%% 輸出 Python 格式參數（best fold）
print_python_parameters(best_model, best_thr, best_fold);

%% 存最佳模型
save('leg_model_best_fold.mat', 'best_model', 'best_thr', 'best_fold', ...
    'best_metrics', 'all_results', 'feature_names');

%% 畫最佳 fold 的 feature importance
best_imp = all_results(best_fold).feat_importance_norm;

figure;
bar(best_imp);
xticks(1:numel(feature_names));
xticklabels(feature_names);
xtickangle(45);
ylabel('Normalized Importance');
title(sprintf('Feature Importance (Best Fold = %d)', best_fold));
grid on;

%% (optional) 用最佳模型看全部資料的預測結果
visualize_leg_prediction(xy_data_all, best_model, best_thr);

end

%% functions
function [X, Y, frame_ids, seg_ids] = build_leg_dataset(xy_data, leg_labels, reviewed_frames)
    X = [];
    Y = [];
    frame_ids = [];
    seg_ids = [];

    num_frames = size(xy_data, 3);
    reviewed_frames = reviewed_frames(reviewed_frames >= 1 & reviewed_frames <= num_frames);

    for t = reviewed_frames
        XY_now = xy_data(:,:,t);

        % 排除無效點：NaN / Inf / (0,0)
        valid = isfinite(XY_now(:,1)) & isfinite(XY_now(:,2)) & ...
                ~(XY_now(:,1)==0 & XY_now(:,2)==0);

        XY_valid = XY_now(valid, :);

        if isempty(XY_valid)
            continue;
        end

        [Seg, Si_n, S_n] = Segment(XY_valid);

        % 對每個 segment 算 10 維特徵
        Xseg = extract_segment_features(XY_valid, Seg, Si_n, S_n);

        % 預設所有 segment 都不是 leg
        Yseg = zeros(S_n, 1);

        % 如果這幀有標註 leg segment，就把對應 segment 設成 1
        if t <= numel(leg_labels) && ~isempty(leg_labels{t})
            ids = leg_labels{t};
            ids = ids(ids >= 1 & ids <= S_n);   % 防呆
            Yseg(ids) = 1;
        end

        % 只保留有成功算出特徵的 segment
        valid_seg = any(Xseg ~= 0, 2);

        X = [X; Xseg(valid_seg,:)];
        Y = [Y; Yseg(valid_seg)];
        kept_seg_ids = find(valid_seg);

        frame_ids = [frame_ids; repmat(t, numel(kept_seg_ids), 1)];
        seg_ids   = [seg_ids; kept_seg_ids(:)];
    end
end

function [acc_tr4, acc_te4, CM_tr4, CM_te4, yhat_tr4, yhat_te4, model] = ...
    Adaboost(Xtr4, ytr4, Xte4, yte4, T)

    % 將 0/1 標籤轉換為 ±1
    ytr4_pm = 2*ytr4 - 1;
    yte4_pm = 2*yte4 - 1;

    [m4, d4] = size(Xtr4);

    % 初始化樣本權重
    D = ones(m4,1) / m4;

    alpha = zeros(T,1);
    stump = struct('j',[],'theta',[],'s',[]);

    fprintf('Starting Adaboost Training for T=%d cycles...\n', T);

    for t = 1:T
        best_err = inf;
        best = struct('j',1,'theta',0,'s',1);

        % 1. 找最佳決策樹樁
        for j = 1:d4
            xj = Xtr4(:,j);
            vals = unique(xj);

            if numel(vals) <= 1
                thetas = vals + 1e-6;
            else
                thetas = (vals(1:end-1) + vals(2:end))/2;
            end

            for th = thetas.'
                for sgn = [+1, -1]
                    pred = stump_predict_column(xj, th, sgn);
                    err = sum(D .* (pred ~= ytr4_pm));

                    if err < best_err
                        best_err = err;
                        best.j = j;
                        best.theta = th;
                        best.s = sgn;
                    end
                end
            end
        end

        % 2. 用最佳 stump 預測
        h_t = stump_predict_column(Xtr4(:,best.j), best.theta, best.s);

        % 3. 算 alpha
        eps_t = max(min(best_err, 1-1e-9), 1e-9);
        alpha(t) = 0.5 * log((1 - eps_t) / eps_t);

        % 4. 更新樣本分布
        D = D .* exp(-alpha(t) .* (ytr4_pm .* h_t));
        D = D / sum(D);

        stump(t) = best;

        fprintf('Cycle %3d: Feature %d, Error %.4f, Alpha %.4f\n', ...
            t, best.j, best_err, alpha(t));
    end

    % 5. 強分類器輸出
    F_tr = zeros(m4,1);
    F_te = zeros(size(Xte4,1),1);

    for t = 1:T
        stump_t = stump(t);
        F_tr = F_tr + alpha(t) * stump_predict_column(Xtr4(:,stump_t.j), stump_t.theta, stump_t.s);
        F_te = F_te + alpha(t) * stump_predict_column(Xte4(:,stump_t.j), stump_t.theta, stump_t.s);
    end

    % 6. 轉回 0/1
    yhat_tr4_pm = sign(F_tr);
    yhat_tr4_pm(yhat_tr4_pm==0) = 1;

    yhat_te4_pm = sign(F_te);
    yhat_te4_pm(yhat_te4_pm==0) = 1;

    yhat_tr4 = (yhat_tr4_pm + 1)/2;
    yhat_te4 = (yhat_te4_pm + 1)/2;

    % 7. 準確率
    acc_tr4 = mean(yhat_tr4 == ytr4);
    acc_te4 = mean(yhat_te4 == yte4);

    fprintf('\n[AdaBoost] Training accuracy = %.2f%%\n', 100*acc_tr4);
    fprintf('[AdaBoost] Testing  accuracy = %.2f%%\n', 100*acc_te4);

    % 8. 訓練集 confusion matrix
    TN_tr4 = sum((ytr4==0) & (yhat_tr4==0));
    FP_tr4 = sum((ytr4==0) & (yhat_tr4==1));
    FN_tr4 = sum((ytr4==1) & (yhat_tr4==0));
    TP_tr4 = sum((ytr4==1) & (yhat_tr4==1));
    CM_tr4 = [TN_tr4 FP_tr4; FN_tr4 TP_tr4];

    % 9. 測試集 confusion matrix
    TN_te4 = sum((yte4==0) & (yhat_te4==0));
    FP_te4 = sum((yte4==0) & (yhat_te4==1));
    FN_te4 = sum((yte4==1) & (yhat_te4==0));
    TP_te4 = sum((yte4==1) & (yhat_te4==1));
    CM_te4 = [TN_te4 FP_te4; FN_te4 TP_te4];

    % 10. 輸出模型
    model.alpha = alpha;
    model.stump = stump;
end

function pred = stump_predict_column(xcol, theta, s)
    pred = ones(size(xcol));
    pred(s*(xcol - theta) < 0) = -1;
end

function scores = adaboost_score(model, X)
    n = size(X,1);
    scores = zeros(n,1);

    T = numel(model.alpha);
    for t = 1:T
        stump_t = model.stump(t);
        h = stump_predict_column(X(:,stump_t.j), stump_t.theta, stump_t.s);
        scores = scores + model.alpha(t) * h;
    end
end

function [best_thr, best_info] = choose_best_threshold(scores, ytrue)
    thr_list = linspace(min(scores), max(scores), 200);

    best_f1 = -inf;
    best_thr = 0;
    best_info = struct('precision',0,'recall',0,'f1',0,'acc',0,'CM',[]);

    for th = thr_list
        yhat = double(scores > th);
        m = evaluate_binary(ytrue, yhat);

        if m.f1 > best_f1
            best_f1 = m.f1;
            best_thr = th;
            best_info = m;
        end
    end
end

function metrics = evaluate_binary(ytrue, yhat)
    TN = sum((ytrue==0) & (yhat==0));
    FP = sum((ytrue==0) & (yhat==1));
    FN = sum((ytrue==1) & (yhat==0));
    TP = sum((ytrue==1) & (yhat==1));

    acc = mean(ytrue == yhat);
    precision = TP / max(TP + FP, 1);
    recall    = TP / max(TP + FN, 1);
    f1        = 2 * precision * recall / max(precision + recall, 1e-12);

    metrics = struct();
    metrics.acc = acc;
    metrics.precision = precision;
    metrics.recall = recall;
    metrics.f1 = f1;
    metrics.CM = [TN FP; FN TP];
end

function print_python_parameters(model, thr, best_fold)
    fprintf('\n--- Best Fold Python Parameters ---\n\n');
    fprintf('BEST_FOLD = %d\n\n', best_fold);

    % ALPHA
    fprintf('ALPHA_LEG = np.array([');
    if numel(model.alpha) > 1
        fprintf('%.6f, ', model.alpha(1:end-1));
    end
    fprintf('%.6f])\n\n', model.alpha(end));

    % STUMPS
    fprintf('STUMPS_LEG = [\n');
    for t = 1:numel(model.stump)
        fprintf('    [%d, %.6f, %d]', ...
            model.stump(t).j - 1, ...
            model.stump(t).theta, ...
            model.stump(t).s);

        if t < numel(model.stump)
            fprintf(',\n');
        else
            fprintf('\n');
        end
    end
    fprintf(']\n');

    fprintf('THR_LEG = %.6f\n', thr);
end

function visualize_leg_prediction(xy_data, model, thr)
    for sec = 1:size(xy_data,3)
        clf;
        XY_now = xy_data(:,:,sec);

        % 只保留有效點
        valid = isfinite(XY_now(:,1)) & isfinite(XY_now(:,2)) & ...
                ~(XY_now(:,1)==0 & XY_now(:,2)==0);

        XY_valid = XY_now(valid,:);

        if isempty(XY_valid)
            continue;
        end

        % 分段
        [Seg, Si_n, S_n] = Segment(XY_valid);

        % 算特徵
        Xseg = extract_segment_features(XY_valid, Seg, Si_n, S_n);
        valid_seg = any(Xseg ~= 0, 2);

        % 模型分數
        scores = nan(S_n,1);
        scores(valid_seg) = adaboost_score(model, Xseg(valid_seg,:));

        % 預測
        pred_leg = scores > thr;
        pred_leg(~valid_seg) = false;

        % 畫圖
        hold on;

        h_all = plot(nan, nan, '.', 'Color', [0.6 0.6 0.6]);
        h_pr  = plot(nan, nan, 'ro', 'LineWidth', 1.5, 'MarkerSize', 8);

        plot(XY_valid(:,1), XY_valid(:,2), '.', 'Color', [0.6 0.6 0.6]);

        for i = 1:S_n
            if pred_leg(i)
                idx = Seg(1:Si_n(i), i);
                idx = idx(idx > 0);

                if isempty(idx)
                    continue;
                end

                pts = XY_valid(idx,:);
                plot(pts(:,1), pts(:,2), 'ro', ...
                    'LineWidth', 1.5, 'MarkerSize', 8);
            end
        end

        title(sprintf('Frame %d', sec));
        axis equal;
        grid on;

        legend([h_all, h_pr], ...
            {'all points', 'predicted leg'}, ...
            'Location', 'northeast');

        hold off;
        pause(0.1);
    end
end

function [Seg,Si_n,S_n] = Segment(xy)
    x = xy(:,1);
    y = xy(:,2);

    threshold = 0.1;

    S_i = 1;
    S_n = 1;

    n0ind = find(x~=0 | y~=0);

    if isempty(n0ind)
        Seg = [];
        Si_n = [];
        S_n = 0;
        return;
    end

    n_0 = numel(n0ind);

    Seg = zeros(n_0, n_0);   % 先大一點，避免動態長大
    Seg(1,1) = n0ind(1);

    for i = 2:n_0
        if sqrt((x(n0ind(i)) - x(n0ind(i-1)))^2 + (y(n0ind(i)) - y(n0ind(i-1)))^2) < threshold
            S_i = S_i + 1;
            Seg(S_i,S_n) = n0ind(i);
        else
            S_n = S_n + 1;
            S_i = 1;
            Seg(S_i,S_n) = n0ind(i);
        end
    end

    Seg = Seg(:,1:S_n);

    Si_n = zeros(S_n,1);
    for j = 1:S_n
        k = size(find(Seg(:,j)~=0));
        Si_n(j) = k(1);
    end
end

function Xseg = extract_segment_features(XY_now, Seg, Si_n, S_n)
    Xseg = zeros(S_n, 10);
    EPS = 1e-12;

    for j = 1:S_n
        idx = Seg(1:Si_n(j), j);
        idx = idx(idx > 0);
        pts = XY_now(idx, :);
        k = size(pts, 1);

        if k < 2
            continue;
        end

        x = pts(:,1);
        y = pts(:,2);

        %% [1] Point Count
        point_count = double(k);

        %% [2] Std. Dev. to Centroid
        mu = mean(pts, 1);
        diff_mu = pts - mu;
        dist2_mu = sum(diff_mu.^2, 2);

        if k > 1
            std_dev_to_centroid = sqrt(sum(dist2_mu) / (k - 1));
        else
            std_dev_to_centroid = 0;
        end

        %% [3] Segment Width
        segment_width = norm(pts(end,:) - pts(1,:));

        %% [4] Circle Fit Radius
        circle_fit_radius = 0;
        if k >= 3
            A = [-2*x, -2*y, ones(k,1)];
            b = -(x.^2 + y.^2);

            theta = pinv(A) * b;
            xc = theta(1);
            yc = theta(2);
            c3 = theta(3);

            rc_sq = xc^2 + yc^2 - c3;
            if isfinite(rc_sq) && rc_sq > 0
                circle_fit_radius = sqrt(rc_sq);
            end
        end

        %% [5] Boundary Std. Dev.
        if k >= 2
            step_vec = diff(pts, 1, 1);
            step_dist = sqrt(sum(step_vec.^2, 2));
        else
            step_dist = [];
        end

        if numel(step_dist) >= 2
            boundary_std_dev = std(step_dist);
        else
            boundary_std_dev = 0;
        end

        %% [6] Mean Curvature
        if k >= 3
            curvatures = zeros(k-2,1);

            for t = 2:k-1
                A_pt = pts(t-1,:);
                B_pt = pts(t,:);
                C_pt = pts(t+1,:);

                dAB = norm(B_pt - A_pt);
                dBC = norm(C_pt - B_pt);
                dAC = norm(C_pt - A_pt);

                area2 = abs( ...
                    (B_pt(1)-A_pt(1))*(C_pt(2)-A_pt(2)) - ...
                    (B_pt(2)-A_pt(2))*(C_pt(1)-A_pt(1)) );
                area_tri = 0.5 * area2;

                denom = dAB * dBC * dAC;
                if denom > EPS
                    curvatures(t-1) = 4 * area_tri / denom;
                else
                    curvatures(t-1) = 0;
                end
            end

            mean_curvature = mean(curvatures);
        else
            mean_curvature = 0;
        end

        %% [7] Mean Angular Difference
        if k >= 3
            betas = zeros(k-2,1);

            for t = 2:k-1
                v1 = pts(t-1,:) - pts(t,:);
                v2 = pts(t+1,:) - pts(t,:);

                n1 = norm(v1);
                n2 = norm(v2);

                if n1 > EPS && n2 > EPS
                    cos_beta = dot(v1, v2) / (n1 * n2);
                    cos_beta = max(-1, min(1, cos_beta));
                    betas(t-1) = acos(cos_beta);
                else
                    betas(t-1) = 0;
                end
            end

            mean_angular_difference = mean(betas);
        else
            mean_angular_difference = 0;
        end

        %% [8], [9] Min / Max Line Fitting Error
        min_line_fitting_error = 0;
        max_line_fitting_error = 0;

        if k >= 2
            centered = pts - mean(pts,1);
            [~,~,V] = svd(centered, 'econ');

            dir_vec = V(:,1);
            normal_vec = [-dir_vec(2); dir_vec(1)];
            normal_vec = normal_vec / (norm(normal_vec) + EPS);

            r_line = mean(pts * normal_vec);

            line_err = abs(pts * normal_vec - r_line);

            min_line_fitting_error = min(line_err);
            max_line_fitting_error = max(line_err);
        end

        %% [10] RANSAC Inlier Ratio
        ransac_inlier_ratio = 0;

        if k >= 2
            best_inlier = 0;
            dist_thr = 0.02;
            max_iter = min(30, nchoosek_safe(k,2));

            if max_iter > 0
                for it = 1:max_iter
                    pair = randperm(k, 2);
                    p1 = pts(pair(1), :);
                    p2 = pts(pair(2), :);

                    v = p2 - p1;
                    nv = norm(v);

                    if nv < EPS
                        continue;
                    end

                    d = abs((pts(:,1)-p1(1))*v(2) - (pts(:,2)-p1(2))*v(1)) / nv;

                    inlier_count = sum(d < dist_thr);

                    if inlier_count > best_inlier
                        best_inlier = inlier_count;
                    end
                end

                ransac_inlier_ratio = best_inlier / k;
            end
        end

        %% 存特徵
        Xseg(j,:) = [ ...
            point_count, ...
            std_dev_to_centroid, ...
            segment_width, ...
            circle_fit_radius, ...
            boundary_std_dev, ...
            mean_curvature, ...
            mean_angular_difference, ...
            min_line_fitting_error, ...
            max_line_fitting_error, ...
            ransac_inlier_ratio ...
        ];
    end
end

function val = nchoosek_safe(n,r)
    if n < r
        val = 0;
    else
        val = nchoosek(n,r);
    end
end