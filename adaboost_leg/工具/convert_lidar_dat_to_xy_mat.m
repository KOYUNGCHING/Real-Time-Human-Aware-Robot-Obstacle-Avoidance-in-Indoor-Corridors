function convert_lidar_dat_to_xy_mat()
% 將 lidar_data.dat 轉成 xy_data.mat
% 假設每列是一個完整 scan:
%   ros_time ranges_data intensities_data

    close all; clc;

    %% 檔名設定
    DATA_FN = 'lidar_data.dat';
    SAVE_FN = 'xy_data.mat';

    assert(isfile(DATA_FN), '找不到檔案: %s', DATA_FN);

    %% 讀檔
    fid = fopen(DATA_FN, 'r');
    assert(fid > 0, '無法開啟檔案 %s', DATA_FN);

    lines = {};
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line)
            lines{end+1} = line; 
        end
    end
    fclose(fid);

    % 去掉空行
    lines = lines(~cellfun(@isempty, lines));

    % 去掉 header
    if startsWith(lines{1}, '#')
        lines = lines(2:end);
    end

    num_frames = length(lines);
    fprintf('總共有 %d 個 scan frames\n', num_frames);

    %% 先看第一列，決定每幀幾個點
    parts = strsplit(lines{1}, sprintf('\t'));
    assert(numel(parts) >= 2, '每列至少要有 ros_time 與 ranges_data');

    range_str = strtrim(parts{2});
    range_vals = sscanf(range_str, '%f')';
    num_pts = length(range_vals);

    fprintf('每幀點數 = %d\n', num_pts);

    %% 設定角度
    % 這裡你可能要改 
    % 若 LiDAR 是 360 度掃描，常可設：
    angle_min = -pi;
    angle_max = pi;

    % 如果你知道不是 360 度，要改這裡
    angles = linspace(angle_min, angle_max, num_pts);

    %% 配置輸出
    xy_data = nan(num_pts, 2, num_frames);
    ros_time = nan(num_frames, 1);

    %% 逐幀轉換
    for t = 1:num_frames
        parts = strsplit(lines{t}, sprintf('\t'));

        if numel(parts) < 2
            warning('第 %d 列格式不對，跳過', t);
            continue;
        end

        % 時間
        ros_time(t) = str2double(strtrim(parts{1}));

        % ranges
        range_str = strtrim(parts{2});
        r = sscanf(range_str, '%f')';

        if length(r) ~= num_pts
            warning('第 %d 幀點數不一致，預期 %d，實際 %d', t, num_pts, length(r));
            continue;
        end

        % 無效值處理：0 視為沒量到
        r(r <= 0) = NaN;

        x = r .* cos(angles);
        y = r .* sin(angles);

        xy_data(:,1,t) = x(:);
        xy_data(:,2,t) = y(:);
    end

    %% 存檔
    save(SAVE_FN, 'xy_data', 'ros_time', 'angles');

    fprintf('已存成 %s\n', SAVE_FN);

end