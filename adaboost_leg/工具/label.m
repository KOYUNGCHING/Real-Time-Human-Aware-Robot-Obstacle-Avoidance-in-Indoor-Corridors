function label_segments_by_polygon()
% 用滑鼠畫 polygon 圈出「腳」區域
% 再自動轉成 segment labels
% 輸出 leg_labels.mat

    close all; clc;

    %% 載入資料
    load('xy_data.mat');   % 需要 xy_data
    [num_pts, dim, num_frames] = size(xy_data);
    assert(dim == 2, 'xy_data 應為 N x 2 x T');

    fprintf('載入完成：%d 幀，每幀最多 %d 個點\n', num_frames, num_pts);

    %% 可調參數
    frame_step = 20;           % 每隔幾幀標一次
    overlap_ratio_th = 0.5;    % 若 segment 中超過 50% 的點被圈到，就算 leg

    frames_to_label = 1:frame_step:num_frames;
    leg_labels = cell(num_frames,1);

    %% 逐幀標記
    for ii = 1:length(frames_to_label)
        t = frames_to_label(ii);

        XY_now = xy_data(:,:,t);

        % 去掉 NaN 點
        valid = isfinite(XY_now(:,1)) & isfinite(XY_now(:,2));
        XY_valid = XY_now(valid,:);

        if isempty(XY_valid)
            fprintf('Frame %d 沒有有效點，跳過\n', t);
            continue;
        end

        % segmentation
        [Seg, Si_n, S_n] = Segment(XY_valid);

        %% 畫圖
        fig = figure(1); clf;
        ax = axes('Parent', fig); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
        xlabel(ax,'X (m)');
        ylabel(ax,'Y (m)');
        title(ax, sprintf('Frame %d / %d：圈選腳印', t, num_frames));

        plot(ax, XY_valid(:,1), XY_valid(:,2), 'b.', 'MarkerSize', 8);
        plot(ax, 0, 0, 'ko', 'MarkerSize', 6, 'LineWidth', 1.2);

        % 可選：顯示 segment 編號
        cmap = lines(max(S_n,7));
        for j = 1:S_n
            idx = Seg(1:Si_n(j), j);
            idx = idx(idx ~= 0);
            pts = XY_valid(idx,:);

            if isempty(pts), continue; end

            color_j = cmap(mod(j-1,size(cmap,1))+1,:);
            plot(ax, pts(:,1), pts(:,2), '.', 'Color', color_j, 'MarkerSize', 10);

            cx = mean(pts(:,1));
            cy = mean(pts(:,2));
            text(cx, cy, sprintf('%d', j), 'Color', 'r', ...
                'FontSize', 9, 'FontWeight', 'bold');
        end

        %% 用滑鼠畫 polygon
        disp('左鍵逐點圈出腳部區域，右鍵或 Enter 結束，Esc 取消。');

        xp=[]; yp=[];
        hpoly = plot(ax,nan,nan,'g-','LineWidth',1.5);
        hdots = plot(ax,nan,nan,'go','MarkerFaceColor','g','MarkerSize',4);

        while true
            [xi, yi, btn] = ginput(1);

            if isempty(btn) || btn==3 || btn==13
                break;
            elseif btn==27
                error('使用者取消標記。');
            else
                xp(end+1,1)=xi; 
                yp(end+1,1)=yi; 

                if numel(xp) >= 2
                    set(hpoly,'XData',xp,'YData',yp);
                end
                set(hdots,'XData',xp,'YData',yp);
                drawnow;
            end
        end

        if numel(xp) < 3
            fprintf('Frame %d: polygon 點數不足，視為沒標到腳\n', t);
            leg_labels{t} = [];
            continue;
        end

        % 封閉 polygon
        xp(end+1)=xp(1);
        yp(end+1)=yp(1);
        set(hpoly,'XData',xp,'YData',yp);
        drawnow;

        %% 判斷哪些點落在 polygon 內
        in = inpolygon(XY_valid(:,1), XY_valid(:,2), xp, yp);

        %% 把點標籤轉成 segment 標籤
        leg_seg_ids = [];

        for j = 1:S_n
            idx = Seg(1:Si_n(j), j);
            idx = idx(idx ~= 0);

            if isempty(idx)
                continue;
            end

            ratio_in = mean(in(idx));   % 這個 segment 有多少比例點在 polygon 內

            if ratio_in >= overlap_ratio_th
                leg_seg_ids(end+1) = j; 
            end
        end

        leg_labels{t} = unique(leg_seg_ids);

        %% 畫結果
        clf; ax = axes('Parent', fig); hold(ax,'on'); grid(ax,'on'); axis(ax,'equal');
        xlabel(ax,'X (m)');
        ylabel(ax,'Y (m)');
        title(ax, sprintf('Frame %d：紅色 = 被標成 leg 的 segments', t));

        for j = 1:S_n
            idx = Seg(1:Si_n(j), j);
            idx = idx(idx ~= 0);
            pts = XY_valid(idx,:);

            if isempty(pts), continue; end

            if ismember(j, leg_labels{t})
                plot(ax, pts(:,1), pts(:,2), 'r.', 'MarkerSize', 12);
            else
                plot(ax, pts(:,1), pts(:,2), 'b.', 'MarkerSize', 10);
            end

            cx = mean(pts(:,1));
            cy = mean(pts(:,2));
            text(cx, cy, sprintf('%d', j), 'Color', 'k', ...
                'FontSize', 9, 'FontWeight', 'bold');
        end

        plot(ax, xp, yp, 'g-', 'LineWidth', 1.2);
        plot(ax, 0, 0, 'ko', 'MarkerSize', 6, 'LineWidth', 1.2);

        fprintf('Frame %d: 標成 leg 的 segments = ', t);
        disp(leg_labels{t});

        disp('按任意鍵進下一幀...');
        pause;
    end

    %% 存檔
    save('leg_labels.mat', 'leg_labels');
    fprintf('已存成 leg_labels.mat\n');
end