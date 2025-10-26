num_nodes = 10;
metric = "test_acc";  % 只画test_acc

% 第一行：主实验CSV文件路径
csv_files = {"math_mix.csv","math_cls.csv","math_sep_cls.csv"};
% 第二行：LoRA对照实验CSV文件路径  
csv_files_lora = {"math_mix_lora.csv","math_cls_lora.csv","math_sep_cls_lora.csv"};

num_sets = numel(csv_files);

% 10色（Spectral）稍暗一点
colors = [...
  213, 109,  79;
  244, 109,  67;
  253, 174,  97;
  254, 224, 139;
  255, 255, 191;
  230, 245, 152;
  171, 221, 164;
  102, 194, 165;
   50, 136, 189;
   94,  79, 162] / 255 * 0.9;

figure('Position',[100,100,1600,900]); 
tl = tiledlayout(2, num_sets, ...
    'TileSpacing','compact', ...
    'Padding','loose');   % 用 loose 给上下左右多留白
% 记录每列第二行(= test_acc)轴句柄，稍后用来放 (a)(b)(c)
outerpos = tl.OuterPosition;   % [x y w h]
outerpos(2) = 0.05;            % 把底部抬高一点，留 15% 空间
outerpos(4) = 0.95;             % 高度相应缩小
tl.OuterPosition = outerpos;
axs = gobjects(2, num_sets);

% 两行：row=1画主实验，row=2画LoRA对照
for row = 1:2
    % 根据行号选择CSV文件
    if row == 1
        csv_file_list = csv_files;
        row_title = 'Main Experiment';
    else
        csv_file_list = csv_files_lora;
        row_title = 'LoRA Baseline';
    end
    
    for f = 1:num_sets
        csv_file = csv_file_list{f};
        
        % 从CSV读入所有节点数据，保留原始列名
        T_all = readtable(csv_file, 'VariableNamingRule', 'preserve');
        
        % 将数据按node_id分组
        data = cell(1, num_nodes);
        for i = 0:num_nodes-1
            % 提取该节点的数据
            node_data = T_all(T_all.node_id == i, :);
            if ~isempty(node_data)
                data{i+1} = node_data;
            end
        end
        
        % 画图
        ax = nexttile((row-1)*num_sets + f); hold(ax,'on');
        axs(row,f) = ax;  % 记录轴
        max_len = 0;
        
        % 先遍历一次找出最大iter
        for node = 1:num_nodes
            T = data{node};
            if ~isempty(T) && ismember('iter', T.Properties.VariableNames)
                max_len = max(max_len, max(T.iter)+1);
            end
        end
        
        % 初始化为NaN矩阵，确保缺失值不参与平均计算
        all_vals = NaN(num_nodes, max_len);

        for node = 1:num_nodes
            T = data{node};
            if isempty(T), continue; end

            % 数据已经是表格格式，直接使用
            if ismember(metric, T.Properties.VariableNames) && ismember('iter', T.Properties.VariableNames)
                vals = T.(metric);
                iters = T.iter;  % 使用实际的iter值
                mask = ~isnan(vals);

                if any(mask)
                    plot(ax, iters(mask), vals(mask), '-', ...
                         'Color', [colors(node,:) 0.5], ...
                         'LineWidth', 1);
                    % 按iter索引存储（iter从0开始）
                    for i = 1:length(iters)
                        if ~isnan(vals(i))
                            all_vals(node, iters(i)+1) = vals(i);
                        end
                    end
                end
            end
        end

        % === 平均线（橘红色）===
        % 使用omitnan确保只对有数据的节点求平均
        if max_len > 0
            mean_vals = mean(all_vals, 1, 'omitnan');
            valid_iters = ~isnan(mean_vals);  % 至少有一个节点有数据的iter
            if any(valid_iters)
                plot(ax, find(valid_iters)-1, mean_vals(valid_iters), '-', 'LineWidth', 2, ...
                     'Color', [158,1,66]/255, ...
                     'DisplayName', sprintf('%s mean', metric));
            end
        end

        % % === test_acc 标最佳值（画点 + 数值）===
        % if ~isempty(all_vals)
        %     [max_val, max_idx] = max(all_vals(:), [], 'omitnan');
        %     [~, iter_idx] = ind2sub(size(all_vals), max_idx);
        % 
        %     % 五角星点
        %     scatter(ax, iter_idx, max_val, 60, 'kp', 'filled', 'LineWidth',2);
        % 
        %     % 数值文字
        %     text(ax, iter_idx, max_val, sprintf('best = %.3f', max_val), ...
        %         'VerticalAlignment','bottom', ...
        %         'HorizontalAlignment','right', ...
        %         'FontSize', 17, ...
        %         'FontName','Times New Roman', ...
        %         'FontWeight','bold', ...
        %         'Color','k');
        % end

        % === 轴样式 ===
        set(ax, 'FontName', 'Times New Roman', ...
        'FontSize', 17, ...
        'Box', 'on', ...
        'LineWidth', 1);   % 边框加粗
        % title(ax, sprintf('%s - %s', row_title, metric), 'Interpreter','none');
        xlabel(ax, 'iteration'); 
        ylabel(ax, metric);
        
        % 设置x轴从0开始，自动适应数据范围
        if max_len > 0
            xlim(ax, [0, max_len-1]);
        end
        
        % grid(ax, 'on');
        ylim(ax, [0.15, 0.7]);
    end
end

% 在所有子图画完后，添加行标签
row_labels = {'SVFT', 'LoRA'};
for row = 1:2
    % 获取该行第一个子图的位置
    ax_first = axs(row, 1);
    pos = ax_first.Position;
    
    % 在子图左侧添加竖排文字
    annotation('textbox', [0.02, pos(2), 0.03, pos(4)], ...
        'String', row_labels{row}, ...
        'EdgeColor', 'none', ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'middle', ...
        'FontName', 'Times New Roman', ...
        'FontSize', 22, ...
        'FontWeight', 'bold', ...
        'Rotation', 90, ...
        'FitBoxToText', 'off');  % 竖着写
end
lgd = legend(axs(1,end), 'show');   % 用第一行最后一个子图的 handle
lgd.Layout.Tile = 'east';           % 把图例放到右侧空白区
lgd.FontName = 'Times New Roman';
lgd.FontSize = 17;
col_subs = {'a) math_mix', 'b) math_cls', 'c) math_sep_cls'};
for f = 1:num_sets
    ax = axs(2,f);
    pos = ax.Position;
    x_center = pos(1) + pos(3)/2;
    y_text = 0.02;  % 靠近最底部
    
    % 确保textbox的位置在[0,1]范围内
    x_box = max(0.01, min(0.95, x_center-0.12));
    annotation('textbox', [x_box, y_text, 0.24, 0.03], ...
        'String', col_subs{f}, ...
        'Units','normalized', ...
        'EdgeColor','none', ...
        'HorizontalAlignment','center', ...
        'VerticalAlignment','bottom', ...
        'FontName','Times New Roman', ...
        'FontSize', 17, ...
        'FontWeight','bold');
end




sgtitle();
