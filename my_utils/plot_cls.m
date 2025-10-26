num_nodes = 10;
metric = "test_acc";  % 只画test_acc

% CSV文件路径
csv_files = {"classify.csv","classify_lora.csv"};

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

figure('Position',[100,100,1600,500]); 
tl = tiledlayout(1, num_sets, ...
    'TileSpacing','compact', ...
    'Padding','loose');   % 用 loose 给上下左右多留白
outerpos = tl.OuterPosition;   % [x y w h]
outerpos(2) = 0.1;            % 把底部抬高一点，留空间
outerpos(4) = 0.85;             % 高度相应缩小
tl.OuterPosition = outerpos;
axs = gobjects(1, num_sets);

% 只画一行
for f = 1:num_sets
    csv_file = csv_files{f};
    
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
    ax = nexttile(f); hold(ax,'on');
    axs(f) = ax;  % 记录轴
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
                     'LineWidth', 1, ...
                     'DisplayName', sprintf('Node_%d', node-1));  % node从1开始，但编号从0开始
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

    % === 轴样式 ===
    set(ax, 'FontName', 'Times New Roman', ...
    'FontSize', 17, ...
    'Box', 'on', ...
    'LineWidth', 1);   % 边框加粗
    xlabel(ax, 'iteration'); 
    ylabel(ax, metric);
    
    % 设置x轴从0开始，自动适应数据范围
    if max_len > 0
        xlim(ax, [0, max_len-1]);
    end
    
    % grid(ax, 'on');
    ylim(ax, [0.0, 0.75]);
end

% 添加图例
lgd = legend(axs(end), 'show');   % 用最后一个子图的 handle
lgd.Layout.Tile = 'east';           % 把图例放到右侧空白区
lgd.FontName = 'Times New Roman';
lgd.FontSize = 17;
col_subs = {'a) SVFT', 'b) LoRA'};
for f = 1:num_sets
    ax = axs(f);
    pos = ax.Position;
    x_center = pos(1) + pos(3)/2;
    y_text = 0.02;  % 靠近最底部
    
    % 确保textbox的位置在[0,1]范围内
    x_box = max(0.01, min(0.95, x_center-0.08));
    annotation('textbox', [x_box, y_text, 0.16, 0.03], ...
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
