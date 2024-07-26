%% t-sne

    %%%
    idx_data_include = 1:length(paths_test);
    numsamp_max = inf;
    if ~(exist('feats_hc_tsne') && ~isempty(feats_hc_tsne))
    
    feats_hc_tsne = {};
    perps = [32 64 128 256];% 256];
    lr = 250;
    
        for perp = perps
        
            feats_hc_tsne_ = tsne(feats_hc(idx_data_include,:), 'Perplexity', perp, 'Standardize',true,'LearnRate',lr);
            feats_hc_tsne{end+1} = feats_hc_tsne_;
        
        end
        save([dir_plot '/hc_tsne.mat'], 'feats_hc_tsne', 'perps', 'idx_data_include', 'lr', 'numsamp_max');
    end
    
    
    %% plot t-sne: only GM GLs, train &val vs test
    
    close all
    
    colors_line = [
        3/4 3/4 3/4;...
        1-(1-[255/255 192/255 0/255])*1/3;...
        1-(1-[255/255 192/255 0/255])*2/3;...
        [255/255 192/255 0/255];...
    %     0.8 0.8 0;...
    %     0.6 0.6 0;...
        1-(1-[237/255 125/255 49/255])*1/3;...
        1-(1-[237/255 125/255 49/255])*2/3;...
        237/255 125/255 49/255;...
    %     0.8 0 0.8;...
    %     0 0.8 0.8;...
        1-(1-[0/255 176/255 80/255])*1/3;...
        1-(1-[0/255 176/255 80/255])*2/3;...
        0/255 176/255 80/255;...
        ];
    
    [colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');
    
    for idx_colors = 1:size(colors_line_,1)
        lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
    end
    
    for perp = perps
    num_dot = 0;
    for line_ = 1:10
        if line_ > 10
            continue
        end
    
        color_plot = colors_line_(line_,:);
    %     if sum(color_plot) > 3
    %         continue
    %     end
    % 
    %     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
    % 
    %     idx_data = [];
    %     for idx_line = 1:length(lines_plot)
    %         line_temp = lines_plot(idx_line);
    %         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
    %     end
        idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
        
    num_dot = num_dot+length(idx_data);
    
        marker_ = 'o';
    
        h_fig = figure(perp + 2000);
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
        
    
    end
    
    ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
    xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
    yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
    xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));
    
        ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
        xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d.fig', perp)])
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d.png', perp)])
    
    end


    %% plot t-sne: only GM GLs, train &val vs test - notime
    
    close all
    
    colors_line = [
        3/4 3/4 3/4;...
        [255/255 192/255 2/255];...
        [255/255 192/255 1/255];...
        [255/255 192/255 0/255];...
    %     0.8 0.8 0;...
    %     0.6 0.6 0;...
        237/255 125/255 41/255;...
        237/255 125/255 50/255;...
        237/255 125/255 49/255;...
    %     0.8 0 0.8;...
    %     0 0.8 0.8;...
        2/255 176/255 80/255;...
        1/255 176/255 80/255;...
        0/255 176/255 80/255;...
        ];
    
    [colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');
    
    for idx_colors = 1:size(colors_line_,1)
        lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
    end
    
    for perp = perps
    num_dot = 0;
    for line_ = 1:10
        if line_ > 10
            continue
        end
    
        color_plot = colors_line_(line_,:);
    %     if sum(color_plot) > 3
    %         continue
    %     end
    % 
    %     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
    % 
    %     idx_data = [];
    %     for idx_line = 1:length(lines_plot)
    %         line_temp = lines_plot(idx_line);
    %         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
    %     end
        idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
        
    num_dot = num_dot+length(idx_data);
    
        marker_ = 'o';
    
        h_fig = figure(perp + 2000);
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
        
    
    end
    
    ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
    xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
    yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
    xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));
    
        ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
        xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_notime.fig', perp)])
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_notime.png', perp)])
    
    end
    
    %% plot t-sne: only GM GLs, train &val vs test - only endo
    
    close all
    
    colors_line = [
        3/4 3/4 3/4;...
        1-(1-[255/255 192/255 0/255])*1/3;...
        1-(1-[255/255 192/255 0/255])*2/3;...
        [255/255 192/255 0/255];...
    %     0.8 0.8 0;...
    %     0.6 0.6 0;...
        3/4 3/4 3/4+0.01;...
        3/4 3/4 3/4+0.02;...
        3/4 3/4 3/4+0.03;...
    %     0.8 0 0.8;...
    %     0 0.8 0.8;...
        3/4 3/4 3/4+0.04;...
        3/4 3/4 3/4+0.05;...
        3/4 3/4 3/4+0.06;...
        ];
    
    [colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');
    
    for idx_colors = 1:size(colors_line_,1)
        lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
    end
    
    for perp = perps
    num_dot = 0;
    for line_ = [1 5 6 7 8 9 10 2 3 4]
        if line_ > 10
            continue
        end
    
        color_plot = colors_line_(line_,:);
    %     if sum(color_plot) > 3
    %         continue
    %     end
    % 
    %     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
    % 
    %     idx_data = [];
    %     for idx_line = 1:length(lines_plot)
    %         line_temp = lines_plot(idx_line);
    %         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
    %     end
        idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
        
    num_dot = num_dot+length(idx_data);
    
        marker_ = 'o';
    
        h_fig = figure(perp + 2000);
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
        
    
    end
    
    ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
    xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
    yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
    xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));
    
        ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
        xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_endo.fig', perp)])
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_endo.png', perp)])
    
    end
    
    %% plot t-sne: only GM GLs, train &val vs test - only Meso
    
    close all
    
    colors_line = [
        3/4 3/4 3/4;...
        3/4 3/4 3/4+0.01;...
        3/4 3/4 3/4+0.02;...
        3/4 3/4 3/4+0.03;...
    %     0.8 0.8 0;...
    %     0.6 0.6 0;...
        1-(1-[237/255 125/255 49/255])*1/3;...
        1-(1-[237/255 125/255 49/255])*2/3;...
        237/255 125/255 49/255;...
    %     0.8 0 0.8;...
    %     0 0.8 0.8;...
        3/4 3/4 3/4+0.04;...
        3/4 3/4 3/4+0.05;...
        3/4 3/4 3/4+0.06;...
        ];
    
    [colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');
    
    for idx_colors = 1:size(colors_line_,1)
        lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
    end
    
    for perp = perps
    num_dot = 0;
    for line_ = [1 2 3 4 8 9 10 5 6 7 ]
        if line_ > 10
            continue
        end
    
        color_plot = colors_line_(line_,:);
    %     if sum(color_plot) > 3
    %         continue
    %     end
    % 
    %     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
    % 
    %     idx_data = [];
    %     for idx_line = 1:length(lines_plot)
    %         line_temp = lines_plot(idx_line);
    %         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
    %     end
        idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
        
    num_dot = num_dot+length(idx_data);
    
        marker_ = 'o';
    
        h_fig = figure(perp + 2000);
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
        
    
    end
    
    ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
    xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
    yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
    xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));
    
        ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
        xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_meso.fig', perp)])
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_meso.png', perp)])
    
    end
    
    %% plot t-sne: only GM GLs, train &val vs test - only Ecto
    
    close all
    
    colors_line = [
        3/4 3/4 3/4;...
        3/4 3/4 3/4+0.01;...
        3/4 3/4 3/4+0.02;...
        3/4 3/4 3/4+0.03;...
    %     0.8 0.8 0;...
    %     0.6 0.6 0;...
        3/4 3/4 3/4+0.04;...
        3/4 3/4 3/4+0.05;...
        3/4 3/4 3/4+0.06;...
    %     0.8 0 0.8;...
    %     0 0.8 0.8;...
        1-(1-[0/255 176/255 80/255])*1/3;...
        1-(1-[0/255 176/255 80/255])*2/3;...
        0/255 176/255 80/255;...
        ];
    
    [colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');
    
    for idx_colors = 1:size(colors_line_,1)
        lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
    end
    
    for perp = perps
    num_dot = 0;
    for line_ = [1 2 3 4 5 6 7 8 9 10 ]
        if line_ > 10
            continue
        end
    
        color_plot = colors_line_(line_,:);
    %     if sum(color_plot) > 3
    %         continue
    %     end
    % 
    %     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
    % 
    %     idx_data = [];
    %     for idx_line = 1:length(lines_plot)
    %         line_temp = lines_plot(idx_line);
    %         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
    %     end
        idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
        
    num_dot = num_dot+length(idx_data);
    
        marker_ = 'o';
    
        h_fig = figure(perp + 2000);
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
        
    
    end
    
    ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
    xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
    yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
    xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));
    
        ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
        xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_ecto.fig', perp)])
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_ecto.png', perp)])
    
    end
    %% plot with hc feat
    close all
    
    perp_set = 256;
    
    perp = perp_set;
    
    cmap = ones(256,3);
    cmap(1:193,2) = 1:-1/192:0;
    cmap(193:256,2) = 0;
    cmap(192:256,1)=1:-1/192:2/3;
    cmap(192:256,3)=1:-1/192:2/3;
    
    % cmap = jet;
    
    for idx_feat_hc = 1:size(feats_hc,2)%[3 15 16 20 21 23 25 32 33]% [1 3 5 13 15 16 20 21 24 25 30 32 33] %[1 3 13 15 16 ] %1:length(names_feat_hc)
    
        close all
        if isfile([dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s.fig', perp, names_feat_hc{idx_feat_hc})])
            continue
        end
        feats_hc_ = feats_hc(:,idx_feat_hc);
        mean_hc_ = mean(feats_hc_);
        std_hc_ = std(feats_hc_);
    
        feats_hc_ = (feats_hc_-mean_hc_)/(std_hc_);
        feats_hc_(feats_hc_<-2) = -2;
        feats_hc_(feats_hc_>2) = 2;
    
        h_fig = figure(idx_feat_hc);
        for idx_data = 1:length(feats_hc)
            if lines_test(idx_data) > 10 %filter out none-GMGL
                continue
            end
            if lines_test(idx_data) == 0 %filter out none-GMGL
                continue
            end
            marker_ = 'o';
            color_plot = cmap(round((feats_hc_(idx_data)+2)/4*255+1),:);
    
            set(0,'CurrentFigure',h_fig)
            h_fig.Position = [0 0 900 900];
            h_fig.Color = [1 1 1];
            hold on
        
            size_ = 66;
            feats_hc_tsne_ = feats_hc_tsne{find(perp==perps)};
            scatter(feats_hc_tsne_(idx_data,1),feats_hc_tsne_(idx_data,2),marker_,...
            'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
            'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
        %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
        %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
        end
        ycen = min(feats_hc_tsne_(:,2))/2+max(feats_hc_tsne_(:,2))/2;
        xcen = min(feats_hc_tsne_(:,1))/2+max(feats_hc_tsne_(:,1))/2;
        yax = max(feats_hc_tsne_(:,2))-min(feats_hc_tsne_(:,2));
        xax = max(feats_hc_tsne_(:,1))-min(feats_hc_tsne_(:,1));
        
            ylim([min(feats_hc_tsne_(:,2)) max(feats_hc_tsne_(:,2))]);
            xlim([min(feats_hc_tsne_(:,1)) max(feats_hc_tsne_(:,1))]);
        
        set(gcf,'color',[1 1 1])
        axis image
        axis off
        saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s.fig', perp, names_feat_hc{idx_feat_hc})])
        saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s.png', perp, names_feat_hc{idx_feat_hc})])
    
    end



%% plot with 10% sample
close all

names_feat_hc = {'volume_colony','area_colony','RI_avg_colony','RI_std_colony',...
                'volume_lip','area_lip','RI_avg_lip','RI_std_lip',...# volume_cyto = volume_colony-volume_lip
                'volume_cyto','area_cyto','RI_avg_cyto','RI_std_cyto',...
                'area_gap','len_bound','roundness','solidity','eccentricity',...
                'n_boundin','n_boundout','ncont_bound',...
                'thick_avg','thick_std', 'spread_thick', 'skew_thick', 'kurt_thick',...
                'spread_dm', 'skew_dm', 'kurt_dm',...
                'spread_lip', 'skew_lip', 'kurt_lip',...
                'ratio_area_gap', 'ratio_volume_lip'};

feats_hc(:,end+1) = feats_hc(:,13)./feats_hc(:,2);
feats_hc(:,end+1) = feats_hc(:,5)./feats_hc(:,1);


perp_set = 256;

perp = perp_set;

cmap = ones(256,3);
cmap(1:193,2) = 1:-1/192:0;
cmap(193:256,2) = 0;
cmap(192:256,1)=1:-1/192:2/3;
cmap(192:256,3)=1:-1/192:2/3;

% cmap = jet;

for idx_feat_hc = [3 20 32 16 15 33 21 34 35] %[1 3 13 15 16 ] %1:length(names_feat_hc)

    close all
    if false%isfile([dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s_s0.25.fig', perp, names_feat_hc{idx_feat_hc})])
        continue
    end
    feats_hc_ = feats_hc(:,idx_feat_hc);
    mean_hc_ = mean(feats_hc_);
    std_hc_ = std(feats_hc_);

    feats_hc_ = (feats_hc_-mean_hc_)/(std_hc_);
    feats_hc_(feats_hc_<-2) = -2;
    feats_hc_(feats_hc_>2) = 2;

    h_fig = figure(idx_feat_hc);%
    for idx_data = 1:length(feats_hc)
        if mod(idx_data,4) ~= 1
            continue
        end
        if lines_test(idx_data) > 10 %filter out none-GMGL
            continue
        end
        if lines_test(idx_data) ==0 %filter out none-GMGL
            continue
        end
        marker_ = 'o';
        color_plot = cmap(round((feats_hc_(idx_data)+2)/4*255+1),:);

        set(0,'CurrentFigure',h_fig)
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_hc_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_hc_tsne_(idx_data,1),feats_hc_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    end
    ycen = min(feats_hc_tsne_(:,2))/2+max(feats_hc_tsne_(:,2))/2;
    xcen = min(feats_hc_tsne_(:,1))/2+max(feats_hc_tsne_(:,1))/2;
    yax = max(feats_hc_tsne_(:,2))-min(feats_hc_tsne_(:,2));
    xax = max(feats_hc_tsne_(:,1))-min(feats_hc_tsne_(:,1));
    
        ylim([min(feats_hc_tsne_(:,2)) max(feats_hc_tsne_(:,2))]);
        xlim([min(feats_hc_tsne_(:,1)) max(feats_hc_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s_s0.25.fig', perp, names_feat_hc{idx_feat_hc})])
    saveas(h_fig, [dir_plot '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s_s0.25.png', perp, names_feat_hc{idx_feat_hc})])

end


%% plot relative table - endo
idxs_feat_select = [3 20 32 16 15 33 21 34 35];

lines_select = [1 2 3 4 5 6 7 8 9 10];

table_feat = zeros(size(feats_hc,2), length(lines_select));

for idx_feat_ = 1:size(feats_hc,2)
for line_ = lines_select
    
    idxs_thisline = find(lines_test == line_);
    feats_hc_ = feats_hc(idxs_thisline,idx_feat_);
    table_feat(idx_feat_,line_) = mean(feats_hc_);

end
end

table_feat_norm = table_feat;

for idx_feat_ = idxs_feat_select

    row_feat = table_feat_norm(idx_feat_, :);
    row_feat = (row_feat-min(row_feat))/(max(row_feat)-min(row_feat));
    table_feat_norm(idx_feat_, :) = row_feat;
end
table_feat_norm_select = table_feat_norm(idxs_feat_select,:);
figure(532),imagesc(table_feat_norm_select), colormap(cmap), axis image

 saveas(gcf, [dir_feat '/' sprintf('table_feat_select.fig')])
    saveas(gcf, [dir_feat '/' sprintf('table_feat_select.png')])
