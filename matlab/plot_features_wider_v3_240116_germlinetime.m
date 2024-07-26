%% load features
close all
clear
clc
addpath('/data01/gkim/Matlab_subcodes/gk/')

dir_total = '/data02/gkim/stem_cell_jwshin/data/';
cd(dir_total)
list_exp = dir('2*feat_wider_v3');

feats_ = [];
lines_ = [];
types_ = [];
%high line: 1
%low line: 2
%high ipsc: 3
%low ipsc: 4

for iter_exp = 1:length(list_exp)
    cd(dir_total)
    cd(list_exp(iter_exp).name)
    cd('00_train')
    list_cls = dir('*_*');

    for iter_cls = 1:length(list_cls)

        fold_cls = list_cls(iter_cls).name;
        if contains(fold_cls, 'GM') || contains(fold_cls, 'gm') || contains(fold_cls, '230719.')  || contains(fold_cls, '230720.') 
            if contains(fold_cls, 'Endo') && contains(fold_cls,{'12h', '12H'})
                line_ = 2;
                type_ = 3;%2;
            elseif contains(fold_cls, 'Endo') && contains(fold_cls,{'24h', '24H'})
                line_ = 3;
                type_ = 3;
            elseif contains(fold_cls,'Meso') && contains(fold_cls,{'12h', '12H'})
                line_ = 4;
                type_ = 5;%4;
            elseif contains(fold_cls,'Meso') && contains(fold_cls,{'24h', '24H'})
                line_ = 5;
                type_ = 5;
            elseif contains(fold_cls,'Ecto') && contains(fold_cls,{'12h', '12H'})
                line_ = 6;
                type_ = 7;%6;
            elseif contains(fold_cls,'Ecto') && contains(fold_cls,{'24h', '24H'})
                line_ = 7;
                type_ = 7;
            elseif contains(fold_cls,{'untreat', 'Untreat'})
                line_ = 1;
                type_ = 1;
            else
                continue
            end

        elseif contains(fold_cls, {'HD02_'})
            line_ = -4 + 13;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD03_'})
            line_ = -4 + 14;
            type_ = 8;%3;
        elseif contains(fold_cls, {'HD04_'})
            line_ = -4 + 15;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD05_'})
            line_ = -4 + 16;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD07_'})
            line_ = -4 + 17;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD09_'})
            line_ = -4 + 18;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD11_'})
            line_ = -4 + 19;
            type_ = 8;%3;
        elseif contains(fold_cls, {'HD12_'})
            line_ = -4 + 20;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD14_'})
            line_ = -4 + 21;
            type_ = 8;%3;%4;
        elseif contains(fold_cls, {'HD15_'})
            line_ = -4 + 22;
            type_ = 8;%3;
        elseif contains(fold_cls, {'HD18_'})
            line_ = -4 + 23;
            type_ = 8;%3;
        elseif contains(fold_cls, {'HD25_'})
            line_ = -4 + 24;
            type_ = 8;%3;
        
        elseif contains(fold_cls, {'B1_', 'BJ01_'})
            line_ = -4 + 26;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B4_', 'BJ04_'})
            line_ = -4 + 27;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B7_', 'BJ07_'})
            line_ = -4 + 28;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B11_', 'BJ11_'})
            line_ = -4 + 29;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B12_', 'BJ12_'})
            line_ = -4 + 30;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B13_', 'BJ13_'})
            line_ = -4 + 31;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B14_', 'BJ14_'})
            line_ = -4 + 32;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B17_', 'BJ17_'})
            line_ = -4 + 33;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B18_', 'BJ18_'})
            line_ = -4 + 34;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B21_', 'BJ21_'})
            line_ = -4 + 35;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B22_', 'BJ22_'})
            line_ = -4 + 36;
            type_ = 8;%3;
        elseif contains(fold_cls, {'B23_', 'BJ23_'})
            line_ = -4 + 37;
            type_ = 8;%3;


        elseif contains(fold_cls, {'A2_'})
            line_ = -4 + 39;
            type_ = 8;%3;
        elseif contains(fold_cls, {'A12_'})
            line_ = -4 + 40;
            type_ = 8;%3;
        elseif contains(fold_cls, {'A19_'})
            line_ = -4 + 41;
            type_ = 8;%4;
        elseif contains(fold_cls, {'A20_'})
            line_ = -4 + 42;
            type_ = 8;%4;
        
        else
            continue
        end
        
        cd(dir_total)
        cd(list_exp(iter_exp).name)
        cd('00_train')
        cd(fold_cls)
        list_feat = dir('*.mat');
        for iter_feat = 1:length(list_feat)
            load(list_feat(iter_feat).name);
            feat_ = [volume_colony,area_colony,RI_avg_colony,RI_std_colony,...
                volume_lip,area_lip,RI_avg_lip,RI_std_lip,...# volume_cyto = volume_colony-volume_lip
                volume_cyto,area_cyto,RI_avg_cyto,RI_std_cyto,...
                area_gap,len_bound,roundness,solidity,eccentricity,...
                n_boundin,n_boundout,ncont_bound,...
                thick_avg,thick_std, spread_thick/sqrt(area_colony), sqrt(sum(abs(skew_thick).^2)), kurt_thick,...
                spread_dm/sqrt(area_colony), sqrt(sum(abs(skew_dm).^2)), kurt_dm,...
                spread_lip/sqrt(area_colony), sqrt(sum(abs(skew_lip).^2)), kurt_lip,...
                volume_lip/volume_colony,...
                area_gap/area_colony,...
                thick_avg/sqrt(area_colony)];
            feats_ = [feats_; feat_];
            lines_ = [lines_; line_];
            types_ = [types_; type_];
        end

    end

end


names_feat = {'volume_colony','area_colony','RI_avg_colony','RI_std_colony',...
                'volume_lip','area_lip','RI_avg_lip','RI_std_lip',...# volume_cyto = volume_colony-volume_lip
                'volume_cyto','area_cyto','RI_avg_cyto','RI_std_cyto',...
                'area_gap','len_bound','roundness','solidity','eccentricity',...
                'n_boundin','n_boundout','ncont_bound',...
                'thick_avg','thick_std', 'spread_thick', 'skew_thick', 'kurt_thick',...
                'spread_dm', 'skew_dm', 'kurt_dm',...
                'spread_lip', 'skew_lip', 'kurt_lip',...
                'volume_ratio_lip',...
                'area_ratio_gap',...
                'aspect_ratio_xz'};
%% plot features

colors_type = [0 0 0;...
    0.5 0.5 0;...
    1 1 0;...
    0 0.5 0.5;...
    0 1 1;...
    0.5 0 0.5;...
    1 0 1];

labels_xtick = {'GM',
        'GM + endo 12h',
        'GM + endo 24h',
        'GM + meso 12h',
        'GM + meso 24h',
        'GM + ecto 12h',
        'GM + ecto 24h'};

dir_plot = '/data02/gkim/stem_cell_jwshin/data_present/240122_features_germlayer';
mkdir(dir_plot)

for iter_feat = 1:size(feats_,2)
    
    close

    h_fig = figure(iter_feat);
    h_fig.Position = [0 0 1200 600];
    h_fig.Color = [1 1 1];
    hold on
    for line_ = unique(lines_)'
        idxs_thisline = find(lines_ == line_);

        y_ = feats_(idxs_thisline,iter_feat);
        x_ = line_;

        bar(x_,mean(y_),'LineWidth',2,'FaceColor','none','FaceColor', colors_type(types_(idxs_thisline(1)),:));
        errorbar(x_,mean(y_),std(y_,0,1), 'k','LineWidth',2);

        
    end
    ax = gca;
    
    % Set the axis and tick width
    axis_line_width = 2;  % adjust as needed
    tick_line_width = 2;  % adjust as needed
    
    ax.LineWidth = axis_line_width;
    
    % Set the tick width
    ax.XAxis.LineWidth = tick_line_width;
    ax.YAxis.LineWidth = tick_line_width;

    if contains(names_feat{iter_feat},'RI_avg')
        ylim([1.337, inf])

    elseif contains(names_feat{iter_feat},'n_')
        ylim([1.337, inf])


    elseif contains(names_feat{iter_feat},'kurt_')
        ylim([0, inf])
    else
        ylim([0 inf])
    end


    xticks(unique(lines_));
    xticklabels(labels_xtick);
    set(gca, 'FontSize', 15)
    title(strrep(names_feat{iter_feat}, '_', ' '));

    saveas(gcf,[dir_plot, '/', names_feat{iter_feat}, '.fig']);
    saveas(gcf,[dir_plot, '/', names_feat{iter_feat}, '.png']);

end


%% plot qPCR
% 
% OCT4_avg = [1	0.777851355	1.450030709	0.922202656		11.03319338	11.43110421	6.915409953	3.973021174		0.69293594	0.783471674	0.712521777	0.923294007	0.854028467	0.846877371	0.78710852	0.853528837		2.290726751	4.648651032	2.150177454	1.865306942	4.492078541	2.472146714	0.611291212	1.363239028	4.389258906];
% OCT4_std = [0.003257945	0.041235028	0.031033553	0.020653638		1.21977653	1.223138904	0.570315383	0.353706597		0.012552804	0.00187261	0.022198777	0.018281663	0.011235409	0.010873495	0.032590236	0.006108661		0.061525869	0.262333613	0.097738797	0.053311902	0.09411293	0.02964216	0.007473565	0.003221493	0.037883939];
% 
% 
% NANOG_avg = [1	0.446624538	1.368657399	0.550257273		7.604590809	7.477475196	4.805419176	1.900036523		0.994066021	0.909748488	1.024658729	0.90678719	1.100288893	0.957245731	1.018915132	1.111536849		2.439288794	10.30103129	1.929762614	4.487055558	9.918823711	7.905816653	1.042549606	3.735464081	9.191721048];
% NANOG_std = [0.003246835	0.006828827	0.011311712	0.006008994		0.617739621	0.555396822	0.172721782	0.06763154		0.019097912	0.003627839	0.024053431	0.017425399	0.015299016	0.012711375	0.014139658	0.008701237		0.129762537	0.573281091	0.095207592	0.178235836	0.371932049	0.239493202	0.014133579	0.086211206	0.113681055];
% 
% 
% ZFP42_avg = [1	1.092502082	0.759993215	0.732676165		3.700124607	2.548564379	2.271408197	1.090771507		1.110665684	0.962625537	0.609387487	0.140564306	1.324166152	1.208097191	1.2648426	1.25441229		2.820450181	3.575514345	2.425259255	1.871443168	3.95841082	1.911016718	0.833510272	1.382972859	4.080982608];
% ZFP42_std = [0.012493487	0.010664712	0.003626181	0.013060486		0.361887044	0.300613362	0.297023957	0.113303467		0.031876047	0.008829379	0.010176988	0.001618759	0.015918313	0.02905865	0.014702339	0.022017362		0.133195915	0.225548579	0.09823669	0.053895685	0.080793433	0.048986529	0.00838595	0.016785835	0.0703423];
% %% OCT4
% close all
% h_fig = figure(iter_feat);
% h_fig.Position = [0 0 1200 600];
% h_fig.Color = [1 1 1];
% hold on
% 
% count_ = 0;
% for line_ = unique(lines_)'
%     count_ = count_+1;
%         idxs_thisline = find(lines_ == line_);
% 
%         x_ = line_;
% 
%         bar(x_,OCT4_avg(count_),'LineWidth',2,'FaceColor','none','FaceColor', colors_type(types_(idxs_thisline(1)),:));
%         errorbar(x_,OCT4_avg(count_),OCT4_std(count_), 'k','LineWidth',2);
%     end
% ax = gca;
% 
% % Set the axis and tick width
% axis_line_width = 2;  % adjust as needed
% tick_line_width = 2;  % adjust as needed
% 
% ax.LineWidth = axis_line_width;
% 
% % Set the tick width
% ax.XAxis.LineWidth = tick_line_width;
% ax.YAxis.LineWidth = tick_line_width;
% 
% xticks(unique(lines_));
% xticklabels(labels_xtick);
% set(gca, 'FontSize', 15)
% title(strrep('OCT4_relative_expression', '_', ' '));
% 
% saveas(gcf,[dir_plot, '/', 'OCT4_relative_expression', '.fig']);
% saveas(gcf,[dir_plot, '/', 'OCT4_relative_expression', '.png']);
% 
% %% NANOG
% close all
% h_fig = figure(iter_feat);
% h_fig.Position = [0 0 1200 600];
% h_fig.Color = [1 1 1];
% hold on
% 
% count_ = 0;
% for line_ = unique(lines_)'
%     count_ = count_+1;
%         idxs_thisline = find(lines_ == line_);
% 
%         x_ = line_;
% 
%         bar(x_,NANOG_avg(count_),'LineWidth',2,'FaceColor','none','FaceColor', colors_type(types_(idxs_thisline(1)),:));
%         errorbar(x_,NANOG_avg(count_),OCT4_std(count_), 'k','LineWidth',2);
%     end
% ax = gca;
% 
% % Set the axis and tick width
% axis_line_width = 2;  % adjust as needed
% tick_line_width = 2;  % adjust as needed
% 
% ax.LineWidth = axis_line_width;
% 
% % Set the tick width
% ax.XAxis.LineWidth = tick_line_width;
% ax.YAxis.LineWidth = tick_line_width;
% 
% xticks(unique(lines_));
% xticklabels(labels_xtick);
% set(gca, 'FontSize', 15)
% title(strrep('NANOG_relative_expression', '_', ' '));
% 
% saveas(gcf,[dir_plot, '/', 'NANOG_relative_expression', '.fig']);
% saveas(gcf,[dir_plot, '/', 'NANOG_relative_expression', '.png']);
% 
% 
% %% ZFP42
% close all
% h_fig = figure(iter_feat);
% h_fig.Position = [0 0 1200 600];
% h_fig.Color = [1 1 1];
% hold on
% 
% count_ = 0;
% for line_ = unique(lines_)'
%     count_ = count_+1;
%         idxs_thisline = find(lines_ == line_);
% 
%         x_ = line_;
% 
%         bar(x_,ZFP42_avg(count_),'LineWidth',2,'FaceColor','none','FaceColor', colors_type(types_(idxs_thisline(1)),:));
%         errorbar(x_,ZFP42_avg(count_),OCT4_std(count_), 'k','LineWidth',2);
%     end
% ax = gca;
% 
% % Set the axis and tick width
% axis_line_width = 2;  % adjust as needed
% tick_line_width = 2;  % adjust as needed
% 
% ax.LineWidth = axis_line_width;
% 
% % Set the tick width
% ax.XAxis.LineWidth = tick_line_width;
% ax.YAxis.LineWidth = tick_line_width;
% 
% xticks(unique(lines_));
% xticklabels(labels_xtick);
% set(gca, 'FontSize', 15)
% title(strrep('ZFP42_relative_expression', '_', ' '));
% 
% saveas(gcf,[dir_plot, '/', 'ZFP42_relative_expression', '.fig']);
% saveas(gcf,[dir_plot, '/', 'ZFP42_relative_expression', '.png']);
