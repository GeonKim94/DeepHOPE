clear
clc
close all
dir_data = dir('/data02/gkim/stem_cell_jwshin/data/23*feature_v3');


smoothness_ = [];
ncont_bound_ = [];
area_col_ = [];
area_nuc_ = [];
area_lip_ = [];
area_gap_ = [];
num_lip_ = [];
thick_mean_ =[];
thick_std_ = [];
thick_peak_ = [];
center_disp_ = [];
spread_thick_ = [];
skew_thick_ = [];
kurt_thick_ = [];
RI_mean_ = [];
RI_std_ = [];
majax_ = [];
minax_ = [];
eccen_ = [];
circu_ = [];


cond_ = [];
hour_ = [];
colony_ = [];

hours = 0:1:24;
        for hour = hours
            cd('/data02/gkim/stem_cell_jwshin/data/230323_feature_v3/00_train/h9_RA&E6/')

            dir_feat = dir(sprintf('*hr%02d.mat',hour));
            for iter_feat = 1:length(dir_feat)
                
                fname_feat = dir_feat(iter_feat).name;

                if contains(fname_feat, '.A1.')
                    cond = 1; % ctl
                elseif contains(fname_feat, '.A2.')
                    cond = 1; % ctl  
                elseif contains(fname_feat, '.A3.')
                    cond = 2; % RA
                elseif contains(fname_feat, '.B3.')
                    cond = 2; % RA  
                elseif contains(fname_feat, '.B1.')
                    cond = 3; % E6
                elseif contains(fname_feat, '.B2.')
                    cond = 3; % E6
                else
                    error('no cond info found on the directory')
                end

                cond_ = [cond_; cond];
                
                load(fname_feat);

                smoothness_ = [smoothness_; smoothness];
                ncont_bound_ = [ncont_bound_; ncont_bound];
                area_col_ = [area_col_; area_col];
                area_nuc_ = [area_nuc_; area_nuc];
                area_lip_ = [area_lip_; area_lip];
                area_gap_ = [area_gap_; area_gap];
                num_lip_ = [num_lip_; num_lip];
                thick_mean_ =[thick_mean_; thick_mean];
                thick_std_ = [thick_std_; thick_std];
                thick_peak_ = [thick_peak_; thick_peak];
                center_disp_ = [center_disp_; center_disp];
                spread_thick_ = [spread_thick_; spread_thick];
                skew_thick_ = [skew_thick_; sqrt(sum(skew_thick).^2)];
                kurt_thick_ = [kurt_thick_; kurt_thick];
                RI_mean_ = [RI_mean_; RI_mean];
                RI_std_ = [RI_std_; RI_std];
                hour_ = [hour_; hour];
                majax_ = [majax_;majax];
                minax_ = [minax_;minax];
                eccen_ = [eccen_;eccen];
                circu_ = [circu_;circu];

                colony_ = [colony_; iter_feat];

            end
        end

ratio_asp_ = thick_mean_./(area_col_).^0.5;
pointiness_ = thick_peak_./(area_col_).^0.5;

ratio_nuc_ = area_nuc_./area_col_;
ratio_lip_ = area_lip_./area_col_;
ratio_gap_ = area_gap_./area_col_;
ratio_ax_ = minax_./majax_;

spread_thick_norm_ = spread_thick_./(area_col_).^0.5;

features_ = [smoothness_, ncont_bound_, area_col_, area_lip_,area_nuc_,...
    area_gap_,...
    ratio_nuc_,ratio_lip_,ratio_gap_, spread_thick_norm_...
    num_lip_, thick_mean_, thick_std_, thick_peak_,...
    center_disp_, spread_thick_, skew_thick_, kurt_thick_,...
    RI_mean_, RI_std_, ratio_asp_, pointiness_];

names_feature = {'smoothness', 'boundary contrast', 'colony area', 'LD area','nucleus area',...
    'gap area','nucleus ratio', 'LD ratio', 'gap ratio','normalized thickness spread'...
    'LD number', 'thickness mean', 'thickness std', 'thickness peak','center displacment',...
    'thickness spread', 'thickness skewness', 'thickness kurtosis','RI mean', 'RI std',...
    'aspect ratio', 'pointiness','majax', 'minax', 'eccen',...
    'circu', 'ratio_ax_'};



%% remove outliers

% idxs_outlier_ = [];
% % for hour = hour_
% % for cond = 1:max(cond_)
% %         for iter_fea = 1:size(features_,2)
% %                 idxs_data = find((hour_ == hour).*(line_ == line));
% %                 if length(idxs_data) == 0
% %                     continue
% %                 end
% %                 
% %                 y = features_(idxs_data,iter_fea);
% %                 
% %                 idxs_outlier = idxs_data([find(y > mean(y)+4*std(y,0,1));
% %                     find(y < mean(y)-4*std(y,0,1))]);
% %                 
% %                 idxs_outlier_ = [idxs_outlier_; idxs_outlier];
% %         end
% % 
% %     end
% % end
% idxs_outlier_ = unique(idxs_outlier_);
% features_(idxs_outlier_,:) = [];
% cond_(idxs_outlier_,:) = [];
% line_(idxs_outlier_,:) = [];
%% plot features

close all

sign_plot_ = {'-o', '-o', '-o'};
color_plot_ = {[0 0 0], [0.25 1 0.25], [1 0.25 0.25]};
for colony = 1:max(colony_)
    idx_colony = find(colony_ == colony);
    cond = cond_(idx_colony(1));

    [nothing, order_sort] = sort(hour_(idx_colony), 'ascend');
    idx_colony = idx_colony(order_sort);

    for iter_fea = 1:size(features_,2)
        h_ = figure(iter_fea);
        h_.Position = [0 0 1024 768];
        h_.Color = [1 1 1];
        hold on
        sign_plot = sign_plot_{cond};
        color_plot = color_plot_{cond};
        plot(hours, features_(idx_colony, iter_fea), sign_plot,...
            'Color',color_plot, 'MarkerSize', 5, ...
            'MarkerFaceColor',color_plot);
        ttl_fig = ['<' names_feature{iter_fea} '>'];
        title(ttl_fig)
    end
    
end


%% comparison between terminal state
clear
clc
close all
dir_data = dir('/data02/gkim/stem_cell_jwshin/data/23*feature_v3');


smoothness_ = [];
ncont_bound_ = [];
area_col_ = [];
area_nuc_ = [];
area_lip_ = [];
area_gap_ = [];
num_lip_ = [];
thick_mean_ =[];
thick_std_ = [];
thick_peak_ = [];
center_disp_ = [];
spread_thick_ = [];
skew_thick_ = [];
kurt_thick_ = [];
RI_mean_ = [];
RI_std_ = [];
majax_ = [];
minax_ = [];
eccen_ = [];
circu_ = [];


cond_ = [];
hour_ = [];
colony_ = [];

hours = [0 24];
        for hour = hours
            cd('/data02/gkim/stem_cell_jwshin/data/230323_feature_v3/00_train/h9_RA&E6/')

            dir_feat = dir(sprintf('*hr%02d.mat',hour));
            for iter_feat = 1:length(dir_feat)
                
                fname_feat = dir_feat(iter_feat).name;

                if contains(fname_feat, '.A1.')
                    cond = 1; % ctl
                elseif contains(fname_feat, '.A2.')
                    cond = 1; % ctl  
                elseif contains(fname_feat, '.A3.')
                    cond = 2; % RA
                elseif contains(fname_feat, '.B3.')
                    cond = 2; % RA  
                elseif contains(fname_feat, '.B1.')
                    cond = 3; % E6
                elseif contains(fname_feat, '.B2.')
                    cond = 3; % E6
                else
                    error('no cond info found on the directory')
                end

                cond_ = [cond_; cond];
                
                load(fname_feat);

                smoothness_ = [smoothness_; smoothness];
                ncont_bound_ = [ncont_bound_; ncont_bound];
                area_col_ = [area_col_; area_col];
                area_nuc_ = [area_nuc_; area_nuc];
                area_lip_ = [area_lip_; area_lip];
                area_gap_ = [area_gap_; area_gap];
                num_lip_ = [num_lip_; num_lip];
                thick_mean_ =[thick_mean_; thick_mean];
                thick_std_ = [thick_std_; thick_std];
                thick_peak_ = [thick_peak_; thick_peak];
                center_disp_ = [center_disp_; center_disp];
                spread_thick_ = [spread_thick_; spread_thick];
                skew_thick_ = [skew_thick_; sqrt(sum(skew_thick).^2)];
                kurt_thick_ = [kurt_thick_; kurt_thick];
                RI_mean_ = [RI_mean_; RI_mean];
                RI_std_ = [RI_std_; RI_std];
                hour_ = [hour_; hour];
                majax_ = [majax_;majax];
                minax_ = [minax_;minax];
                eccen_ = [eccen_;eccen];
                circu_ = [circu_;circu];

                colony_ = [colony_; iter_feat];

            end
        end

ratio_asp_ = thick_mean_./(area_col_).^0.5;
pointiness_ = thick_peak_./(area_col_).^0.5;

ratio_nuc_ = area_nuc_./area_col_;
ratio_lip_ = area_lip_./area_col_;
ratio_gap_ = area_gap_./area_col_;
ratio_ax_ = minax_./majax_;

spread_thick_norm_ = spread_thick_./(area_col_).^0.5;

features_ = [smoothness_, ncont_bound_, area_col_, area_lip_,area_nuc_,...
    area_gap_,...
    ratio_nuc_,ratio_lip_,ratio_gap_, spread_thick_norm_...
    num_lip_, thick_mean_, thick_std_, thick_peak_,...
    center_disp_, spread_thick_, skew_thick_, kurt_thick_,...
    RI_mean_, RI_std_, ratio_asp_, pointiness_];

names_feature = {'smoothness', 'boundary contrast', 'colony area', 'LD area','nucleus area',...
    'gap area','nucleus ratio', 'LD ratio', 'gap ratio','normalized thickness spread'...
    'LD number', 'thickness mean', 'thickness std', 'thickness peak','center displacment',...
    'thickness spread', 'thickness skewness', 'thickness kurtosis','RI mean', 'RI std',...
    'aspect ratio', 'pointiness','majax', 'minax', 'eccen',...
    'circu', 'ratio_ax_'};


%%

sign_plot_ = {'o', 'o', 'o'};
color_plot_ = {[0 0 0], [0.25 1 0.25], [1 0.25 0.25]};
for colony = 1:max(colony_)
    idx_colony = find(colony_ == colony);
    cond = cond_(idx_colony(1));

    [nothing, order_sort] = sort(hour_(idx_colony), 'ascend');
    idx_colony = idx_colony(order_sort);

    for iter_fea = 1:size(features_,2)
        h_ = figure(100+iter_fea);
        h_.Position = [0 0 1024 768];
        h_.Color = [1 1 1];
        hold on
        sign_plot = sign_plot_{cond};
        color_plot = color_plot_{cond};
        plot(hours+cond*4+0.4*randn(1,length(hours)), features_(idx_colony, iter_fea), sign_plot,...
            'Color',color_plot, 'MarkerSize', 5, ...
            'MarkerFaceColor',color_plot);
        ttl_fig = ['<' names_feature{iter_fea} '>'];
        title(ttl_fig)
    end
    
end

%%
idx_fea = 9;

hour = 0;
cond = 1;

idx_chosen = find((hour_ == hour).*(cond_ == cond));

features = features_(idx_chosen,idx_fea);
[mean(features), std(features,0,1)]


    idx_colony = find(colony_ == colony);
    cond = cond_(idx_colony(1));

    [nothing, order_sort] = sort(hour_(idx_colony), 'ascend');
    idx_colony = idx_colony(order_sort);

for iter_fea = 1:size(features_,2)


    for colony = 1:max(colony_)

        hold on
        sign_plot = sign_plot_{cond};
        color_plot = color_plot_{cond};
        plot(hours+cond*4+0.4*randn(1,length(hours)), features_(idx_colony, iter_fea), sign_plot,...
            'Color',color_plot, 'MarkerSize', 5, ...
            'MarkerFaceColor',color_plot);

    end   
end
    