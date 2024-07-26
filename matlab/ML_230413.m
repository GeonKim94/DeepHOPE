clear
clc
close all
dir_data = dir('/data02/gkim/stem_cell_jwshin/data/23*feature_v2');


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

cond_ = [];
line_ = [];

for iter_data = 1:length(dir_data)
    cd('/data02/gkim/stem_cell_jwshin/data')
    cd(dir_data(iter_data).name)
    dir_set = dir('0*');
    
    for iter_set = 1:length(dir_set)
    cd('/data02/gkim/stem_cell_jwshin/data')
        cd(dir_data(iter_data).name)
        cd(dir_set(iter_set).name)
        dir_line = dir('*_*');

        for iter_line = 1:length(dir_line)
    cd('/data02/gkim/stem_cell_jwshin/data')
            cd(dir_data(iter_data).name)
            cd(dir_set(iter_set).name)
            cd(dir_line(iter_line).name)

            dir_feat = dir('*hr*.mat');
            for iter_feat = 1:length(dir_feat)
                
                fname_feat = dir_feat(iter_feat).name;
                
                if contains(dir_line(iter_line).name, 'jipsc')
                    line = 1;
                elseif contains(dir_line(iter_line).name, 'Jax')
                    line = 1;
                elseif contains(dir_line(iter_line).name, 'GM')
                    line = 2;
                elseif contains(dir_line(iter_line).name, 'h9')
                    line = 3;
                else
                    error('no line info found on the directory')
                end

                if contains(fname_feat, 'hr00')
                    cond = 1;
                elseif contains(fname_feat, '.A')
                    if contains(fname_feat, 'hr24')
                        cond = 2;
                    elseif contains(fname_feat, 'hr48')
                        cond = 3;   
                    else
                        error('no cond info found on the directory')
                    end

                elseif contains(fname_feat, '.B')
                    if contains(fname_feat, 'hr24')
                        cond = 4;
                    elseif contains(fname_feat, 'hr48')
                        cond = 5;
                    else
                        error('no cond info found on the directory')
                    end
                else
                    error('no cond info found on the directory');
                end
                
                cond_ = [cond_; cond];
                line_ = [line_; line];
                
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

            end

        end
        
    end

end

ratio_asp_ = thick_mean_./(area_col_).^0.5;
pointiness_ = thick_peak_./(area_col_).^0.5;

ratio_nuc_ = area_nuc_./area_col_;
ratio_lip_ = area_lip_./area_col_;
ratio_gap_ = area_gap_./area_col_;

spread_thick_norm_ = spread_thick_./(area_col_).^0.5;

features_ = [smoothness_, ncont_bound_, area_col_, area_lip_,area_nuc_,...
    area_gap_,...
    ratio_nuc_,ratio_lip_,ratio_gap_, spread_thick_norm_...
    num_lip_, thick_mean_, thick_std_, thick_peak_,...
    center_disp_, spread_thick_, skew_thick_, kurt_thick_,...
    RI_mean_, RI_std_, ratio_asp_, pointiness_];

names_feature = {'smoothness', 'boundary contrast', 'colony area', 'LD area','nucleus area',...
    'gap area','nucleus ratio', 'LD ratio', 'gap ratio','normalized thickness spread'...
    'LD number', 'thickness mean', 'thickness std', 'thickness peak',...
    'center displacment', 'thickness spread', 'thickness skewness', 'thickness kurtosis',...
    'RI mean', 'RI std', 'aspect ratio', 'pointiness'};



%% remove outliers

idxs_outlier_ = [];
for cond = 1:max(cond_)
    for line = 1:max(line_)
        for iter_fea = 1:size(features_,2)
                idxs_data = find((cond_ == cond).*(line_ == line));
                if length(idxs_data) == 0
                    continue
                end
                
                y = features_(idxs_data,iter_fea);
                
                idxs_outlier = idxs_data([find(y > mean(y)+4*std(y,0,1));
                    find(y < mean(y)-4*std(y,0,1))]);
                
                idxs_outlier_ = [idxs_outlier_; idxs_outlier];
        end

    end
end
idxs_outlier_ = unique(idxs_outlier_);
features_(idxs_outlier_,:) = [];
cond_(idxs_outlier_,:) = [];
line_(idxs_outlier_,:) = [];
%% plot features
close all
for iter_fea = 1:size(features_,2)
    

    h_ = figure(iter_fea);
    h_.Position = [0 0 1024 768];
    h_.Color = [1 1 1];

    hold on
    y_ = {};
    
    for cond = 1:max(cond_)
        for line = 1:max(line_)
            c3 = zeros(1,3);
            c3(line) = 1;
            idxs_data = find((cond_ == cond).*(line_ == line));
            if length(idxs_data) == 0
                continue
            end
            
            x0 = (max(line_)+2)*(cond-1)+line;
            x = x0 + 0.05*randn(length(idxs_data),1);
            y = features_(idxs_data,iter_fea);

            y_mean = mean(y);
            y_std = std(y,0,1);
            errorbar( x0+0.3, y_mean, y_std, 'k', "CapSize", 10)
            plot([x0+0.1 x0+0.5], [y_mean y_mean], 'k');
            plot(x,y,'.',"MarkerEdgeColor", c3, "MarkerSize", 8)

            y_{end+1} = y;

        end
    end

    %p12 = ranksum(y_(:,1),y_(:,2));

    ttl_fig = ['<' names_feature{iter_fea} '>'];%;,...
%         sprintf(' - p = %0.3f, %0.3f, %0.3f', p12, p23 ,p13)];
    xlim([0.5 (max(cond_))*(max(line_)+2) + 0.5])
    title(ttl_fig)

    %saveas(gcf, [dir_save, '/', names_feature{iter_fea},'.fig'])
    hold off

end