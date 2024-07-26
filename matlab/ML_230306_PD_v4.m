clear
clc
close all
dir_src = ('/data02/gkim/stem_cell_jwshin/data/230306_feature_v4/00_train');


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

RI_mean_nolip_ = [];
RI_mean_lip_ = [];
vol_colony_ = [];
vol_lipid_ = [];

cond_ = [];
hour_ = [];
colony_ = [];

hours = 0;%0:1:24;
for hour = hours
    cd(dir_src)
    dir_line = dir('*_*');
    for iter_line = 1:length(dir_line)
            
        cd(dir_src)
        cd(dir_line(iter_line).name)
            dir_feat = dir(sprintf('*.mat',hour));
            for iter_feat = 1:length(dir_feat)
                
                fname_feat = dir_feat(iter_feat).name;

                if contains(dir_line(iter_line).name, 'A19')
                    cond = 1;
                elseif contains(dir_line(iter_line).name, 'A20')
                    cond = 2; 
                elseif contains(dir_line(iter_line).name, 'A2')
                    cond = 3;
                elseif contains(dir_line(iter_line).name, 'A12')
                    cond = 4;
                else
                    continue
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

                RI_mean_nolip_ = [RI_mean_nolip_;RI_mean_nolip];
                RI_mean_lip_ = [RI_mean_lip_;RI_mean_lip];
                vol_colony_ = [vol_colony_;vol_colony];
                vol_lipid_ = [vol_lipid_;vol_lipid];
            end
    end
end

ratio_asp_ = thick_mean_./(area_col_).^0.5;
pointiness_ = thick_peak_./(area_col_).^0.5;

ratio_nuc_ = area_nuc_./area_col_;
ratio_lip_ = area_lip_./area_col_;
ratio_gap_ = area_gap_./area_col_;
ratio_ax_ = minax_./majax_;
idx_inv = find(ratio_ax_>1);
fea_rep = minax_(idx_inv);
minax(idx_inv) = majax_(idx_inv);
majax_(idx_inv) = fea_rep;
ratio_ax_(idx_inv) = 1./ratio_ax_(idx_inv);

spread_thick_norm_ = spread_thick_./(area_col_).^0.5;

features_ = [smoothness_, ncont_bound_, area_col_, area_lip_,area_nuc_,...
    area_gap_,...
    ratio_nuc_,ratio_lip_,ratio_gap_, spread_thick_norm_...
    num_lip_, thick_mean_, thick_std_, thick_peak_,...
    center_disp_, spread_thick_, skew_thick_, kurt_thick_,...
    RI_mean_, RI_std_, ratio_asp_, pointiness_, majax_, minax_, eccen_, circu_, ratio_ax_,...
    (RI_mean_nolip_-1.337)/0.0018, RI_mean_lip_  vol_colony_, vol_lipid_, vol_lipid_./vol_colony_];
%RI_mean_nolip_, RI_mean_lip_, vol_colony_, vol_lipid_, vol_lipid_./vol_colony_];

names_feature = {'smoothness', 'boundary contrast', 'colony area (μm^2)', 'LD area (μm^2)','nucleus area (μm^2)',...
    'gap area (μm^2)','nucleus ratio', 'LD ratio', 'gap ratio','normalized thickness spread (μm)'...
    'LD number', 'thickness mean (μm)', 'thickness std (μm)', 'thickness peak (μm)','center displacment (μm)',...
    'thickness spread (μm)', 'thickness skewness', 'thickness kurtosis', 'RI mean', 'RI std',...
    'aspect ratio', 'pointiness','major axis (μm)', 'minor axis (μm)', 'eccentricity',...
    'circularity', 'axis ratio', ...
    'dry mass density (g per dL)', 'LD RI mean', 'colony volume (fL)', 'LD volume (fL)',...
    'LD volume ratio'};
%'non-LD RI mean', 'LD RI mean', 'colony volume (fL)', 'LD volume (fL)',...


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

sign_plot_ = {'o', 'o', 'o'};
color_plot_ = {[0 0 1], [0 0.25 0.75], [1 0 0], [0.75 0.25 0]};
sign_text = 1;
for cond = unique(cond_)'
    idx_data = find(cond_ == cond);
    for iter_fea = 1:size(features_,2)
        x0 = floor(cond/10)*10 + (cond-floor(cond/10)*10)*2;
        x = x0 + 0.1*randn(1, length(idx_data));
        h_ = figure(iter_fea);
        h_.Position = [0 0 1024 768];
        h_.Color = [1 1 1];
        hold on
        sign_plot = sign_plot_{floor(cond/10)+1};
        color_plot = color_plot_{floor(cond/10)+1};
        y = features_(idx_data, iter_fea);
        plot(x, y, sign_plot,...
            'Color',color_plot, 'MarkerSize', 5, ...
            'MarkerFaceColor',color_plot);

        errorbar(x0+0.5, mean(y), std(y,0,1) ,'k', 'CapSize', 10);
        plot([x0+0 x0+1.0],[mean(y) mean(y)] ,'k');
        
        text(x0, mean(y)+sign_text*std(y,0,1)*1.5, ...
            sprintf('%.3g ± %.3g', mean(y), std(y,0,1)),...
            'HorizontalAlignment', 'center')

        ttl_fig = names_feature{iter_fea};
        title(ttl_fig)
    end

    sign_text = sign_text*-1;
    
end


cd '/data02/gkim/stem_cell_jwshin/data/230502_feature_v4'
for iter_fea = 1:size(features_,2)
    h_ = figure(iter_fea);
    saveas(h_, [names_feature{iter_fea}, '.fig'])
end

%% bar plot
close all 

color_plot_ = {[0 0 1], [0 0.25 0.75], [1 0 0], [0.75 0.25 0]};
sign_text = 1;
for cond = unique(cond_)'
    idx_data = find(cond_ == cond);
    for iter_fea = 1:size(features_,2)
        x0 = floor(cond/10)*2 + (cond-floor(cond/10)*10)*10;
        x = x0 + 0.1*randn(1, length(idx_data));
        h_ = figure(iter_fea);
        h_.Position = [0 0 1024 768];
        h_.Color = [1 1 1];
        hold on
        color_plot = color_plot_{cond};
        y = features_(idx_data, iter_fea);
%         plot(x, y, sign_plot,...
%             'Color',color_plot, 'MarkerSize', 5, ...
%             'MarkerFaceColor',color_plot);

        bar(x0,mean(y),'FaceColor',color_plot,'BarWidth',1);
        errorbar(x0, mean(y), std(y,0,1) ,'k', 'CapSize', 10);
%         errorbar(x0+0.5, mean(y), std(y,0,1) ,'k', 'CapSize', 10);
%         plot([x0+0 x0+1.0],[mean(y) mean(y)] ,'k');
        
        text(x0, mean(y)+sign_text*std(y,0,1)*1.5, ...
            sprintf('%.3g ± %.3g', mean(y), std(y,0,1)),...
            'HorizontalAlignment', 'center')

        ttl_fig = names_feature{iter_fea};
        title(ttl_fig)
    end

    sign_text = sign_text*-1;
    
end

cd '/data02/gkim/stem_cell_jwshin/data/230502_feature_v4'
for iter_fea = 1:size(features_,2)
    h_ = figure(iter_fea);
    saveas(h_, ['01_bar_' names_feature{iter_fea}, '.fig'])
end

%% plot features with each other
names_feature = {'smoothness', 'boundary contrast', 'colony area (μm^2)', 'LD area (μm^2)','nucleus area (μm^2)',...
    'gap area (μm^2)','nucleus ratio', 'LD ratio', 'gap ratio','normalized thickness spread (μm)'...
    'LD number', 'thickness mean (μm)', 'thickness std (μm)', 'thickness peak (μm)','center displacment (μm)',...
    'thickness spread (μm)', 'thickness skewness', 'thickness kurtosis', 'RI mean', 'RI std',...
    'aspect ratio', 'pointiness','major axis (μm)', 'minor axis (μm)', 'eccentricity',...
    'circularity', 'axis ratio', ...
    'dry mass density (g per dL)', 'LD RI mean', 'colony volume (fL)', 'LD volume (fL)',...
    'LD volume ratio'};
%'non-LD RI mean', 'LD RI mean', 'colony volume (fL)', 'LD volume (fL)',...names_feature = {'smoothness', 'boundary contrast', 'colony area (μm^2)', 'LD area (μm^2)','nucleus area (μm^2)',...
idxs_feat_chosen = [12 26 28 30];%[1 2 8 9 12 13 14 19 20 22 26];

close all

sign_plot_ = {'o', 's', '^'};
color_plot_ = {[0 0.66 1], [0.5 0.75 0.5], [1 0.5 0]};
sign_text = 1;
for cond = unique(cond_)'
    idx_data = find(cond_ == cond);
    for iter_fea1 = idxs_feat_chosen
        for iter_fea2 = idxs_feat_chosen(idxs_feat_chosen> iter_fea1)
            
            y1 = features_(idx_data, iter_fea1);
            y2 = features_(idx_data, iter_fea2);

            h_ = figure(iter_fea1*100+iter_fea2);
            h_.Position = [0 0 1024 768];
            h_.Color = [1 1 1];
            hold on
            sign_plot = sign_plot_{cond-floor(cond/10)*10};
            color_plot = color_plot_{floor(cond/10)+1};
            
            plt = plot(y1, y2, sign_plot,...
                'Color',color_plot, 'MarkerSize', 10, ...
                'MarkerEdgeColor',color_plot, 'Linewidth', 2);
            
            
            ttl_fig = [names_feature{iter_fea1} ' - ' names_feature{iter_fea2}];
            title(ttl_fig)
            axis square
        end
    end

    sign_text = sign_text*-1;
    
end


cd '/data02/gkim/stem_cell_jwshin/data/230502_feature_v4'
for iter_fea1 = idxs_feat_chosen
    for iter_fea2 = idxs_feat_chosen(idxs_feat_chosen> iter_fea1)
        h_ = figure(iter_fea1*100+iter_fea2);
        alpha(0.5)
        axis off
        try
            saveas(h_, ['COPLOT ' names_feature{iter_fea1} ' - ' names_feature{iter_fea2}, '.fig'])
        end
        axis on
    end
end


%% ML features
classes_ = floor(cond_/10)+1;
idxs_test = [find(cond_ == 1); find(cond_ == 11); find(cond_ == 21)];

idxs_feat_chosen = 1:27;%[1 2 8 9 12 13 14 19 20 22 26];
features_train = features_(:, idxs_feat_chosen);
features_train(idxs_test,:) = [];
classes_train = classes_(:,1);
classes_train(idxs_test,:) = [];

features_test = features_(idxs_test,idxs_feat_chosen);
classes_test= classes_(idxs_test,1);


%% KNN
    accs_knn = [];
    
    for numnei = 1:50
        mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',1);
        classes_test_ = predict(mdl_knn,features_test);
        accs_knn = [accs_knn; sum((classes_test_-classes_test)==0)/length(classes_test_)];
    end
    
    numnei = find(accs_knn == max(accs_knn));
    numnei = numnei(1);
    mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',1);
        classes_test_ = predict(mdl_knn,features_test);
        
        confusion_knn = zeros(3,3);
        for iter1 = 1:size(classes_test,1)
            confusion_knn(classes_test(iter1), classes_test_(iter1)) = confusion_knn(classes_test(iter1), classes_test_(iter1))+1;
        end
    
    figure(21), imagesc(confusion_knn./repmat(sum(confusion_knn,2),[1 3]), [0 1]), axis square, colormap(gray)
    
    %% SVM - gaussian
    t = templateSVM('Standardize',true,'KernelFunction','gaussian');
    mdl_svm = fitcecoc(features_train, classes_train,'Coding','onevsall', 'Learners', t);
    [class_test_pred, scores] = predict(mdl_svm, features_test, 'Verbose', 2);
    
    confusion_svm_gauss = zeros(3,3);
    for idx_data = 1:size(class_test_pred,1)
        pred_temp = find(scores(idx_data,:) == max(scores(idx_data,:)));
        confusion_svm_gauss(classes_test(idx_data),pred_temp) = confusion_svm_gauss(classes_test(idx_data),pred_temp)+1;
    
    end
    
    
    figure(31), imagesc(confusion_svm_gauss./repmat(sum(confusion_svm_gauss,2),[1 3]), [0 1]), axis square, colormap(gray)
    
    %% SVM - linear
    t = templateSVM('Standardize',true,'KernelFunction','linear');
    mdl_svm = fitcecoc(features_train, classes_train,'Coding','onevsall', 'Learners', t);
    [class_test_pred, scores] = predict(mdl_svm, features_test, 'Verbose', 2);
    
    confusion_svm_linear = zeros(3,3);
    for idx_data = 1:size(class_test_pred,1)
        pred_temp = find(scores(idx_data,:) == max(scores(idx_data,:)));
        confusion_svm_linear(classes_test(idx_data),pred_temp) = confusion_svm_linear(classes_test(idx_data),pred_temp)+1;
    
    end
    
    figure(41), imagesc(confusion_svm_linear./repmat(sum(confusion_svm_linear,2),[1 3]), [0 1]), axis square, colormap(gray)

    max(accs_knn)
    trace(confusion_svm_gauss)/sum(sum(confusion_svm_gauss))
    trace(confusion_svm_linear)/sum(sum(confusion_svm_linear))


%% t-sne
close all
classes_ = floor(cond_/10)+1;
idxs_test = [find(cond_ == 1); find(cond_ == 11); find(cond_ == 21)];

idxs_feat_chosen = [1 2 8 9 12 13 14 19 20 22 26];
features_train = features_(:, idxs_feat_chosen);
features_train(idxs_test,:) = [];
classes_train = classes_(:,1);
classes_train(idxs_test,:) = [];

features_test = features_(idxs_test,idxs_feat_chosen);
classes_test= classes_(idxs_test,1);


%%
perp = 50;
stan = true;
lr = 100;
features_tsne_ = tsne(features_, 'perplexity', perp, 'Standardize', stan, 'LearnRate', lr);

save('tsne.mat', 'features_tsne_', 'perp', 'stan', 'lr')


%% plot according to line

close all
sign_plot_ = {'o', 'o', 'o'};
%color_plot_ = {[0 0.66 1], [0.5 0.75 0.5], [1 0.5 0]};
color_plot_ = {[0.33 0.33 1],[0 0.33 0.66],[0 0.66 1],...
    [0 1.0 0],[0.5 0.75 0.5],...
    [0.75 0.25 0],[1 0 0],[1 0.5 0]};
sign_text = 1;
for cond = unique(cond_)'
    idx_data = find(cond_ == cond);
            
            y1 = features_tsne_(idx_data, 1);
            y2 = features_tsne_(idx_data, 2);

            h_ = figure(9999);
            h_.Position = [0 0 1024 768];
            h_.Color = [1 1 1];
            hold on
            sign_plot = sign_plot_{floor(cond/10)+1};
            color_plot = color_plot_{find(unique(cond_) == cond)};
            
            plot(y1, y2, sign_plot,...
                'Color',color_plot, 'MarkerSize', 5, ...
                'MarkerFaceColor',color_plot);
            
            
            ttl_fig = 't-sne using chosen features';
            title(ttl_fig)
            axis square

    sign_text = sign_text*-1;
    
end

saveas(gcf, 't-sne.fig')
%% plot according to specific features


for iter_feat = 1:length(idxs_feat_chosen)
close all
idx_feat_plot = idxs_feat_chosen(iter_feat);

sign_plot_ = {'o', 'o', 'o'};
sign_text = 1;

features_plot_rsc_ = features_(:,idx_feat_plot);
features_plot_max_ = mean(features_plot_rsc_,1)-1.5*std(features_plot_rsc_,0,1);
features_plot_min_ = mean(features_plot_rsc_,1)+1.5*std(features_plot_rsc_,0,1);
features_plot_rsc_ = (features_plot_rsc_-(features_plot_min_))./(features_plot_max_-features_plot_min_);
features_plot_rsc_(features_plot_rsc_<0) = 0;
features_plot_rsc_(features_plot_rsc_>1) = 1;

cmap = spring;%flipud(parula);

for idx_data = 1:size(features_tsne_,1)
            
            y1 = features_tsne_(idx_data, 1);
            y2 = features_tsne_(idx_data, 2);

            h_ = figure(10000+iter_feat);
            h_.Position = [0 0 1024 768];
            h_.Color = [1 1 1];
            hold on
            sign_plot = sign_plot_{floor(cond/10)+1};
            idx_color = round(features_plot_rsc_(idx_data)*255)+1;
            color_plot = cmap(idx_color,:);
            
            plot(y1, y2, sign_plot,...
                'Color',color_plot, 'MarkerSize', 5, ...
                'MarkerFaceColor',color_plot);
            
            
            ttl_fig = sprintf('t-sne (color: %s)', names_feature{idx_feat_plot});
            title(ttl_fig)
            axis square

    sign_text = sign_text*-1;
    
end

saveas(gcf, sprintf('t-sne color %s.fig',names_feature{idx_feat_plot}))

end