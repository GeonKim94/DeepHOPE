
%% read features
close all
clear
clc
dir_save = '/data02/gkim/stem_cell_jwshin/data/230220_feature';

smoothness_ = [];
ncont_bound_ = [];
area_col_ = [];
area_nuc_ = [];
area_lip_ = [];
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

idx_set_ = [];
idx_cls_ = [];

cd(dir_save)
dir_set = dir('0*');
for iter_set = 1:length(dir_set)
    cd(dir_save);
    cd(dir_set(iter_set).name);
    dir_cls = dir('0*');
    
    for iter_cls = 1:length(dir_cls)
        cd(dir_save)
        cd(dir_set(iter_set).name);
        cd(dir_cls(iter_cls).name);
        
        dir_fea = dir('2*.mat');
        
        for iter_fea = 1:length(dir_fea)
            cd(dir_save)
            cd(dir_set(iter_set).name);
            cd(dir_cls(iter_cls).name);
            load(dir_fea(iter_fea).name);

            smoothness_ = [smoothness_; smoothness];
            ncont_bound_ = [ncont_bound_; ncont_bound];
            area_col_ = [area_col_; area_col];
            area_nuc_ = [area_nuc_; area_nuc];
            area_lip_ = [area_lip_; area_lip];
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

            idx_set_ = [idx_set_; iter_set];
            idx_cls_ = [idx_cls_; iter_cls];

        end
    end
end

ratio_asp_ = thick_mean_./(area_col_).^0.5;
pointiness_ = thick_peak_./(area_col_).^0.5;

ratio_nuc_ = area_nuc_./area_col_;
ratio_lip_ = area_lip_./area_col_;

spread_thick_norm_ = spread_thick_./(area_col_).^0.5;

features_ = [smoothness_, ncont_bound_, area_col_, area_lip_,area_nuc_,...
    ratio_nuc_,ratio_lip_, spread_thick_norm_...
    num_lip_, thick_mean_, thick_std_, thick_peak_,...
    center_disp_, spread_thick_, skew_thick_, kurt_thick_,...
    RI_mean_, RI_std_, ratio_asp_, pointiness_];

names_feature = {'smoothness', 'boundary contrast', 'colony area', 'LD area','nucleus area',...
    'nucleus ratio', 'LD ratio','normalized thickness spread'...
    'LD number', 'thickness mean', 'thickness std', 'thickness peak',...
    'center displacment', 'thickness spread', 'thickness skewness', 'thickness kurtosis',...
    'RI mean', 'RI std', 'aspect ratio', 'pointiness'};

accs_knn_addfeat = [];
accs_svm_gauss_addfeat = [];
accs_svm_linear_addfeat = [];
idxs_addfeat = [0 3 4 5 7:20];
for idx_addfeat = idxs_addfeat
    features_ML_ = features_(:, [1 2 6]);
    try
        features_ML_ = [features_ML_, features_(:, idx_addfeat)];
    end
    % [area_col_, area_lip_, num_lip_, ratio_lip_, ratio_nuc_,...
    %     ncont_bound_, smoothness_, thick_mean_, thick_std_, thick_peak_,...
    %     spread_thick_norm_, skew_thick_, kurt_thick_];
    
    %% ml
    % set teset
    features_train = features_ML_(find(idx_set_ ~= 3),:);
    classes_train = idx_cls_(find(idx_set_ ~= 3),:);
    features_test = features_ML_(find(idx_set_ == 3),:);
    classes_test = idx_cls_(find(idx_set_ == 3),:);
    
    %normalize
    features_train_mean = mean(features_train,1);
    features_train_std = std(features_train,0,1);
    
    features_train = (features_train-features_train_mean)./features_train_std;
    features_test = (features_test-features_train_mean)./features_train_std;
    
    
    
    %% KNN
    accs_knn = [];
    
    for numnei = 1:20
        mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',0);
        classes_test_ = predict(mdl_knn,features_test);
        accs_knn = [accs_knn; sum((classes_test_-classes_test)==0)/length(classes_test_)];
    end
    
    numnei = find(accs_knn == max(accs_knn));
    numnei = numnei(1);
    mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',0);
        classes_test_ = predict(mdl_knn,features_test);
        
        confusion_knn = zeros(3,3);
        for iter1 = 1:size(classes_test,1)
            confusion_knn(classes_test(iter1), classes_test_(iter1)) = confusion_knn(classes_test(iter1), classes_test_(iter1))+1;
        end
    
%     figure(21), imagesc(confusion_knn./repmat(sum(confusion_knn,2),[1 3]), [0 1]), axis square, colormap(gray)
    
    %% SVM - gaussian
    t = templateSVM('Standardize',true,'KernelFunction','gaussian');
    mdl_svm = fitcecoc(features_train, classes_train,'Coding','onevsall', 'Learners', t);
    [class_test_pred, scores] = predict(mdl_svm, features_test, 'Verbose', 2);
    
    confusion_svm_gauss = zeros(3,3);
    for idx_data = 1:size(class_test_pred,1)
        pred_temp = find(scores(idx_data,:) == max(scores(idx_data,:)));
        confusion_svm_gauss(classes_test(idx_data),pred_temp) = confusion_svm_gauss(classes_test(idx_data),pred_temp)+1;
    
    end
    
    
%     figure(31), imagesc(confusion_svm_gauss./repmat(sum(confusion_svm_gauss,2),[1 3]), [0 1]), axis square, colormap(gray)
    
    %% SVM - linear
    t = templateSVM('Standardize',true,'KernelFunction','linear');
    mdl_svm = fitcecoc(features_train, classes_train,'Coding','onevsall', 'Learners', t);
    [class_test_pred, scores] = predict(mdl_svm, features_test, 'Verbose', 2);
    
    confusion_svm_linear = zeros(3,3);
    for idx_data = 1:size(class_test_pred,1)
        pred_temp = find(scores(idx_data,:) == max(scores(idx_data,:)));
        confusion_svm_linear(classes_test(idx_data),pred_temp) = confusion_svm_linear(classes_test(idx_data),pred_temp)+1;
    
    end
    
%     figure(41), imagesc(confusion_svm_linear./repmat(sum(confusion_svm_linear,2),[1 3]), [0 1]), axis square, colormap(gray)


    accs_knn_addfeat = [accs_knn_addfeat accs_knn];
    accs_svm_gauss_addfeat = [accs_svm_gauss_addfeat trace(confusion_svm_gauss)/sum(sum(confusion_svm_gauss))];
    accs_svm_linear_addfeat = [accs_svm_linear_addfeat trace(confusion_svm_linear)/sum(sum(confusion_svm_linear))];

end
accs_ML_avg = mean([accs_svm_linear_addfeat;accs_svm_gauss_addfeat;mean(accs_knn_addfeat,1)],1);
figure(101), imagesc(accs_knn_addfeat, [0 1]), axis image;
figure(102),plot(mean(accs_knn_addfeat,1),'r')
hold on, plot(accs_svm_gauss_addfeat,'g')
plot(accs_svm_linear_addfeat,'b'), 
plot(accs_ML_avg,'k'), 
hold off, ylim([0.2 0.6]);

%% manual feature picking
idxs_addfeat_pick = idxs_addfeat([7,3,14, 10, 2, 13, 12]);


%% ML with picked features
features_ML_ = features_(:, [1 2 6 idxs_addfeat_pick(1:4)]);
    % [area_col_, area_lip_, num_lip_, ratio_lip_, ratio_nuc_,...
    %     ncont_bound_, smoothness_, thick_mean_, thick_std_, thick_peak_,...
    %     spread_thick_norm_, skew_thick_, kurt_thick_];
    
    %% ml
    % set teset
    features_train = features_ML_(find(idx_set_ ~= 3),:);
    classes_train = idx_cls_(find(idx_set_ ~= 3),:);
    features_test = features_ML_(find(idx_set_ == 3),:);
    classes_test = idx_cls_(find(idx_set_ == 3),:);
    
    %normalize
    features_train_mean = mean(features_train,1);
    features_train_std = std(features_train,0,1);
    
    features_train = (features_train-features_train_mean)./features_train_std;
    features_test = (features_test-features_train_mean)./features_train_std;
    
    
    
    %% KNN
    accs_knn = [];
    
    for numnei = 1:20
        mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',0);
        classes_test_ = predict(mdl_knn,features_test);
        accs_knn = [accs_knn; sum((classes_test_-classes_test)==0)/length(classes_test_)];
    end
    
    numnei = find(accs_knn == max(accs_knn));
    numnei = numnei(1);
    mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',0);
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

%% plot features


for iter_fea = 1:size(features_,2)
    
    figure(iter_fea)
    hold on
    y_ = {};
    for iter_cls = 1:length(dir_cls)
        x = iter_cls + 0.05*randn(sum(idx_cls_==iter_cls),1);
        y = features_(idx_cls_==iter_cls,iter_fea);
            
        y_mean = mean(y);
        y_std = std(y,0,1);
        errorbar(iter_cls+0.25, y_mean, y_std,'k')
        plot([iter_cls+0.15 iter_cls+0.35], [y_mean y_mean] ,'k');

        plot(x,y,'k.')
        
        y_{end+1} = y;
    end

    p12 = ranksum(y_{1},y_{2});
    p23 = ranksum(y_{2},y_{3});
    p13 = ranksum(y_{1},y_{3});
    
    ttl_fig = ['<' names_feature{iter_fea} '>',...
        sprintf(' - p = %0.3f, %0.3f, %0.3f', p12, p23 ,p13)];
    xlim([0.5 length(dir_cls)+0.5])
    title(ttl_fig)

    saveas(gcf, [dir_save, '/', names_feature{iter_fea},'.fig'])
    hold off

end