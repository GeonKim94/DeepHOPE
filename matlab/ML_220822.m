dir_feat = '/data02/gkim/stem_cell_jwshin/data/220822_feature';
cd(dir_feat)
dir_set = dir('0*');

features = [];
classes = [];
sets = [];

for iter_set = 1:length(dir_set)
    cd(dir_feat)
    cd(dir_set(iter_set).name)
    dir_cls = dir('0*');
    
    for iter_cls = 1:length(dir_cls)
        cd(dir_feat)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_data = dir('2*.mat');
        
        for iter_data = 1:length(dir_data)
            
            cd(dir_feat)
            cd(dir_set(iter_set).name)
            cd(dir_cls(iter_cls).name)
            
            load(dir_data(iter_data).name);
            features = [features; [nc_ratio ncont_bound smoothness]];
            classes = [classes; iter_cls];
            sets = [sets; iter_set];
        end
        
    end
end


features_train = features(find(sets == 1),:);
classes_train = classes(find(sets == 1),:);
features_test = features(find(sets == 2),:);
classes_test = classes(find(sets == 2),:);
%% simple plot
close all


figure(21), hold on
Xs1 = ones(sum(classes == 1),1)+0.05*randn(sum(classes == 1),1);
Ys1 = features(find(classes == 1), 1);
plot(Xs1, Ys1, 'k.');
errorbar(1.2, mean(Ys1), std(Ys1,0,1),'k');
plot([1.15 1.25], [mean(Ys1) mean(Ys1)], 'k');
Xs2 = 2*ones(sum(classes == 2),1)+0.05*randn(sum(classes == 2),1);
Ys2 = features(find(classes == 2), 1);
plot(Xs2, Ys2, 'k.');
errorbar(2.2, mean(Ys2), std(Ys2,0,1),'k');
plot([2.15 2.25], [mean(Ys2) mean(Ys2)], 'k');
Xs3 = 3*ones(sum(classes == 3),1)+0.05*randn(sum(classes == 3),1);
Ys3 = features(find(classes == 3), 1);
plot(Xs3, Ys3, 'k.');
errorbar(3.2, mean(Ys3), std(Ys3,0,1),'k');
plot([3.15 3.25], [mean(Ys3) mean(Ys3)], 'k');
xlim([0 4]);
title('Pseudo-NCRatio');

p12 = ranksum(Ys1,Ys2);
p23 = ranksum(Ys2,Ys3);
p31 = ranksum(Ys3,Ys1);


[h, p12] = ttest2(Ys1,Ys2);
[h, p23] = ttest2(Ys2,Ys3);
[h, p31] = ttest2(Ys3,Ys1);

[p12 p23 p31]


figure(22), hold on
Xs1 = ones(sum(classes == 1),1)+0.05*randn(sum(classes == 1),1);
Ys1 = features(find(classes == 1), 2);
plot(Xs1, Ys1, 'k.');
errorbar(1.2, mean(Ys1), std(Ys1,0,1),'k');
plot([1.15 1.25], [mean(Ys1) mean(Ys1)], 'k');
Xs2 = 2*ones(sum(classes == 2),1)+0.05*randn(sum(classes == 2),1);
Ys2 = features(find(classes == 2), 2);
plot(Xs2, Ys2, 'k.');
errorbar(2.2, mean(Ys2), std(Ys2,0,1),'k');
plot([2.15 2.25], [mean(Ys2) mean(Ys2)], 'k');
Xs3 = 3*ones(sum(classes == 3),1)+0.05*randn(sum(classes == 3),1);
Ys3 = features(find(classes == 3), 2);
plot(Xs3, Ys3, 'k.');
errorbar(3.2, mean(Ys3), std(Ys3,0,1),'k');
plot([3.15 3.25], [mean(Ys3) mean(Ys3)], 'k');
xlim([0 4]);
title('Boundary contrast');

p12 = ranksum(Ys1,Ys2);
p23 = ranksum(Ys2,Ys3);
p31 = ranksum(Ys3,Ys1);


[h, p12] = ttest2(Ys1,Ys2);
[h, p23] = ttest2(Ys2,Ys3);
[h, p31] = ttest2(Ys3,Ys1);

[p12 p23 p31]


figure(23), hold on
Xs1 = ones(sum(classes == 1),1)+0.05*randn(sum(classes == 1),1);
Ys1 = features(find(classes == 1), 3);
plot(Xs1, Ys1, 'k.');
errorbar(1.2, mean(Ys1), std(Ys1,0,1),'k');
plot([1.15 1.25], [mean(Ys1) mean(Ys1)], 'k');
Xs2 = 2*ones(sum(classes == 2),1)+0.05*randn(sum(classes == 2),1);
Ys2 = features(find(classes == 2), 3);
plot(Xs2, Ys2, 'k.');
errorbar(2.2, mean(Ys2), std(Ys2,0,1),'k');
plot([2.15 2.25], [mean(Ys2) mean(Ys2)], 'k');
Xs3 = 3*ones(sum(classes == 3),1)+0.05*randn(sum(classes == 3),1);
Ys3 = features(find(classes == 3), 3);
plot(Xs3, Ys3, 'k.');
errorbar(3.2, mean(Ys3), std(Ys3,0,1),'k');
plot([3.15 3.25], [mean(Ys3) mean(Ys3)], 'k');
xlim([0 4]);
title('Boundary smoothness');

p12 = ranksum(Ys1,Ys2);
p23 = ranksum(Ys2,Ys3);
p31 = ranksum(Ys3,Ys1);


[h, p12] = ttest2(Ys1,Ys2);
[h, p23] = ttest2(Ys2,Ys3);
[h, p31] = ttest2(Ys3,Ys1);

[p12 p23 p31]

%% normalize
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

numnei = 2;
mdl_knn = fitcknn(features_train,classes_train,'NumNeighbors',numnei,'Standardize',0);
    classes_test_ = predict(mdl_knn,features_test);
    
    confusion = zeros(3,3);
    for iter1 = 1:size(classes_test,1)
        confusion(classes_test(iter1), classes_test_(iter1)) = confusion(classes_test(iter1), classes_test_(iter1))+1;
    end

