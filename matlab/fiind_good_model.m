
%%  metric search in specific class sensitivity
clc
cls_include = [2 5 8];
dir_out = '/workspace01/gkim/stem_cell_jwshin/outs/23_SEC1H5_wider_v3_pick3h0_GM_germline_fishdeep10_b012_in_lr0.001/';

cd(dir_out)

list_epoch = dir('epoch*');
list_epoch = list_epoch(find([list_epoch.isdir]));


for iter_epoch = 1:length(list_epoch)

    cd(dir_out)
    dir_epoch = list_epoch(iter_epoch).name;
    cd(dir_epoch)
    load('result_valid.mat');

    test_wholesize = length(targets);
    scores = reshape(scores, [length(scores)/(test_wholesize), test_wholesize]);

    scores = scores';
    targets = targets';
    paths = paths';

    [val_max idx_max] = max(scores,[],2);

    recall_val = sum((idx_max == targets+1).*(ismember(targets, cls_include)))/sum(ismember(targets, cls_include));
    prec_val = sum((ismember(idx_max, cls_include+1)).*(ismember(targets, cls_include)))/sum(ismember(idx_max, cls_include+1));

    load('result_train.mat');

    test_wholesize = length(targets);
    scores = reshape(scores, [length(scores)/(test_wholesize), test_wholesize]);

    scores = scores';
    targets = targets';
    paths = paths';

    [val_max idx_max] = max(scores,[],2);

    recall_train = sum((idx_max == targets+1).*(ismember(targets, cls_include)))/sum(ismember(targets, cls_include));
    prec_train = sum((ismember(idx_max, cls_include+1)).*(ismember(targets, cls_include)))/sum(ismember(idx_max, cls_include+1));

    sprintf('%s, recall & precision: %0.4f & %0.4f (train) %0.4f & %0.4f (val)', dir_epoch(1:12), recall_train, prec_train, recall_val,prec_val)
end

%%  metric search in specific class sensitivity (confusion between exp are okay)
clc
cls_include = [0 3 6];%[2 5 8];%[0 3 6];


dir_out = '/workspace01/gkim/stem_cell_jwshin/outs/23_SEC1H5_wider_v3_pick3h0_GM_germline_fishdeep10_b012_in_lr0.001/';

cd(dir_out)

list_epoch = dir('epoch*');
list_epoch = list_epoch(find([list_epoch.isdir]));


for iter_epoch = 1:length(list_epoch)

    cd(dir_out)
    dir_epoch = list_epoch(iter_epoch).name;
    cd(dir_epoch)
    load('result_valid.mat');

    test_wholesize = length(targets);
    scores = reshape(scores, [length(scores)/(test_wholesize), test_wholesize]);

    scores = scores';
    targets = targets';
    paths = paths';
    scores = scores(ismember(targets,[cls_include 9]),:);
    paths = paths(ismember(targets,[cls_include 9]));
    targets = targets(ismember(targets,[cls_include 9]));
    targets(targets<9) = 0;

    [val_max idx_max] = max(scores,[],2);
    idx_max(idx_max<10) = 1;

    recall_val = sum((idx_max == targets+1).*(ismember(targets, 0)))/sum(ismember(targets, 0));
    prec_val = sum((ismember(idx_max, 0+1)).*(ismember(targets, 0)))/sum(ismember(idx_max, 0+1));

    load('result_train.mat');

    test_wholesize = length(targets);
    scores = reshape(scores, [length(scores)/(test_wholesize), test_wholesize]);

    
    scores = scores';
    targets = targets';
    paths = paths';
    scores = scores(ismember(targets,[cls_include 9]),:);
    paths = paths(ismember(targets,[cls_include 9]));
    targets = targets(ismember(targets,[cls_include 9]));
    targets(targets<9) = 0;

    [val_max idx_max] = max(scores,[],2);
    idx_max(idx_max<10) = 1;

    recall_train = sum((idx_max == targets+1).*(ismember(targets, 0)))/sum(ismember(targets, 0));
    prec_train = sum((ismember(idx_max, 0+1)).*(ismember(targets, 0)))/sum(ismember(idx_max, 0+1));

    sprintf('%s, recall & precision: %0.4f & %0.4f (train) %0.4f & %0.4f (val)', dir_epoch(1:12), recall_train, prec_train, recall_val,prec_val)
end

