clear
close all
clc
maindir = '/data02/gkim/stem_cell_jwshin/outs/230209_3D_b016_lr0.001/epoch[00133]_train[0.9605]_valid[0.9082]_test[0.6528]';
cd(maindir);
test_wholesize = 2229;
valid_wholesize = 1078;
train_wholesize = 9601;


%% get valid result

cd(maindir)
idxs_del = [];
% for iter1 = 1:length(epnames)
%     epname = epnames{iter1};
%     epoch = str2num(epname(7:11));
%     valacc = str2num(epname(34:39));
%     if epoch < 0
%         idxs_del = [idxs_del iter1];
%     elseif valacc < 0.87
%         idxs_del = [idxs_del iter1];
%     end
% end

confusions_valid = [];
scores_valid = [];
  
cd(maindir)
load('result_valid.mat');
scores_valid = cat(3,scores_valid,reshape(scores, [length(scores)/(valid_wholesize),valid_wholesize]));

targets_valid = targets+1;
scores = reshape(scores, [length(scores)/valid_wholesize,valid_wholesize]);
pred = zeros(size(targets_valid));

for iter2 = 1:size(scores,2)
    pred(iter2) = find(scores(:,iter2)==max(scores(:,iter2)));
end

confusion = zeros(size(scores,1),size(scores,1));
for iter2 = 1:length(targets_valid)
    confusion(targets_valid(iter2), pred(iter2)) = confusion(targets_valid(iter2), pred(iter2))+1;
end

trace(confusion)/sum(confusion(:));
    confusions_valid = cat(3,confusions_valid, confusion);

scores_valid_avg = mean(scores_valid,3);
pred = zeros(size(targets_valid));
for iter2 = 1:size(scores,2)
    pred(iter2) = find(scores_valid_avg(:,iter2)==max(scores_valid_avg(:,iter2)));
end
confusion_valid_avg = zeros(size(scores,1),size(scores,1));
for iter2 = 1:length(targets_valid)
    confusion_valid_avg(targets_valid(iter2), pred(iter2)) = confusion_valid_avg(targets_valid(iter2), pred(iter2))+1;
end
trace(confusion_valid_avg)/sum(confusion_valid_avg(:))
%% get test result
confusions_test = [];
scores_test = [];
  
cd(maindir)
load('result_test.mat');
scores_test = cat(3,scores_test,reshape(scores, [length(scores)/(test_wholesize),test_wholesize]));

targets_test = targets+1;

scores = reshape(scores, [length(scores)/test_wholesize,test_wholesize]);
pred = zeros(size(targets_test));

for iter2 = 1:size(scores,2)
    pred(iter2) = find(scores(:,iter2)==max(scores(:,iter2)));
end

confusion = zeros(size(scores,1),size(scores,1));
for iter2 = 1:length(targets_test)
    confusion(targets_test(iter2), pred(iter2)) = confusion(targets_test(iter2), pred(iter2))+1;
end
trace(confusion)/sum(confusion(:));
    confusions_test = cat(3,confusions_test, confusion);

scores_test_avg = mean(scores_test,3);
pred = zeros(size(targets_test));
for iter2 = 1:size(scores,2)
    pred(iter2) = find(scores_test_avg(:,iter2)==max(scores_test_avg(:,iter2)));
end
confusion_test_avg = zeros(size(scores,1),size(scores,1));
for iter2 = 1:length(targets_test)
    confusion_test_avg(targets_test(iter2), pred(iter2)) = confusion_test_avg(targets_test(iter2), pred(iter2))+1;
end
trace(confusion_test_avg)/sum(confusion_test_avg(:))


%% get train result
confusions_train = [];
scores_train = [];
  
cd(maindir)
load('result_train.mat');
scores_train = cat(3,scores_train,reshape(scores, [length(scores)/(train_wholesize),train_wholesize]));

targets_train = targets+1;
scores = reshape(scores, [length(scores)/train_wholesize,train_wholesize]);
pred = zeros(size(targets_train));

for iter2 = 1:size(scores,2)
    pred(iter2) = find(scores(:,iter2)==max(scores(:,iter2)));
end

confusion = zeros(size(scores,1),size(scores,1));
for iter2 = 1:length(targets_train)
    confusion(targets_train(iter2), pred(iter2)) = confusion(targets_train(iter2), pred(iter2))+1;
end
trace(confusion)/sum(confusion(:))
    confusions_train = cat(3,confusions_train, confusion);


scores_train_avg = mean(scores_train,3);
pred = zeros(size(targets_train));
for iter2 = 1:size(scores,2)
    pred(iter2) = find(scores_train_avg(:,iter2)==max(scores_train_avg(:,iter2)));
end
confusion_train_avg = zeros(size(scores,1),size(scores,1));
for iter2 = 1:length(targets_train)
    confusion_train_avg(targets_train(iter2), pred(iter2)) = confusion_train_avg(targets_train(iter2), pred(iter2))+1;
end
trace(confusion_train_avg)/sum(confusion_train_avg(:))
%% single confusion

% figure(1)
% imagesc(confusion_test_avg./repmat(sum(confusion_test_avg,2), [1 2]), [0 1]), axis image, colormap(gray)
% 
% confusion_normal = confusion_test_avg./repmat(sum(confusion_test_avg,2), [1 2]);
% bar(confusion_normal(:)');

%%
figure(1)
imagesc(confusion_train_avg./repmat(sum(confusion_train_avg,2), [1 3]), [0 1]), axis image, colormap(gray)
figure(2)
imagesc(confusion_valid_avg./repmat(sum(confusion_valid_avg,2), [1 3]), [0 1]), axis image, colormap(gray)
figure(3)
imagesc(confusion_test_avg./repmat(sum(confusion_test_avg,2), [1 3]), [0 1]), axis image, colormap(gray)

figure(4)
bar(confusion_train_avg)

figure(5)
bar(confusion_valid_avg)

figure(6)
bar(confusion_test_avg)


%% Only test set
clear
close all
clc
load('/data02/gkim/stem_cell_jwshin/outs/230209_3D_b016_lr0.001/epoch[00133]_train[0.9605]_valid[0.9082]_test[0.6528]/result_test.mat')

test_wholesize = 2229;
valid_wholesize = 1078;
train_wholesize = 9601;

scores = reshape(scores, [length(scores)/(test_wholesize),test_wholesize]);
pred = -ones(size(targets));
for iter_dat = 1:length(targets)
pred(iter_dat) = find(scores(:,iter_dat) == max(scores(:,iter_dat)))-1;
end
% pred

confusion = zeros(3,3);

for iter_dat = 1:length(targets)
    confusion(targets(iter_dat)+1, pred(iter_dat)+1) = confusion(targets(iter_dat)+1, pred(iter_dat)+1)+1;
end

paths = paths';
paths = [paths, num2cell(targets)', num2cell(pred)'];%num2cell(abs(pred-double(targets))')];
figure(5), imagesc(confusion./repmat(sum(confusion,2),[1 3]), [0 1]), colormap gray, axis image


idxs_dat = [];
fnames_base ={};

idx_dat = 1;
fname = paths{1,1};
idxs_dot = strfind(fname,'.');
idxs_slash = strfind(fname,'/');
fname_base = fname(idxs_slash(end)+1:idxs_dot(2)-1);
idxs_dat = [idxs_dat; idx_dat];
fnames_base{1} = fname_base;
for iter_dat = 2:length(targets)
    fname = paths{iter_dat,1};
    idxs_dot = strfind(fname,'.');
    idxs_slash = strfind(fname,'/');
    fname_base_ = fname(idxs_slash(end)+1:idxs_dot(2)-1);
    if ~strcmp(fname_base, fname_base_)
        idx_dat = idx_dat + 1;
        fname_base = fname_base_;
        fnames_base{idx_dat} = fname_base;
    end
    idxs_dat = [idxs_dat; idx_dat];
end

trace(confusion/sum(sum(confusion)))
%% colony vote
list_col_error_vote = [];
confusion_col_vote = zeros(3,3);
for iter_col = 1:max(idxs_dat)
    choose_col = find(idxs_dat == iter_col);
    pred_col = mode(pred(choose_col));
    if length(pred_col) > 1
        'there are multiple voitng indexs'
        pred_col = pred_col(find(scores(choose_col,pred_col) == max(scores(choose_col,pred_col))));
    end
    confusion_col_vote(targets(choose_col(1))+1,pred_col+1) = confusion_col_vote(targets(choose_col(1))+1,pred_col+1)+1;

    if targets(choose_col(1)) ~= pred_col
        list_col_error_vote = [list_col_error_vote iter_col];
    end
end


figure(1), imagesc(confusion_col_vote./repmat(sum(confusion_col_vote,2), [1 3]), [0 1]), axis image, colormap gray

fnames_base(list_col_error_vote)

trace(confusion_col_vote/sum(sum(confusion_col_vote)))
%% colony avg
list_col_error_avg = [];
confusion_col_avg = zeros(3,3);
for iter_col = 1:max(idxs_dat)
    choose_col = find(idxs_dat == iter_col);
    pred_col = find(sum(scores(:,choose_col),2) == max(sum(scores(:,choose_col),2)))-1;

    confusion_col_avg(targets(choose_col(1))+1,pred_col+1) = confusion_col_avg(targets(choose_col(1))+1,pred_col+1)+1;
    
    if targets(choose_col(1)) ~= pred_col
        list_col_error_avg = [list_col_error_avg iter_col];
    end

    if (targets(choose_col(1)) == 1) && (pred_col == 2)
        fnames_base{iter_col}
    end
end
figure(2), imagesc(confusion_col_avg./repmat(sum(confusion_col_avg,2),[1 3]), [0 1]), axis image, colormap gray

trace(confusion_col_avg/sum(sum(confusion_col_avg)))
fnames_base(list_col_error_avg)

%% Only test set - h9 data
clear
close all
clc
load('/data02/gkim/stem_cell_jwshin/outs/230209_3D_b016_lr0.001/epoch[00133]_train[0.3333]_valid[0.3333]_test[0.1371]_h9/result_test.mat')

test_wholesize = 1641;

scores = reshape(scores, [length(scores)/(test_wholesize),test_wholesize]);
pred = -ones(size(targets));
for iter_dat = 1:length(targets)
pred(iter_dat) = find(scores(:,iter_dat) == max(scores(:,iter_dat)))-1;
end
% pred

confusion = zeros(3,3);

for iter_dat = 1:length(targets)
    confusion(targets(iter_dat)+1, pred(iter_dat)+1) = confusion(targets(iter_dat)+1, pred(iter_dat)+1)+1;
end

paths = paths';
paths = [paths, num2cell(targets)', num2cell(pred)'];%num2cell(abs(pred-double(targets))')];
figure(5), imagesc(confusion./repmat(sum(confusion,2),[1 3]), [0 1]), colormap gray, axis image


idxs_dat = [];
fnames_base ={};

idx_dat = 1;
fname = paths{1,1};
idxs_dot = strfind(fname,'.');
idxs_slash = strfind(fname,'/');
fname_base = fname(idxs_slash(end)+1:idxs_dot(2)-1);
idxs_dat = [idxs_dat; idx_dat];
fnames_base{1} = fname_base;
for iter_dat = 2:length(targets)
    fname = paths{iter_dat,1};
    idxs_dot = strfind(fname,'.');
    idxs_slash = strfind(fname,'/');
    fname_base_ = fname(idxs_slash(end)+1:idxs_dot(2)-1);
    if ~strcmp(fname_base, fname_base_)
        idx_dat = idx_dat + 1;
        fname_base = fname_base_;
        fnames_base{idx_dat} = fname_base;
    end
    idxs_dat = [idxs_dat; idx_dat];
end

trace(confusion/sum(sum(confusion)))
%% colony vote
list_col_error_vote = [];
confusion_col_vote = zeros(3,3);
for iter_col = 1:max(idxs_dat)
    choose_col = find(idxs_dat == iter_col);
    pred_col = mode(pred(choose_col));
    if length(pred_col) > 1
        'there are multiple voitng indexs'
        pred_col = pred_col(find(scores(choose_col,pred_col) == max(scores(choose_col,pred_col))));
    end
    confusion_col_vote(targets(choose_col(1))+1,pred_col+1) = confusion_col_vote(targets(choose_col(1))+1,pred_col+1)+1;

    if targets(choose_col(1)) ~= pred_col
        list_col_error_vote = [list_col_error_vote iter_col];
    end
end


figure(1), imagesc(confusion_col_vote./repmat(sum(confusion_col_vote,2), [1 3]), [0 1]), axis image, colormap gray

fnames_base(list_col_error_vote)

trace(confusion_col_vote/sum(sum(confusion_col_vote)))
%% colony avg
list_col_error_avg = [];
confusion_col_avg = zeros(3,3);
for iter_col = 1:max(idxs_dat)
    choose_col = find(idxs_dat == iter_col);
    pred_col = find(sum(scores(:,choose_col),2) == max(sum(scores(:,choose_col),2)))-1;

    confusion_col_avg(targets(choose_col(1))+1,pred_col+1) = confusion_col_avg(targets(choose_col(1))+1,pred_col+1)+1;
    
    if targets(choose_col(1)) ~= pred_col
        list_col_error_avg = [list_col_error_avg iter_col];
    end

    if (targets(choose_col(1)) == 0)% && (pred_col == 2)
        fnames_base{iter_col}
    end
end
figure(2), imagesc(confusion_col_avg./repmat(sum(confusion_col_avg,2),[1 3]), [0 1]), axis image, colormap gray

trace(confusion_col_avg/sum(sum(confusion_col_avg)))
fnames_base(list_col_error_avg)


figure(61), bar(sum(confusion,1))
figure(62), bar(sum(confusion_col_avg,1))
