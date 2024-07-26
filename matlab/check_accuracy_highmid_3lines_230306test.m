close all
clear
clc
load('/data02/gkim/stem_cell_jwshin/outs/230502_MIP_b004_in_lr0.001_highmid_3lines_230306test/epoch[00202]_train[0.5000]_valid[0.5000]_test[0.6527]/result_test.mat')


targets = single(targets);
test_wholesize = length(paths);
scores_test = reshape(scores, [length(scores)/(test_wholesize), test_wholesize])';
feats = cell2mat(feats)';
feats_test = reshape(feats, [length(feats)/(test_wholesize), test_wholesize]);
paths_test = paths';
targets_test = single(targets');
preds_test = -ones(size(targets_test));
for iter_data = 1:length(targets_test)
    preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
end


lines_test = zeros(size(targets_test));
for iter_data = 1:length(targets_test)
    if contains(paths_test{iter_data}, '.A2.') || contains(paths_test{iter_data}, '.A2_')
        lines_test(iter_data) = 1;
    elseif contains(paths_test{iter_data}, '.A4.') || contains(paths_test{iter_data}, '.A4_')
        lines_test(iter_data) = 2;
    elseif contains(paths_test{iter_data}, '.A12.') || contains(paths_test{iter_data}, '.A12_')
        lines_test(iter_data) = 3;
    elseif contains(paths_test{iter_data}, '.A15.') || contains(paths_test{iter_data}, '.A15_')
        lines_test(iter_data) = 4;
    elseif contains(paths_test{iter_data}, '.A19.') || contains(paths_test{iter_data}, '.A19_')
        lines_test(iter_data) = 5;
    elseif contains(paths_test{iter_data}, '.A20.') || contains(paths_test{iter_data}, '.A20_')
        lines_test(iter_data) = 6;
    else
        error('No line found')
    end
end
%%
for iter_line = 1:6
    [sum(preds_test(lines_test == iter_line, :) == 1), sum(preds_test(lines_test == iter_line, :) == 2)]
end

%%

save('test_summary.mat', 'paths_test', 'targets_test', 'scores_test', 'preds_test', 'feats_test');