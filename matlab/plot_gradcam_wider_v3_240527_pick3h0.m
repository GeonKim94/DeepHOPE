%% find the best epoch based on line validation

close all
clear
clc
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

save_fig =true;
save_png =true;

dir_cam = '/data02/gkim/stem_cell_jwshin/outs/23_SEC1H5_wider_v3_pick3h0_GM_germline_fishdeep10_b012_in_lr0.001/gdcam';

dir_out0 = '/data02/gkim/stem_cell_jwshin/outs';
dir_case = '23_SEC1H5_wider_v3_pick3h0_GM_germline_fishdeep10_b012_in_lr0.001';%%'23_SEC1H5_wider_v3_allh0_GM_germline_fishdeep10_b012_in_lr0.001_ens';

dir_mask = '/workspace01/gkim/stem_cell_jwshin/data/23_mask_wider_v3_allh_onRA/';


dir_data2 = '/workspace01/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_testiPSC/';
dir_data1 = '/workspace01/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA/';

dir_out = [dir_out0 '/' ...
    dir_case];
dir_outr = [dir_out '_testRA'];
dir_outi = [dir_out '_testiPSC'];

epochs = [];
accs_tr = [];
accs_va = [];
accs_te = [];

cd(dir_out)
list_mdl = dir('epoch*.tar');

for iter_mdl = 1:length(list_mdl)
    cd(dir_out)

    name_mdl = list_mdl(iter_mdl).name;
    idxs_lbra = strfind(name_mdl,'[');
    epochs = [epochs; str2num(name_mdl(idxs_lbra(1)+1:idxs_lbra(1)+5))];
    accs_tr = [accs_tr; str2num(name_mdl(idxs_lbra(5)+1:idxs_lbra(5)+5))];
    accs_va = [accs_va; str2num(name_mdl(idxs_lbra(6)+1:idxs_lbra(6)+5))];
    accs_te = [accs_te; str2num(name_mdl(idxs_lbra(6)+1:idxs_lbra(6)+5))];


    
end
    metrics = accs_va+accs_tr;
    idx_bestmdl = min(find(metrics == max(metrics)));
    dir_bestmdl = list_mdl(idx_bestmdl).name;
        idxs_lbra = strfind(dir_bestmdl,'[');
    epoch_bestmdl_str = (dir_bestmdl(idxs_lbra(1)+1:idxs_lbra(1)+5));

epoch_bestmdl_str = '00371'


targets_test = [];
scores_test = [];
feats_test = [];
paths_test = {};
sets_test = [];
%% get line train result

cd(dir_out)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
dir_plot = [dir_out0 '_present/', dir_case, '/' dir_bestmdl '/'];
mkdir(dir_plot);

cd(dir_bestmdl)
load('result_train')


targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];

for iter_data = 1:length(targets_test)
    preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
end

sets_test = [sets_test; zeros(1, length(paths))'];

%% get line val result

cd(dir_out)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
cd(dir_bestmdl)

load('result_valid')


targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];

for iter_data = 1:length(targets_test)
    preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
end

sets_test = [sets_test; zeros(1, length(paths))'];
%% get line test result

cd(dir_out)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
cd(dir_bestmdl)

load('result_test')


targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];

for iter_data = 1:length(targets_test)
    preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
end
preds_test = preds_test-1;

sets_test = [sets_test; ones(1, length(paths))'];
%% get RA test_result


cd(dir_outr)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
cd(dir_bestmdl)

load('result_test')

targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];

sets_test = [sets_test; ones(1, length(paths))'];
%% get RA val_result


cd(dir_outr)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
cd(dir_bestmdl)

load('result_valid')

targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];

sets_test = [sets_test; ones(1, length(paths))'];
%% get RA train_result


cd(dir_outr)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
cd(dir_bestmdl)

load('result_train')

targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];

sets_test = [sets_test; ones(1, length(paths))'];
%% get iPSC test_result

cd(dir_outi)
dir_bestmdl = dir(['epoch[' epoch_bestmdl_str '*]']);
dir_bestmdl = dir_bestmdl(1).name;
cd(dir_bestmdl)

load('result_test')

targets = single(targets);
test_wholesize = length(paths);
scores_test = [scores_test; reshape(scores, [length(scores)/(test_wholesize), test_wholesize])'];
feats = cell2mat(feats)';
feats_test = [feats_test; reshape(feats, [length(feats)/(test_wholesize), test_wholesize])'];
paths_test = [paths_test; paths'];
targets_test = [targets_test; single(targets')];
sets_test = [sets_test; ones(1, length(paths))'];

%%
for iter_data = 1:length(paths_test)
    path_test = paths_test{iter_data};
    path_test = strrep(path_test, '/data01/','/workspace01/');
    path_test = strrep(path_test, '/data02/','/workspace01/');

    paths_test{iter_data} = path_test;
end

[paths_test, idxs_unq] = unique(paths_test);
feats_test = feats_test(idxs_unq,:);
targets_test = targets_test(idxs_unq,:);
sets_test = sets_test(idxs_unq,:);
scores_test = scores_test(idxs_unq,:);

%% tried to make the class assigning easier but this is harder lmao

% targets_test = -ones(size(targets_test));
% preds_test = -ones(size(targets_test));
% for iter_data = 1:length(targets_test)
%     preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
% end
% preds_test = preds_test-1;
% lines_test = zeros(size(targets_test)); % represents plot location and (potentially) markers 
% % types_test = zeros(size(targets_test)); % represents the color
% 
% keywords = {
%     {'cat'},              % Class 1
%     {'dog'},              % Class 2
%     {'bird'},             % Class 3
%     {{'cat', 'dog'}, {'bird', 'fish'}} % Class 4 (either 'cat' and 'dog' OR 'bird' and 'fish')
% };
% classValues = [1, 2, 3, 4];
% 
% % Iterate through the filenames and assign class values
% for i = 1:length(filenames)
%     for k = 1:length(keywords)
%         if iscell(keywords{k}{1}) % Check if it's a nested cell array (for multiple conditions)
%             % Check if any set of conditions for the current class are met
%             anyConditionMet = any(cellfun(@(condition) all(cellfun(@(word) contains(filenames{i}, word, 'IgnoreCase', true), condition)), keywords{k}));
%             if anyConditionMet
%                 classes(i) = classValues(k);
%                 break; % Break the loop once the class is assigned
%             end
%         else
%             % Check if all keywords for the current class are present in the filename
%             allKeywordsPresent = all(cellfun(@(word) contains(filenames{i}, word, 'IgnoreCase', true), keywords{k}));
%             if allKeywordsPresent
%                 classes(i) = classValues(k);
%                 break; % Break the loop once the class is assigned
%             end
%         end
%     end
% end
% 
% % Display the result
% disp(classes);


%%

targets_test = -ones(size(targets_test));
preds_test = -ones(size(targets_test));
for iter_data = 1:length(targets_test)
    preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
end
preds_test = preds_test-1;

lines_test = zeros(size(targets_test)); % represents plot location and (potentially) markers 
% types_test = zeros(size(targets_test)); % represents the color
for iter_data = 1:length(targets_test)
    fname = paths_test{iter_data};
    idx_slash = strfind(fname,'/');
    fname = fname(idx_slash(end)+1:end);
    if contains(fname, 'GM') || contains(fname, 'gm') || contains(fname, '230719.')  || contains(fname, '230720.') 
        if contains(fname, 'Endo') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 2;
            % types_test(iter_data) = 2;%2;
            targets_test(iter_data) = 6;
        elseif contains(fname, 'Endo') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 3;
            % types_test(iter_data) = 3;%2;
            targets_test(iter_data) = 7;
        elseif contains(fname, 'Endo') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 4;
            % types_test(iter_data) = 3;
            targets_test(iter_data) = 8;
        elseif contains(fname, 'Endo') && contains(fname,{'48h', '48H'})
            continue
            lines_test(iter_data) = 5;
            % types_test(iter_data) = 3;
            targets_test(iter_data) = 10;
        elseif contains(fname, 'Meso') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 5;
            % types_test(iter_data) = 4;%4;
            targets_test(iter_data) = 3;
        elseif contains(fname,'Meso') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 6;
            % types_test(iter_data) = 4;%4;
            targets_test(iter_data) = 4;
        elseif contains(fname,'Meso') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 7;
            % types_test(iter_data) = 4;
            targets_test(iter_data) = 5;
        elseif contains(fname,'Meso') && contains(fname,{'48h', '48H'})
            continue
            lines_test(iter_data) = 9;
            % types_test(iter_data) = 4;
            targets_test(iter_data) = 6;
        elseif contains(fname, 'Ecto') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 8;
            % types_test(iter_data) = 5;%4;
            targets_test(iter_data) = 0;
        elseif contains(fname,'Ecto') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 9;
            % types_test(iter_data) = 5;%6;
            targets_test(iter_data) = 1;
        elseif contains(fname,'Ecto') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 10;
            % types_test(iter_data) = 5;
            targets_test(iter_data) = 2;
        elseif contains(fname,'Ecto') && contains(fname,{'48h', '48H'})
            continue
            lines_test(iter_data) = 13;
            % types_test(iter_data) = 5;
            targets_test(iter_data) = 3;
        elseif contains(fname,'Ecto') && contains(fname,{'72h', '72H'})
            continue
            lines_test(iter_data) = 14;
            % types_test(iter_data) = 5;
            targets_test(iter_data) = 4;
        elseif contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 12;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 11;
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 11;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 10;
        else
            lines_test(iter_data) = 1;
            targets_test(iter_data) = 9;
        end

    elseif contains(fname, {'H9', 'h9', '230714.'}) 
        if contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 16;
            % types_testd(iter_data) = 7;
            targets_test(iter_data) = 11;
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 15;
            targets_test(iter_data) = 10;
        else
            lines_test(iter_data) = 14;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 9;
        end

    elseif contains(fname, {'230427', '230713'})%contains(fname, 'JAX', 'Jax', '230713.') 
        % old jax
        continue

    elseif contains(fname,{'JAX','Jax','jax'})
        % late jax
        if contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 20;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 11;
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 19;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 10;
        else
            lines_test(iter_data) = 18;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 9;
        end

    % high
    elseif contains(fname, {'.HD09.', 'HD09_'})
        lines_test(iter_data) = 22;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 12;
    elseif contains(fname, {'.HD11.', 'HD11_'})
        lines_test(iter_data) = 23;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 12;
    elseif contains(fname, {'.A2.', 'A2_'})
        lines_test(iter_data) = 24;
        % types_test(iter_data) = 7;
        targets_test(iter_data) = 12;
    elseif contains(fname, {'.A12.', 'A12_'})
        lines_test(iter_data) = 25;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 12;

    % low

    elseif contains(fname, {'.HD18.', 'HD18_'})
        lines_test(iter_data) = 26;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 13;
    elseif contains(fname, {'.B13.', 'B13_'})
        lines_test(iter_data) = 27;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 13;
    elseif contains(fname, {'.A19.', 'A19_'})
        lines_test(iter_data) = 28;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 13;
    elseif contains(fname, {'.A20.', 'A20_'})
        lines_test(iter_data) = 29;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 13;


    elseif contains(fname, {'.HD02.', 'HD02_'})
%         lines_test(iter_data) = 26;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.HD03.', 'HD03_'})
%         lines_test(iter_data) = 27;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.HD04.', 'HD04_'})
%         lines_test(iter_data) = 28;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.HD05.', 'HD05_'})
%         lines_test(iter_data) = 29;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.HD07.', 'HD07_'})
%         lines_test(iter_data) = 30;
%         % types_test(iter_data) = 6;
        continue
    
    elseif contains(fname, {'.HD12.', 'HD12_'})
%         lines_test(iter_data) = 33;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.HD14.', 'HD14_'})
%         lines_test(iter_data) = 34;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.HD15.', 'HD15_'})
%         lines_test(iter_data) = 35;
%         % types_test(iter_data) = 6;
        continue
    
    elseif contains(fname, {'.HD25.', 'HD25_'})
%         lines_test(iter_data) = 37;
%         % types_test(iter_data) = 6;
        continue

    elseif contains(fname, {'.B1.', 'B1_','.BJ01.'})
%         lines_test(iter_data) = 39;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.BJ04.', 'BJ04_'})
%         lines_test(iter_data) = 40;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.B7.', 'B7_'})
%         lines_test(iter_data) = 41;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.BJ11.'})
%         lines_test(iter_data) = 42;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.B12.', 'B12_'})
%         lines_test(iter_data) = 43;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.BJ14.'})
%         lines_test(iter_data) = 45;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.B17.', 'B17_', '.BJ17.'})
%         lines_test(iter_data) = 46;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.B18.', 'B18_'})
%         lines_test(iter_data) = 47;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.B21.', 'B21_'})
%         lines_test(iter_data) = 48;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.BJ22.'})
%         lines_test(iter_data) = 49;
%         % types_test(iter_data) = 6;
        continue
    elseif contains(fname, {'.B23.', 'B23_'})
%         lines_test(iter_data) = 50;
%         % types_test(iter_data) = 6;
        continue


    else
        continue
        error('No line found')
    end

end
%%

corrects_test = -ones(size(targets_test));
for iter_target = 0:9
    
    idxs_data = targets_test == iter_target;

    [~, preds_] = max(scores_test(idxs_data,:),[],2);

    corrects_test(idxs_data) = (preds_-1) == targets_test(idxs_data);

end

for iter_target = 10:13
    
    idxs_data = targets_test == iter_target;

    [~, preds_] = max(scores_test(idxs_data,:),[],2);

    corrects_test(idxs_data) = (preds_-1) ~= 9;

end


%% plot t-SNE
colors_line = [[3/4 3/4 3/4]%'GM untreated',%1
        [3/4 1 1]%'GM + endo 6h',%2
        [1/2 1 1]%'GM + endo 12h',%3
        [0 1 1]%'GM + endo 24h',%4
%         [0 0.8 0.8]%'GM + endo 48h',%5
        [1 3/4 1]%'GM + meso 6h',%6
        [1 1/2 1]%'GM + meso 12h',%7
        [1 0 1]%'GM + meso 24h',%8
%         [0.8 0 0.8]%'GM + meso 48h',%9
        [1 1 3/4]%'GM + ecto 6h',%10
        [1 1 1/2]%'GM + ecto 12h',%11
        [1 1 0]%'GM + ecto 24h',%12
%         [0.8 0.8 0]%'GM + ecto 48h',%13
%         [0.6 0.6 0]%'GM + ecto 72h',%14
        [1 1/2 1/2]%'GM + RA 12h',%15
        [1 0 0]%'GM + RA 24h',%16
        [1 1 1]%% nothing
        [3/4 3/4 3/4]%'H9 untreated',%18
        [1 1/2 1/2]%'H9 + RA 12h',%19
        [1 0 0]%'H9 + RA 24h',%20
        [1 1 1]%% nothing
        [3/4 3/4 3/4]%'JAX untreated',%22
        [1 1/2 1/2]%'JAX + RA 12h',%23
        [1 0 0]%'JAX + RA 24h',%24
        [1 1 1]%% nothing
        [1/2 1/2 1/2]%'HD 09',%26
        [1/2 1/2 1/2+0.01]%'HD 11',%27
        [1/2 1/2 1/2+0.02]%'PD 02',%@8
        [1/2 1/2 1/2+0.03]%'PD 12',%29
        [1/2 1 0]%]%'HD 18',%30
        [1/2+0.01 1 0]%'BJ 13',%31
        [1/2+0.02 1 0]%'PD 19',%32
        [1/2+0.03 1 0]%'PD 20',%33
        [1 1 1]%'HD 02',%25
        [1 1 1]%[1/2 1/2 1/2]%'HD 03',%26
        [1 1 1]%[1/2 1/2 1/2]%'HD 04',%27
        [1 1 1]%[1/2 1/2 1/2]%'HD 05',%28
        [1 1 1]%[1/2 1/2 1/2]%'HD 07',%29
        [1 1 1]%[1 0 0]%'HD 12',%32
        [1 1 1]%[1/2 1/2 1/2]%'HD 14',%33
        [1 1 1]%[1/2 1/2 1/2]%'HD 15',%34
        [1 1 1]%[1/2 1/2 1/2]%'HD 25',%36
        [1 1 1]%% nothing
        [1 1 1]%[1/2 1/2 1/2]%'BJ 01',%38
        [1 1 1]%[1/2 1/2 1/2]%'BJ 04',%39
        [1 1 1]%[1/2 1/2 1/2]%'BJ 07',%40
        [1 1 1]%[1 0 0]%'BJ 11',%41
        [1 1 1]%[1/2 1/2 1/2]%'BJ 12',%42
        [1 1 1]%[1/2 1/2 1/2]%'BJ 14',%44
        [1 1 1]%[1/2 1/2 1/2]%'BJ 17',%45
        [1 1 1]%[1/2 1/2 1/2]%'BJ 18',%46
        [1 1 1]%[1/2 1/2 1/2]%'BJ 21',%47
        [1 1 1]%[1/2 1/2 1/2]%'BJ 22',%48
        [1 1 1]%[1 0 0]%'BJ 23',%49
        [1 1 1]%% nothing
        ];


colors_line_notime = [[2/4 2/4 2/4]%'GM untreated',%1
        [0 1 1]%'GM + endo 6h',%2
        [0 1 1]%'GM + endo 12h',%3
        [0 1 1]%'GM + endo 24h',%4
%         [0 0.8 0.8]%'GM + endo 48h',%5
        [1 0 1]%'GM + meso 6h',%6
        [1 0 1]%'GM + meso 12h',%7
        [1 0 1]%'GM + meso 24h',%8
%         [0.8 0 0.8]%'GM + meso 48h',%9
        [1 1 0]%'GM + ecto 6h',%10
        [1 1 0]%'GM + ecto 12h',%11
        [1 1 0]%'GM + ecto 24h',%12
%         [0.8 0.8 0]%'GM + ecto 48h',%13
%         [0.6 0.6 0]%'GM + ecto 72h',%14
        [1 0 0]%'GM + RA 12h',%15
        [1 0 0]%'GM + RA 24h',%16
        [1 1 1]%% nothing
        [3/4 3/4 3/4]%'H9 untreated',%18
        [1 1/2 1/2]%'H9 + RA 12h',%19
        [1 0 0]%'H9 + RA 24h',%20
        [1 1 1]%% nothing
        [3/4 3/4 3/4]%'JAX untreated',%22
        [1 1/2 1/2]%'JAX + RA 12h',%23
        [1 0 0]%'JAX + RA 24h',%24
        [1 1 1]%% nothing
        [1/2 1/2 1/2]%'HD 09',%26
        [1/2 1/2 1/2+0.01]%'HD 11',%27
        [1/2 1/2 1/2+0.02]%'PD 02',%@8
        [1/2 1/2 1/2+0.03]%'PD 12',%29
        [1/2 1 0]%]%'HD 18',%30
        [1/2+0.01 1 0]%'BJ 13',%31
        [1/2+0.02 1 0]%'PD 19',%32
        [1/2+0.03 1 0]%'PD 20',%33
        [1 1 1]%'HD 02',%25
        [1 1 1]%[1/2 1/2 1/2]%'HD 03',%26
        [1 1 1]%[1/2 1/2 1/2]%'HD 04',%27
        [1 1 1]%[1/2 1/2 1/2]%'HD 05',%28
        [1 1 1]%[1/2 1/2 1/2]%'HD 07',%29
        [1 1 1]%[1 0 0]%'HD 12',%32
        [1 1 1]%[1/2 1/2 1/2]%'HD 14',%33
        [1 1 1]%[1/2 1/2 1/2]%'HD 15',%34
        [1 1 1]%[1/2 1/2 1/2]%'HD 25',%36
        [1 1 1]%% nothing
        [1 1 1]%[1/2 1/2 1/2]%'BJ 01',%38
        [1 1 1]%[1/2 1/2 1/2]%'BJ 04',%39
        [1 1 1]%[1/2 1/2 1/2]%'BJ 07',%40
        [1 1 1]%[1 0 0]%'BJ 11',%41
        [1 1 1]%[1/2 1/2 1/2]%'BJ 12',%42
        [1 1 1]%[1/2 1/2 1/2]%'BJ 14',%44
        [1 1 1]%[1/2 1/2 1/2]%'BJ 17',%45
        [1 1 1]%[1/2 1/2 1/2]%'BJ 18',%46
        [1 1 1]%[1/2 1/2 1/2]%'BJ 21',%47
        [1 1 1]%[1/2 1/2 1/2]%'BJ 22',%48
        [1 1 1]%[1 0 0]%'BJ 23',%49
        [1 1 1]%% nothing
        ];

%% load stuff
% hand-crafted features
dir_feat = '/workspace01/gkim/stem_cell_jwshin/data/23_feat_wider_v3_allh_onRA_new';

load([dir_feat '/feats.mat']);

    for iter_data = 1:length(paths_test)
        path_test = paths_test{iter_data};
        path_test = strrep(path_test, '/data01/','/workspace01/');
        path_test = strrep(path_test, '/data02/','/workspace01/');
   
        paths_test{iter_data} = path_test;
    end

    feats_hc = feats_hc(idxs_unq,:);
    paths_test = paths_test(idxs_unq,:);

%% search good data
size_crop = 2048;

% cd([dir_cam sprintf('/epoch[%s]', epoch_bestmdl_str)])
% name_layer = 'tail_layer3';

dir_gdcam = [dir_cam sprintf('/epoch[%s]', epoch_bestmdl_str) '/tail_layer4/'];

fnames_select = { %layer3
    '231101.132003.HD09.026.Group1.A1.S026' %good ipsc
    '231215.155545.HD18.036.Group1.A1.S036' %bad ipsc
    '231211.160713.GM25256_Endoderm_24h.046.Group1.A1.S046' %endo 24
    '230914.181401.B13.007.Group1.A1.S007' % badipsc
    '230914.181808.B13.011.Group1.A1.S011'
    '230914.184110.B13.027.Group1.A1.S027'
    '230914.191731.B13.048.Group1.A1.S048'
    '230914.192531.B13.054.Group1.A1.S054'
    '230914.192804.B13.056.Group1.A1.S056'
    '231102.195748.HD11.028.Group1.A1.S028'
%     '230921.145659.GM25256_untreated.019.Group1.A1.S01' % internal ctl ctl
%     '230921.160343.GM25256_untreated.039.Group1.A1.S03' % both ctl endo6
    };

fnames_select = { % layer4
    '230914.181401.B13.007.Group1.A1.S007'

    };

fnames_select = { % layer2
    '230914.191934.B13.050.Group1.A1.S050'

    };
idxs_select = zeros(length(fnames_select),1);
h_fig = figure();
    h_fig.Position = [0 0 1024 1024];
    h_fig.Color = [1 1 1];

for iter_data = find((lines_test>0).*(lines_test<=inf).*(sets_test>0).*(corrects_test == 1))'

    path_data = paths_test{iter_data};
    [dir_data,fname_data,ext_data] = fileparts(path_data);
%     idxs_select(iter_data) = find(contains(paths_test,fname_data));

    [path_mask, found] = search_recursive_v2(dir_mask,fname_data,false);
    [path_gdcam, found] = search_recursive_v2(dir_gdcam,fname_data,false);
    [path_data, found] = search_recursive_v2(dir_data1,fname_data,false);
    if strcmp(path_data, 'none')
        [path_data, found] = search_recursive_v2(dir_data2,fname_data,false);
    end
    
    
    try    
        load(path_mask);
        load(path_gdcam);
        data = h5read(path_data, '/ri');
        data = cencrop2d(data,size_crop,size_crop);
    catch
        continue
    end

    set(0, 'CurrentFigure', h_fig)
    subplot(4,3,1),
    imagesc(max(data,[],3),[13300 13800])
    ax = gca;
    set(gca,'Colormap',gray), axis image

    for iter_cls = 1:10

        gdcam_cls = gdcam_stack(:,:,iter_cls);
        gdcam_cls = imresize(gdcam_cls,[size_crop,size_crop], 'bilinear')';
        gdcam_cls = gdcam_cls/mean2(gdcam_cls);

        gdcam_cls = cencrop2d(gdcam_cls,size(mask_stitch,1),size(mask_stitch,2));
%         
%     
%         mask_peri_ = mask_boundin+mask_boundout;
%         if size(mask_boundin,1) > size_crop || size(mask_boundin,2) > size_crop 
%             mask_peri_ = cencrop2d(mask_peri_,size_crop,size_crop);
%         end
% %         mask_peri_ = cenput2d(false(size_crop,size_crop),mask_peri_);
%         mask_bg_ = ~imfill(mask_peri_,'holes');
%         mask_gap_ = mask_gap;
%         if size(mask_boundin,1) > size_crop || size(mask_boundin,2) > size_crop 
%             mask_gap_ = cencrop2d(mask_gap_,size_crop,size_crop);
%         end
% %         mask_gap_ = cenput2d(false(size_crop,size_crop),mask_gap_);
%         mask_cyto_ = (imfill(mask_peri_,'holes').*~(mask_peri_+mask_gap_))>0;
%     
%         gdcam_peri = mean(gdcam_cls(find(mask_peri_)));
%         gdcam_bg = mean(gdcam_cls(mask_bg_));
%         gdcam_gap = mean(gdcam_cls(mask_gap_));
%         gdcam_cyto = mean(gdcam_cls(mask_cyto_));
    
    
        set(0, 'CurrentFigure', h_fig)
        subplot(4,3,iter_cls + 2),
        imagesc(gdcam_cls, [0 10])
    ax = gca;
    set(gca,'Colormap',turbo), axis image



    end

    path_save = strrep(strrep(path_gdcam, '/epoch','_png2/epoch'), '.mat', '.png');
    [dir_save fname_save ext_save] = fileparts(path_save);
    mkdir(dir_save)%(strrep(dir_gdcam, '/epoch','_png2/epoch'))
    
    saveas(h_fig, path_save);

end


%% representative data 
size_crop = 2048;

% cd([dir_cam sprintf('/epoch[%s]', epoch_bestmdl_str)])
% name_layer = 'tail_layer3';

dir_gdcam = [dir_cam sprintf('/epoch[%s]', epoch_bestmdl_str) '/tail_layer4/'];

fnames_select = {
    '230914.181808.B13.011.Group1.A1.S011'
    '230914.192804.B13.056.Group1.A1.S056' %badipsc
    '231101.132003.HD09.026.Group1.A1.S026' %good ipsc
    '231215.155545.HD18.036.Group1.A1.S036' %bad ipsc
    '231211.160713.GM25256_Endoderm_24h.046.Group1.A1.S046' %endo 24
%     '230921.145659.GM25256_untreated.019.Group1.A1.S01' % internal ctl ctl
%     '230921.160343.GM25256_untreated.039.Group1.A1.S03' % both ctl endo6
    };

idxs_select = zeros(length(fnames_select),1);
h_fig = figure();
    h_fig.Position = [0 0 1024 1024];
    h_fig.Color = [1 1 1];
for iter_data = 1:length(fnames_select)
    fname_data = fnames_select{iter_data};
%     idxs_select(iter_data) = find(contains(paths_test,fname_data));

     [path_mask, found] = search_recursive_v2(dir_mask,fname_data,false);
    [path_gdcam, found] = search_recursive_v2(dir_gdcam,fname_data,false);
    [path_data, found] = search_recursive_v2(dir_data1,fname_data,false);
    if strcmp(path_data, 'none')
        [path_data, found] = search_recursive_v2(dir_data2,fname_data,false);
    end
    
    
    try    
        load(path_mask);
        load(path_gdcam);
        data = h5read(path_data, '/ri');
        data = cencrop2d(data,size_crop,size_crop);
    catch
        continue
    end

    set(0, 'CurrentFigure', h_fig)
    subplot(4,3,1),
    imagesc(max(data,[],3),[13300 13800])
    ax = gca;
    set(gca,'Colormap',gray), axis image

    for iter_cls = 1:10

        gdcam_cls = gdcam_stack(:,:,iter_cls);
        gdcam_cls = imresize(gdcam_cls,[size_crop,size_crop], 'bicubic')';
        gdcam_cls = gdcam_cls/mean2(gdcam_cls);

        gdcam_cls = cencrop2d(gdcam_cls,size(mask_stitch,1),size(mask_stitch,2));
%         
%     
%         mask_peri_ = mask_boundin+mask_boundout;
%         if size(mask_boundin,1) > size_crop || size(mask_boundin,2) > size_crop 
%             mask_peri_ = cencrop2d(mask_peri_,size_crop,size_crop);
%         end
% %         mask_peri_ = cenput2d(false(size_crop,size_crop),mask_peri_);
%         mask_bg_ = ~imfill(mask_peri_,'holes');
%         mask_gap_ = mask_gap;
%         if size(mask_boundin,1) > size_crop || size(mask_boundin,2) > size_crop 
%             mask_gap_ = cencrop2d(mask_gap_,size_crop,size_crop);
%         end
% %         mask_gap_ = cenput2d(false(size_crop,size_crop),mask_gap_);
%         mask_cyto_ = (imfill(mask_peri_,'holes').*~(mask_peri_+mask_gap_))>0;
%     
%         gdcam_peri = mean(gdcam_cls(find(mask_peri_)));
%         gdcam_bg = mean(gdcam_cls(mask_bg_));
%         gdcam_gap = mean(gdcam_cls(mask_gap_));
%         gdcam_cyto = mean(gdcam_cls(mask_cyto_));
    
    
        set(0, 'CurrentFigure', h_fig)
        subplot(4,3,iter_cls + 2),
        imagesc(gdcam_cls, [0 10])
    ax = gca;
    set(gca,'Colormap',turbo), axis image



    end
    pause()

end


%%

pause(600)
maskgdcam_test = -ones(size(targets_test,1),4);

for iter_data = find((lines_test>0).*(lines_test<=10).*(corrects_test == 1))'%.*(sets_test>0)
    
    path_data = paths_test{iter_data};
    [dir_data,fname_data,ext_data] = fileparts(path_data);

    [path_mask, found] = search_recursive_v3(dir_mask,fname_data,false);
    [path_gdcam, found] = search_recursive_v3(dir_gdcam,fname_data,false);

%     if contains(path_gdcam, '/train/')
%         continue
%     end
    
    try
    load(path_mask);
    load(path_gdcam);
    catch
        continue
    end
    gdcam_cls = gdcam_stack(:,:,targets_test(iter_data)+1);
    gdcam_cls = imresize(gdcam_cls,[size_crop,size_crop], 'nearest')';
%     gdcam_cls = gdcam_cls/max(max(gdcam_cls));

    mask_peri_ = mask_boundin+mask_boundout;
    if size(mask_boundin,1) > size_crop || size(mask_boundin,2) > size_crop 
        mask_peri_ = cencrop2d(mask_peri_,size_crop,size_crop);
    end
    mask_peri_ = cenput2d(false(size_crop,size_crop),mask_peri_);
    se = strel('disk',size_crop/64);
    mask_peri_ = imdilate(bwskel(mask_peri_), se);
    mask_bg_ = ~imfill(mask_peri_,'holes');
    mask_gap_ = mask_gap;
    if size(mask_boundin,1) > size_crop || size(mask_boundin,2) > size_crop 
        mask_gap_ = cencrop2d(mask_gap_,size_crop,size_crop);
    end
    mask_gap_ = cenput2d(false(size_crop,size_crop),mask_gap_);
    mask_gap_ = imdilate(bwskel(mask_gap_), se);
    mask_cyto_ = (imfill(mask_peri_,'holes').*~(mask_peri_+mask_gap_))>0;

    gdcam_cls = cencrop2d(gdcam_cls, size(mask_gap,1),size(mask_gap,2));
    mask_peri_ = cencrop2d(mask_peri_, size(mask_gap,1),size(mask_gap,2));
    mask_bg_ = cencrop2d(mask_bg_, size(mask_gap,1),size(mask_gap,2));
    mask_cyto_ = cencrop2d(mask_cyto_, size(mask_gap,1),size(mask_gap,2));
    mask_gap_ = cencrop2d(mask_gap_, size(mask_gap,1),size(mask_gap,2));


    gdcam_peri = mean(gdcam_cls(mask_peri_));
    gdcam_bg = mean(gdcam_cls(mask_bg_));
    gdcam_gap = mean(gdcam_cls(mask_gap_));
    gdcam_cyto = mean(gdcam_cls(mask_cyto_));

    
    maskgdcam_test(iter_data,:) = [gdcam_bg, gdcam_cyto, gdcam_peri, gdcam_gap];

end


save([dir_gdcam, '/maskgdcam.mat'], 'maskgdcam_test');

%%

for iter_data = 1:size(maskgdcam_test,1)
    if any(isnan(maskgdcam_test(iter_data,:)))
        maskgdcam_test(iter_data,:) = -ones(size(maskgdcam_test(iter_data,:)));
    end

end

%%
close all
h_ = figure(1);
h_.Color = [1 1 1];
h_.Position = [0 100 1200 900];
idx_subplot = 2;
for iter_cls = [9 6 7 8 3 4 5 0 1 2]
    idxs_cls = find((targets_test == iter_cls).*(maskgdcam_test(:,1)~= -1));

    sprintf('length of cls %d : %d', iter_cls, length(idxs_cls))

    idx_subplot = idx_subplot+1;


    set(0,'CurrentFigure', h_);
    ax = subplot(4,3,idx_subplot);
    hold on
    x_ = 0;
    for iter_part = 1:size(maskgdcam_test, 2)

        x_ = x_+1;
        b = bar(x_,mean(maskgdcam_test(idxs_cls,iter_part),1), 'FaceColor', [3/4 3/4 3/4]);

        errorbar(x_, mean(maskgdcam_test(idxs_cls,iter_part),1), std(maskgdcam_test(idxs_cls,iter_part),0,1), 'k-')

        ylim([0 5.5*10^-11])
        xlim([0 5])

    end
    ax.TickLength = [0 0];
    ax.YTick = [];
    hold off
end


saveas(h_, [dir_gdcam, '/regional_gradcam_scores.fig']);


%%
