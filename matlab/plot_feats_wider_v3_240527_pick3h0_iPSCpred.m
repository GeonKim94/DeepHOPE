%% find the best epoch based on line validation

close all
clear
clc
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

save_fig =true;
save_png =true;

dir_feat = '/workspace01/gkim/stem_cell_jwshin/data/23_feat_wider_v3_allh_onRA_new';
dir_out0 = '/data02/gkim/stem_cell_jwshin/outs';
dir_case = '23_SEC1H5_wider_v3_pick3h0_GM_germline_fishdeep10_b012_in_lr0.001';%%'23_SEC1H5_wider_v3_allh0_GM_germline_fishdeep10_b012_in_lr0.001_ens';
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


    
    metrics = accs_va+accs_tr;
    idx_bestmdl = min(find(metrics == max(metrics)));
    dir_bestmdl = list_mdl(idx_bestmdl).name;
        idxs_lbra = strfind(dir_bestmdl,'[');
    epoch_bestmdl_str = (dir_bestmdl(idxs_lbra(1)+1:idxs_lbra(1)+5));
end

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
% dir_plot = [dir_out0 '_present/', dir_case, '/' dir_bestmdl '/'];
% mkdir(dir_plot);

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

sets_test = [sets_test; zeros(1, length(paths_test))'];

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

sets_test = [sets_test; zeros(1, length(paths_test))'];
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

sets_test = [sets_test; ones(1, length(paths_test))'];
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

sets_test = [sets_test; ones(1, length(paths_test))'];
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

sets_test = [sets_test; ones(1, length(paths_test))'];
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

sets_test = [sets_test; ones(1, length(paths_test))'];
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
sets_test = [sets_test; ones(1, length(paths_test))'];

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
                'aspect_ratio_xz',...
                'curvature'};
try

    cd(dir_feat)
    load('feats.mat', 'feats_hc', 'paths_test', 'names_feat')
    
    %%
    for iter_data = 1:length(paths_test)
        path_test = paths_test{iter_data};
        path_test = strrep(path_test, '/data01/','/workspace01/');
        path_test = strrep(path_test, '/data02/','/workspace01/');
    
        paths_test{iter_data} = path_test;
    end

    feats_hc = feats_hc(idxs_unq,:);
    paths_test = paths_test(idxs_unq,:);

catch


feats_hc = [];
    for iter_data = 1:length(paths_test)
    
        path_data = paths_test{iter_data};
        [dir_data, fileName, fileExt] = fileparts(path_data);
    
        [dir_find, found] = search_recursive_v2(dir_feat, fileName,false);
    
        load(dir_find)
    
        feat_hc = [volume_colony,area_colony,RI_avg_colony,RI_std_colony,...
                    volume_lip,area_lip,RI_avg_lip,RI_std_lip,...# volume_cyto = volume_colony-volume_lip
                    volume_cyto,area_cyto,RI_avg_cyto,RI_std_cyto,...
                    area_gap,len_bound,roundness,solidity,eccentricity,...
                    n_boundin,n_boundout,ncont_bound,...
                    thick_avg,thick_std, spread_thick/sqrt(area_colony), sqrt(sum(abs(skew_thick).^2)), kurt_thick,...
                    spread_dm/sqrt(area_colony), sqrt(sum(abs(skew_dm).^2)), kurt_dm,...
                    spread_lip/sqrt(area_colony), sqrt(sum(abs(skew_lip).^2)), kurt_lip,...
                    volume_lip/volume_colony,...
                    area_gap/area_colony,...
                    sqprt(pi)*thick_avg/2/sqrt(area_colony),...
                    curvature];
        feats_hc = [feats_hc; feat_hc];
        
    end
    
    cd(dir_feat)
    save('feats.mat', 'feats_hc', 'paths_test', 'names_feat')
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

%% plot features - ipscpred

% 
colors_type = [
    1 0 1;...
    %     0.5 0.5 0;...
%     0 0.5 0.5;...
    0 1 1;...
%     0.5 0 0.5;...
    1 1 0;... 
    0 0 0;...
    0.5 0.5 0.5];



xs = [24 25 27 28 30 31];
labels_xtick = {'High -> High',
        'High -> Low',
        'Low -> High',
        'Low -> Low'
        };

colors_line=[
    [3/4 3/4 3/4]
    [2/4 2/4 1]
    [3/4 3/4 3/4]
    [2/4 2/4 1]
    ];

% dir_plot = dir_feat;
% mkdir(dir_plot)

for iter_feat = 1:size(feats_hc,2)

    h_fig = figure(iter_feat);
    h_fig.Position = [0 0 800 600];
    h_fig.Color = [1 1 1];
    hold on
    for target_ = 12:13

        for correct_ = 0:1
            idxs_thisline = intersect(find(targets_test == target_),find(corrects_test == correct_));
 
            
            y_ = feats_hc(idxs_thisline,iter_feat);
            x_ = target_*2+(1-correct_)*(13-target_)+(correct_+1)*(target_-12);
    
            bar(x_,mean(y_),'LineWidth',2,'FaceColor','none','FaceColor', colors_line(find(xs == x_),:));
            errorbar(x_,mean(y_),std(y_,0,1), 'k','LineWidth',2);
        end
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
        ylim([0 inf])
    else
        ylim([0 inf])
    end


    xticks(unique(xs));
    xticklabels(labels_xtick);
    set(gca, 'FontSize', 12)
    title(strrep(names_feat{iter_feat}, '_', ' '));

    saveas(gcf,[dir_feat, '/iPSC_featvspred_', names_feat{iter_feat}, '.fig']);
    saveas(gcf,[dir_feat, '/iPSC_featvspred_', names_feat{iter_feat}, '.png']);

end

%% plot features - RA pred
close all
% 
colors_type = [
    1 0 1;...
    %     0.5 0.5 0;...
%     0 0.5 0.5;...
    0 1 1;...
%     0.5 0 0.5;...
    1 1 0;... 
    0 0 0;...
    0.5 0.5 0.5];



xs = [24 25 28 27 31 30];
labels_xtick = {'Ctl -> Ctl',
        'CTL -> Exp',
        'RA 12h -> Exp',
        'RA 12h -> Ctl',
        'RA 24h -> Exp'
        'RA 24h -> Ctl',
        };

colors_line=[
    [3/4 3/4 3/4]
    [2/4 2/4 1]
    [2/4 2/4 1]
    [3/4 3/4 3/4]
    [2/4 2/4 1]
    [3/4 3/4 3/4]
    ];

% dir_plot = dir_feat;
% mkdir(dir_plot)

for iter_feat = 1:size(feats_hc,2)

    h_fig = figure(iter_feat);
    h_fig.Position = [0 0 800 600];
    h_fig.Color = [1 1 1];
    hold on
    
    count_x = 0;
    for target_ = [9 10 11]

        for correct_ = [1 0]
            
            count_x = count_x+1;

            idxs_thisline = intersect(find(lines_test < 13),...
                intersect(find(targets_test == target_),find(corrects_test == correct_))...
                );
 
            
            y_ = feats_hc(idxs_thisline,iter_feat);
            x_ = xs(count_x);
    
            bar(x_,mean(y_),'LineWidth',2,'FaceColor','none','FaceColor', colors_line(find(xs == x_),:));
            errorbar(x_,mean(y_),std(y_,0,1), 'k','LineWidth',2);
        end
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
        ylim([0 inf])
    else
        ylim([0 inf])
    end


    xticks(unique(xs));
    xticklabels(labels_xtick);
    set(gca, 'FontSize', 12)
    title(strrep(names_feat{iter_feat}, '_', ' '));

    saveas(gcf,[dir_feat, '/RA_featvspred_', names_feat{iter_feat}, '.fig']);
    saveas(gcf,[dir_feat, '/RA_featvspred_', names_feat{iter_feat}, '.png']);

end
