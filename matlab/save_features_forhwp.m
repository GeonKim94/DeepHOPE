%% find the best epoch based on line validation

close all
clear
clc
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

save_fig =true;
save_png =true;


dir_feat = '/data02/gkim/stem_cell_jwshin/data/23_feat_wider_v3_allh_onRA';
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

epoch_bestmdl_str = '00235'

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
linenames_test ={};
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
            targets_test(iter_data) = 7;
            linenames_test{iter_data} = 'GM endo 6 h';
        elseif contains(fname, 'Endo') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 3;
            % types_test(iter_data) = 3;%2;
            targets_test(iter_data) = 8;
            linenames_test{iter_data} = 'GM endo 12 h';
        elseif contains(fname, 'Endo') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 4;
            % types_test(iter_data) = 3;
            targets_test(iter_data) = 9;
            linenames_test{iter_data} = 'GM endo 24 h';
        elseif contains(fname, 'Endo') && contains(fname,{'48h', '48H'})
            lines_test(iter_data) = 5;
            % types_test(iter_data) = 3;
            targets_test(iter_data) = 10;
            linenames_test{iter_data} = 'GM endo 48 h';
        elseif contains(fname, 'Meso') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 6;
            % types_test(iter_data) = 4;%4;
            targets_test(iter_data) = 3;
            linenames_test{iter_data} = 'GM meso 6 h';
        elseif contains(fname,'Meso') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 7;
            % types_test(iter_data) = 4;%4;
            targets_test(iter_data) = 4;
            linenames_test{iter_data} = 'GM meso 12 h';
        elseif contains(fname,'Meso') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 8;
            % types_test(iter_data) = 4;
            targets_test(iter_data) = 5;
            linenames_test{iter_data} = 'GM meso 24 h';
        elseif contains(fname,'Meso') && contains(fname,{'48h', '48H'})
            lines_test(iter_data) = 9;
            % types_test(iter_data) = 4;
            targets_test(iter_data) = 6;
            linenames_test{iter_data} = 'GM meso 48 h';
        elseif contains(fname, 'Ecto') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 10;
            % types_test(iter_data) = 5;%4;
            targets_test(iter_data) = 0;
            linenames_test{iter_data} = 'GM ecto 6 h';
        elseif contains(fname,'Ecto') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 11;
            % types_test(iter_data) = 5;%6;
            targets_test(iter_data) = 1;
            linenames_test{iter_data} = 'GM ecto 12 h';
        elseif contains(fname,'Ecto') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 12;
            % types_test(iter_data) = 5;
            targets_test(iter_data) = 2;
            linenames_test{iter_data} = 'GM ecto 24 h';
        elseif contains(fname,'Ecto') && contains(fname,{'48h', '48H'})
    
            lines_test(iter_data) = 0;
            % types_test(iter_data) = 5;
            targets_test(iter_data) = 3;
            linenames_test{iter_data} = 'GM ecto 48 h';
        elseif contains(fname,'Ecto') && contains(fname,{'72h', '72H'})
            
            lines_test(iter_data) = 0;
            % types_test(iter_data) = 5;
            targets_test(iter_data) = 4;
            linenames_test{iter_data} = 'GM ecto 72 h';
        elseif contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 14;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 13;
            linenames_test{iter_data} = 'GM RA 24 h';
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 13;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 12;
            linenames_test{iter_data} = 'GM RA 12 h';
        else
            lines_test(iter_data) = 1;
            targets_test(iter_data) = 11;
            linenames_test{iter_data} = 'GM ctl';
        end

    elseif contains(fname, {'H9', 'h9', '230714.'}) 
        if contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 18;
            % types_testd(iter_data) = 7;
            targets_test(iter_data) = 13;
            linenames_test{iter_data} = 'H9 24 h';
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 17;
            targets_test(iter_data) = 12;
            linenames_test{iter_data} = 'H9 12 h';
        else
            lines_test(iter_data) = 16;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 11;
            linenames_test{iter_data} = 'H9 ctl';
        end

    elseif contains(fname, {'230427', '230713'})%contains(fname, 'JAX', 'Jax', '230713.') 
        % old jax
        continue

    elseif contains(fname,{'JAX','Jax','jax'})
        % late jax
        if contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 22;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 13;
            linenames_test{iter_data} = 'Jax 24 h';
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 21;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 12;
            linenames_test{iter_data} = 'Jax 12 h';
        else
            lines_test(iter_data) = 20;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 11;
            linenames_test{iter_data} = 'Jax ctl';
        end

    % high
    elseif contains(fname, {'.HD09.', 'HD09_'})
        lines_test(iter_data) = 24;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD09';
    elseif contains(fname, {'.HD11.', 'HD11_'})
        lines_test(iter_data) = 25;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD11';
    elseif contains(fname, {'.A2.', 'A2_'})
        lines_test(iter_data) = 26;
        % types_test(iter_data) = 7;
        targets_test(iter_data) = 14;
        linenames_test{iter_data} = 'PD02';
    elseif contains(fname, {'.A12.', 'A12_'})
        lines_test(iter_data) = 27;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 14;
        linenames_test{iter_data} = 'PD12';

    % low

    elseif contains(fname, {'.HD18.', 'HD18_'})
        lines_test(iter_data) = 28;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 15;
        linenames_test{iter_data} = 'HD18';
    elseif contains(fname, {'.B13.', 'B13_'})
        lines_test(iter_data) = 29;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 15;
        linenames_test{iter_data} = 'BJ13';
    elseif contains(fname, {'.A19.', 'A19_'})
        lines_test(iter_data) = 30;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 15;
        linenames_test{iter_data} = 'PD19';
    elseif contains(fname, {'.A20.', 'A20_'})
        lines_test(iter_data) = 31;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 15;
        linenames_test{iter_data} = 'PD20';


 
    elseif contains(fname, {'.HD02.', 'HD02_'})
        lines_test(iter_data) = 31+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD02';
    elseif contains(fname, {'.HD03.', 'HD03_'})
        lines_test(iter_data) = 32+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD03';
    elseif contains(fname, {'.HD04.', 'HD04_'})
        lines_test(iter_data) = 33+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD04';
    elseif contains(fname, {'.HD05.', 'HD05_'})
        lines_test(iter_data) = 34+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD06';
    elseif contains(fname, {'.HD07.', 'HD07_'})
        lines_test(iter_data) = 35+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD07';
    elseif contains(fname, {'.HD12.', 'HD12_'})
        lines_test(iter_data) = 36+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD12';
    elseif contains(fname, {'.HD14.', 'HD14_'})
        lines_test(iter_data) = 37+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD14';
    elseif contains(fname, {'.HD15.', 'HD15_'})
        lines_test(iter_data) = 38+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD15';
    elseif contains(fname, {'.HD25.', 'HD25_'})
        lines_test(iter_data) = 39+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'HD25';

    elseif contains(fname, {'.B1.', 'B1_','.BJ01.'})
        lines_test(iter_data) = 41+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ01';
    elseif contains(fname, {'.BJ04.', 'BJ04_'})
        lines_test(iter_data) = 42+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ04';
    elseif contains(fname, {'.B7.', 'B7_'})
        lines_test(iter_data) = 43+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ07';
    elseif contains(fname, {'.BJ11.'})
        lines_test(iter_data) = 44+2;
        types_test(iter_data) = 6;
        linenames_test{iter_data} = 'BJ11';
    elseif contains(fname, {'.B12.', 'B12_'})
        lines_test(iter_data) = 45+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ12';
    elseif contains(fname, {'.BJ14.'})
        lines_test(iter_data) = 46+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ14';
    elseif contains(fname, {'.B17.', 'B17_', '.BJ17.'})
        lines_test(iter_data) = 47+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ17';
    elseif contains(fname, {'.B18.', 'B18_'})
        lines_test(iter_data) = 48+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ18';
    elseif contains(fname, {'.B21.', 'B21_'})
        lines_test(iter_data) = 49+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ21';
    elseif contains(fname, {'.BJ22.'})
        lines_test(iter_data) = 50+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ22';
    elseif contains(fname, {'.B23.', 'B23_'})
        lines_test(iter_data) = 51+2;
        types_test(iter_data) = 14;
        linenames_test{iter_data} = 'BJ23';



    else
        continue
        error('No line found')
    end

end

%%
feats_hc = [];
for iter_data = 1:length(paths_test)

    path_data = paths_test{iter_data};
    [dir_data, fileName, fileExt] = fileparts(path_data);

    [dir_find, found] = search_recursive(dir_feat, fileName,false);

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
                thick_avg/sqrt(area_colony)];
    feats_hc = [feats_hc; feat_hc];
    
end

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
                'aspect_ratio_xz'};
fnames_test = paths_test;
for iter_data = 1:length(fnames_test)

    path_temp = fnames_test{iter_data};
    [filepath,fname,ext] = fileparts(path_temp);
    fnames_test{iter_data} = [fname ext];

end

%%

t1 = cell2table(fnames_test, 'VariableNames', {'file name'});
t2 = cell2table(linenames_test', 'VariableNames', {'line name'});
t3 = array2table(sets_test, 'VariableNames', {'is test'});
t4 = array2table(feats_hc,'VariableNames', names_feat);

t_all = [t1 t2 t3 t4];%t_all = join(join(t1,t2),join(t3,t4));
save('/data02/gkim/stem_cell_jwshin/data/23_feat_wider_v3_allh_onRA/table_handcrafted_features.mat', 't_all')
writetable(t_all, '/data02/gkim/stem_cell_jwshin/data/23_feat_wider_v3_allh_onRA/table_handcrafted_features.csv')