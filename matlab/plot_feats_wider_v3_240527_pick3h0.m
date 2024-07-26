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

epoch_bestmdl_str = '00297'

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

%% spider_plot

close all

colors_line = [
    3/4 3/4 3/4;...
    1-(1-[255/255 192/255 0/255])*1/3;...
    1-(1-[255/255 192/255 0/255])*2/3;...
    [255/255 192/255 0/255];...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    1-(1-[237/255 125/255 49/255])*1/3;...
    1-(1-[237/255 125/255 49/255])*2/3;...
    237/255 125/255 49/255;...
%     0.8 0 0.8;...
%     0 0.8 0.8;...
    1-(1-[0/255 176/255 80/255])*1/3;...
    1-(1-[0/255 176/255 80/255])*2/3;...
    0/255 176/255 80/255;...
    ];

feats_hc_spider = [];

idxs_feat_plot = [3 20 32 16 15 33 21 34 35];
for idx_line = 1:10
    idxs_thisline = find(lines_test == idx_line);
    feats_hc_ = feats_hc(idxs_thisline,idxs_feat_plot);

    feats_hc_spider = [feats_hc_spider; mean(feats_hc_,1)];

end
spider_plot_R2019b(feats_hc_spider, 'Color', colors_line, 'AxesLabels', names_feat(idxs_feat_plot))
cd(dir_feat)
saveas(gcf,'spider_select.fig')
saveas(gcf,'spider_select.png')



%% plot features

% 
% colors_type = [
%     1 0 1;...
%     %     0.5 0.5 0;...
% %     0 0.5 0.5;...
%     0 1 1;...
% %     0.5 0 0.5;...
%     1 1 0;... 
%     0 0 0;...
%     0.5 0.5 0.5];




xs = [1 3 4 5 7 8 9 11 12 13];


labels_xtick = {'GM untreated',%1
        'GM + endo 6h',%2
        'GM + endo 12h',%3
        'GM + endo 24h',%4
%         'GM + endo 48h',%5
        'GM + meso 6h',%6
        'GM + meso 12h',%7
        'GM + meso 24h',%8
%         'GM + meso 48h',%9
        'GM + ecto 6h',%10
        'GM + ecto 12h',%11
        'GM + ecto 24h',%12
%         'GM + ecto 48h',%13
%         'GM + ecto 72h',%14
        'GM + RA 12h',%15
        'GM + RA 24h',%16


        'H9 untreated',%18
        'H9 + RA 12h',%19
        'H9 + RA 24h',%20

        'JAX untreated',%22
        'JAX + RA 12h',%23
        'JAX + RA 24h',%24


        'HD 09',%26
        'HD 11',%27
        'PD 02',%28
        'PD 12',%29

        'HD 18',%30
        'BJ 13',%31
        'PD 19',%32
        'PD 20',%33
        
        };

labels_xtick = {'Control',
        'Endo 6h',
        'Endo 12h',
        'Endo 24h',
        'Meso 6h',
        'Meso 12h',
        'Meso 24h',
        'Ecto 6h',
        'Ecto 12h',
        'Ecto 24h'
        };


% dir_plot = dir_feat;
% mkdir(dir_plot)

for iter_feat = 1:size(feats_hc,2)

    h_fig = figure(iter_feat);
    h_fig.Position = [0 0 1200 600];
    h_fig.Color = [1 1 1];
    hold on
    for line_ = 1:10
        idxs_thisline = find(lines_test == line_);

        y_ = feats_hc(idxs_thisline,iter_feat);
        x_ = xs(line_);

        bar(x_,mean(y_),'LineWidth',2,'FaceColor','none','FaceColor', colors_line(line_,:));
        errorbar(x_,mean(y_),std(y_,0,1), 'k','LineWidth',2);

        
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
        ylim([0, inf])
    else
        ylim([0 inf])
    end


    xticks(unique(xs));
    xticklabels(labels_xtick);
    set(gca, 'FontSize', 12)
    title(strrep(names_feat{iter_feat}, '_', ' '));

    saveas(gcf,[dir_feat, '/GLGM_', names_feat{iter_feat}, '.fig']);
    saveas(gcf,[dir_feat, '/GLGM_', names_feat{iter_feat}, '.png']);

end

%%%
idx_data_include = 1:length(paths_test);
numsamp_max = inf;
if ~(exist('feats_hc_tsne') && ~isempty(feats_hc_tsne))

feats_hc_tsne = {};
perps = [32 64 128 256];% 256];
lr = 250;

    for perp = perps
    
        feats_hc_tsne_ = tsne(feats_hc(idx_data_include,:), 'Perplexity', perp, 'Standardize',true,'LearnRate',lr);
        feats_hc_tsne{end+1} = feats_hc_tsne_;
    
    end
    save([dir_feat '/hc_tsne.mat'], 'feats_hc_tsne', 'perps', 'idx_data_include', 'lr', 'numsamp_max');
end


%% plot t-sne: only GM GLs, train &val vs test

close all

colors_line = [
    3/4 3/4 3/4;...
    1-(1-[255/255 192/255 0/255])*1/3;...
    1-(1-[255/255 192/255 0/255])*2/3;...
    [255/255 192/255 0/255];...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    1-(1-[237/255 125/255 49/255])*1/3;...
    1-(1-[237/255 125/255 49/255])*2/3;...
    237/255 125/255 49/255;...
%     0.8 0 0.8;...
%     0 0.8 0.8;...
    1-(1-[0/255 176/255 80/255])*1/3;...
    1-(1-[0/255 176/255 80/255])*2/3;...
    0/255 176/255 80/255;...
    ];

[colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');

for idx_colors = 1:size(colors_line_,1)
    lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
end

for perp = perps
num_dot = 0;
for line_ = 1:10
    if line_ > 10
        continue
    end

    color_plot = colors_line_(line_,:);
%     if sum(color_plot) > 3
%         continue
%     end
% 
%     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
% 
%     idx_data = [];
%     for idx_line = 1:length(lines_plot)
%         line_temp = lines_plot(idx_line);
%         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
%     end
    idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
    
num_dot = num_dot+length(idx_data);

    marker_ = 'o';

    h_fig = figure(perp + 2000);
    h_fig.Position = [0 0 900 900];
    h_fig.Color = [1 1 1];
    hold on

    size_ = 66;
    feats_tsne_ = feats_hc_tsne{find(perp==perps)};
    scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
    'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
    'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
%     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
%     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    

end

ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));

    ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
    xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);

set(gcf,'color',[1 1 1])
axis image
axis off
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d.fig', perp)])
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d.png', perp)])

end


%% plot t-sne: only GM GLs, train &val vs test - notime

close all

colors_line = [
    3/4 3/4 3/4;...
    [255/255 192/255 2/255];...
    [255/255 192/255 1/255];...
    [255/255 192/255 0/255];...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    237/255 125/255 41/255;...
    237/255 125/255 50/255;...
    237/255 125/255 49/255;...
%     0.8 0 0.8;...
%     0 0.8 0.8;...
    2/255 176/255 80/255;...
    1/255 176/255 80/255;...
    0/255 176/255 80/255;...
    ];

[colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');

for idx_colors = 1:size(colors_line_,1)
    lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
end

for perp = perps
num_dot = 0;
for line_ = 1:10
    if line_ > 10
        continue
    end

    color_plot = colors_line_(line_,:);
%     if sum(color_plot) > 3
%         continue
%     end
% 
%     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
% 
%     idx_data = [];
%     for idx_line = 1:length(lines_plot)
%         line_temp = lines_plot(idx_line);
%         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
%     end
    idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
    
num_dot = num_dot+length(idx_data);

    marker_ = 'o';

    h_fig = figure(perp + 2000);
    h_fig.Position = [0 0 900 900];
    h_fig.Color = [1 1 1];
    hold on

    size_ = 66;
    feats_tsne_ = feats_hc_tsne{find(perp==perps)};
    scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
    'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
    'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
%     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
%     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    

end

ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));

    ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
    xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);

set(gcf,'color',[1 1 1])
axis image
axis off
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_notime.fig', perp)])
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_notime.png', perp)])

end

%% plot t-sne: only GM GLs, train &val vs test - only endo

close all

colors_line = [
    3/4 3/4 3/4;...
    1-(1-[255/255 192/255 0/255])*1/3;...
    1-(1-[255/255 192/255 0/255])*2/3;...
    [255/255 192/255 0/255];...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    3/4 3/4 3/4+0.01;...
    3/4 3/4 3/4+0.02;...
    3/4 3/4 3/4+0.03;...
%     0.8 0 0.8;...
%     0 0.8 0.8;...
    3/4 3/4 3/4+0.04;...
    3/4 3/4 3/4+0.05;...
    3/4 3/4 3/4+0.06;...
    ];

[colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');

for idx_colors = 1:size(colors_line_,1)
    lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
end

for perp = perps
num_dot = 0;
for line_ = [1 5 6 7 8 9 10 2 3 4]
    if line_ > 10
        continue
    end

    color_plot = colors_line_(line_,:);
%     if sum(color_plot) > 3
%         continue
%     end
% 
%     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
% 
%     idx_data = [];
%     for idx_line = 1:length(lines_plot)
%         line_temp = lines_plot(idx_line);
%         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
%     end
    idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
    
num_dot = num_dot+length(idx_data);

    marker_ = 'o';

    h_fig = figure(perp + 2000);
    h_fig.Position = [0 0 900 900];
    h_fig.Color = [1 1 1];
    hold on

    size_ = 66;
    feats_tsne_ = feats_hc_tsne{find(perp==perps)};
    scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
    'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
    'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
%     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
%     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    

end

ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));

    ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
    xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);

set(gcf,'color',[1 1 1])
axis image
axis off
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_endo.fig', perp)])
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_endo.png', perp)])

end

%% plot t-sne: only GM GLs, train &val vs test - only Meso

close all

colors_line = [
    3/4 3/4 3/4;...
    3/4 3/4 3/4+0.01;...
    3/4 3/4 3/4+0.02;...
    3/4 3/4 3/4+0.03;...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    1-(1-[237/255 125/255 49/255])*1/3;...
    1-(1-[237/255 125/255 49/255])*2/3;...
    237/255 125/255 49/255;...
%     0.8 0 0.8;...
%     0 0.8 0.8;...
    3/4 3/4 3/4+0.04;...
    3/4 3/4 3/4+0.05;...
    3/4 3/4 3/4+0.06;...
    ];

[colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');

for idx_colors = 1:size(colors_line_,1)
    lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
end

for perp = perps
num_dot = 0;
for line_ = [1 2 3 4 8 9 10 5 6 7 ]
    if line_ > 10
        continue
    end

    color_plot = colors_line_(line_,:);
%     if sum(color_plot) > 3
%         continue
%     end
% 
%     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
% 
%     idx_data = [];
%     for idx_line = 1:length(lines_plot)
%         line_temp = lines_plot(idx_line);
%         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
%     end
    idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
    
num_dot = num_dot+length(idx_data);

    marker_ = 'o';

    h_fig = figure(perp + 2000);
    h_fig.Position = [0 0 900 900];
    h_fig.Color = [1 1 1];
    hold on

    size_ = 66;
    feats_tsne_ = feats_hc_tsne{find(perp==perps)};
    scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
    'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
    'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
%     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
%     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    

end

ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));

    ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
    xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);

set(gcf,'color',[1 1 1])
axis image
axis off
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_meso.fig', perp)])
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_meso.png', perp)])

end

%% plot t-sne: only GM GLs, train &val vs test - only Ecto

close all

colors_line = [
    3/4 3/4 3/4;...
    3/4 3/4 3/4+0.01;...
    3/4 3/4 3/4+0.02;...
    3/4 3/4 3/4+0.03;...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    3/4 3/4 3/4+0.04;...
    3/4 3/4 3/4+0.05;...
    3/4 3/4 3/4+0.06;...
%     0.8 0 0.8;...
%     0 0.8 0.8;...
    1-(1-[0/255 176/255 80/255])*1/3;...
    1-(1-[0/255 176/255 80/255])*2/3;...
    0/255 176/255 80/255;...
    ];

[colors_line_,idx_clr2cls,idx_line2clr] = unique(colors_line, 'row', 'stable');

for idx_colors = 1:size(colors_line_,1)
    lines = find(ismember(colors_line, colors_line_(idx_colors,:), 'rows'));
end

for perp = perps
num_dot = 0;
for line_ = [1 2 3 4 5 6 7 8 9 10 ]
    if line_ > 10
        continue
    end

    color_plot = colors_line_(line_,:);
%     if sum(color_plot) > 3
%         continue
%     end
% 
%     lines_plot= find(ismember(colors_line, color_plot , 'rows'))
% 
%     idx_data = [];
%     for idx_line = 1:length(lines_plot)
%         line_temp = lines_plot(idx_line);
%         idx_data = [idx_data; find(lines_test(idx_data_include) == line_temp)];
%     end
    idx_data = find(lines_test== line_); % find(lines_test(idx_data_include) == line_);
    
num_dot = num_dot+length(idx_data);

    marker_ = 'o';

    h_fig = figure(perp + 2000);
    h_fig.Position = [0 0 900 900];
    h_fig.Color = [1 1 1];
    hold on

    size_ = 66;
    feats_tsne_ = feats_hc_tsne{find(perp==perps)};
    scatter(feats_tsne_(idx_data,1),feats_tsne_(idx_data,2),marker_,...
    'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
    'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
%     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
%     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    

end

ycen = min(feats_tsne_(:,2))/2+max(feats_tsne_(:,2))/2;
xcen = min(feats_tsne_(:,1))/2+max(feats_tsne_(:,1))/2;
yax = max(feats_tsne_(:,2))-min(feats_tsne_(:,2));
xax = max(feats_tsne_(:,1))-min(feats_tsne_(:,1));

    ylim([min(feats_tsne_(:,2)) max(feats_tsne_(:,2))]);
    xlim([min(feats_tsne_(:,1)) max(feats_tsne_(:,1))]);

set(gcf,'color',[1 1 1])
axis image
axis off
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_ecto.fig', perp)])
saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_ecto.png', perp)])

end
%% plot with hc feat
close all

names_feat_hc = {'volume_colony','area_colony','RI_avg_colony','RI_std_colony',...
                'volume_lip','area_lip','RI_avg_lip','RI_std_lip',...# volume_cyto = volume_colony-volume_lip
                'volume_cyto','area_cyto','RI_avg_cyto','RI_std_cyto',...
                'area_gap','len_bound','roundness','solidity','eccentricity',...
                'n_boundin','n_boundout','ncont_bound',...
                'thick_avg','thick_std', 'spread_thick', 'skew_thick', 'kurt_thick',...
                'spread_dm', 'skew_dm', 'kurt_dm',...
                'spread_lip', 'skew_lip', 'kurt_lip',...
                'ratio_area_gap', 'ratio_volume_lip'};

feats_hc(:,end+1) = feats_hc(:,13)./feats_hc(:,2);
feats_hc(:,end+1) = feats_hc(:,5)./feats_hc(:,1);
%%

perp_set = 64;

perp = perp_set;

cmap = ones(256,3);
cmap(1:193,2) = 1:-1/192:0;
cmap(193:256,2) = 0;
cmap(192:256,1)=1:-1/192:2/3;
cmap(192:256,3)=1:-1/192:2/3;

% cmap = jet;

for idx_feat_hc =[3 15 16 20 21 23 25 32 33]% [1 3 5 13 15 16 20 21 24 25 30 32 33] %[1 3 13 15 16 ] %1:length(names_feat_hc)

    close all
    if isfile([dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s.fig', perp, names_feat_hc{idx_feat_hc})])
        continue
    end
    feats_hc_ = feats_hc(:,idx_feat_hc);
    mean_hc_ = mean(feats_hc_);
    std_hc_ = std(feats_hc_);

    feats_hc_ = (feats_hc_-mean_hc_)/(std_hc_);
    feats_hc_(feats_hc_<-2) = -2;
    feats_hc_(feats_hc_>2) = 2;

    h_fig = figure(idx_feat_hc);
    for idx_data = 1:length(feats_hc)
        if lines_test(idx_data) > 10 %filter out none-GMGL
            continue
        end
        if lines_test(idx_data) == 0 %filter out none-GMGL
            continue
        end
        marker_ = 'o';
        color_plot = cmap(round((feats_hc_(idx_data)+2)/4*255+1),:);

        set(0,'CurrentFigure',h_fig)
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_hc_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_hc_tsne_(idx_data,1),feats_hc_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    end
    ycen = min(feats_hc_tsne_(:,2))/2+max(feats_hc_tsne_(:,2))/2;
    xcen = min(feats_hc_tsne_(:,1))/2+max(feats_hc_tsne_(:,1))/2;
    yax = max(feats_hc_tsne_(:,2))-min(feats_hc_tsne_(:,2));
    xax = max(feats_hc_tsne_(:,1))-min(feats_hc_tsne_(:,1));
    
        ylim([min(feats_hc_tsne_(:,2)) max(feats_hc_tsne_(:,2))]);
        xlim([min(feats_hc_tsne_(:,1)) max(feats_hc_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s.fig', perp, names_feat_hc{idx_feat_hc})])
    saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s.png', perp, names_feat_hc{idx_feat_hc})])

end


%% plot with 10% sample
close all

perp_set = 64;

perp = perp_set;

cmap = ones(256,3);
cmap(1:193,2) = 1:-1/192:0;
cmap(193:256,2) = 0;
cmap(192:256,1)=1:-1/192:2/3;
cmap(192:256,3)=1:-1/192:2/3;

% cmap = jet;

for idx_feat_hc = [3 20 32 16 15 33 21 34 35] %[1 3 13 15 16 ] %1:length(names_feat_hc)

    close all
    if isfile([dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s_s0.25.fig', perp, names_feat_hc{idx_feat_hc})])
        continue
    end
    feats_hc_ = feats_hc(:,idx_feat_hc);
    mean_hc_ = mean(feats_hc_);
    std_hc_ = std(feats_hc_);

    feats_hc_ = (feats_hc_-mean_hc_)/(std_hc_);
    feats_hc_(feats_hc_<-2) = -2;
    feats_hc_(feats_hc_>2) = 2;

    h_fig = figure(idx_feat_hc);%
    for idx_data = 1:length(feats_hc)
        if mod(idx_data,4) ~= 1
            continue
        end
        if lines_test(idx_data) > 10 %filter out none-GMGL
            continue
        end
        if lines_test(idx_data) ==0 %filter out none-GMGL
            continue
        end
        marker_ = 'o';
        color_plot = cmap(round((feats_hc_(idx_data)+2)/4*255+1),:);

        set(0,'CurrentFigure',h_fig)
        h_fig.Position = [0 0 900 900];
        h_fig.Color = [1 1 1];
        hold on
    
        size_ = 66;
        feats_hc_tsne_ = feats_hc_tsne{find(perp==perps)};
        scatter(feats_hc_tsne_(idx_data,1),feats_hc_tsne_(idx_data,2),marker_,...
        'SizeData', size_, 'MarkerEdgeColor', [1 1 1], 'MarkerFaceColor', color_plot,...
        'MarkerFaceAlpha', 1,'MarkerEdgeAlpha', 1)
    %     plot(feats_tsne_(idxs_thisline,1),feats_tsne_(idxs_thisline,2), marker_,...
    %     'MarkerSize', size_, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', color_)
    end
    ycen = min(feats_hc_tsne_(:,2))/2+max(feats_hc_tsne_(:,2))/2;
    xcen = min(feats_hc_tsne_(:,1))/2+max(feats_hc_tsne_(:,1))/2;
    yax = max(feats_hc_tsne_(:,2))-min(feats_hc_tsne_(:,2));
    xax = max(feats_hc_tsne_(:,1))-min(feats_hc_tsne_(:,1));
    
        ylim([min(feats_hc_tsne_(:,2)) max(feats_hc_tsne_(:,2))]);
        xlim([min(feats_hc_tsne_(:,1)) max(feats_hc_tsne_(:,1))]);
    
    set(gcf,'color',[1 1 1])
    axis image
    axis off
    saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s_s0.25.fig', perp, names_feat_hc{idx_feat_hc})])
    saveas(h_fig, [dir_feat '/' sprintf('tsne_onlyGLGM_hc_p%03d_%s_s0.25.png', perp, names_feat_hc{idx_feat_hc})])

end


%% plot relative table
feats_hc_norm = normalize(feats_hc,1);
idxs_feat_select = 1:size(feats_hc_norm,2);
% idxs_feat_select = [3 20 33 16 15 32 21 23 35];


%%
lines_select = [1 2 3 4 5 6 7 8 9 10];

table_feat = zeros(length(idxs_feat_select), length(lines_select));


for iter_feat_ = 1:length(idxs_feat_select)
    idx_feat_ = idxs_feat_select(iter_feat_);
for line_ = lines_select
    idxs_thisline = find(lines_test == line_);
    feats_hc_ = feats_hc_norm(idxs_thisline,idx_feat_);
    mean_feats = mean(feats_hc_);
    table_feat(iter_feat_,line_) = mean_feats;

end
end

table_feat_norm = table_feat;

for iter_feat_ = 1:length(idxs_feat_select)
    idx_feat_ = idxs_feat_select(iter_feat_);
    row_feat = table_feat_norm(iter_feat_, :);
    row_feat = (row_feat-min(row_feat))/(max(row_feat)-min(row_feat));
    table_feat_norm(iter_feat_, :) = row_feat;
end


%% sort

diff_feat = mean(feats_hc_norm(find(lines_test == 1),idxs_feat_select),1) - ...
    mean(feats_hc_norm(find(ismember(lines_test,2:10)),idxs_feat_select),1);

[vals_sort, idxs_sort_diff] = sort(diff_feat, 'descend');
% idxs_feat_select_sort = idxs_feat_select(idxs_sort);

[vals_sort, idxs_sort_val] = sort(table_feat_norm(:,1),'descend');

%%
table_feat_norm_sort = table_feat_norm(idxs_sort_val,:);
idxs_feat_select_sort = idxs_feat_select(idxs_sort_val);
figure(532),imagesc(table_feat_norm_sort), colormap(cmap), axis image

 saveas(gcf, [dir_feat '/' sprintf('table_feat_select.fig')])
    saveas(gcf, [dir_feat '/' sprintf('table_feat_select.png')])
%%
lines_endo = [1 2 3 4]; % endo
lines_meso = [1 5 6 7]; % meso
lines_ecto = [1 8 9 10]; % ecto

names_feat_hc{idxs_sort_val(idxs_feat_plot_sort)}
idxs_feat_plot = [3 20 32 16 15 33 21 34 35];

idxs_feat_plot_sort = find(ismember(idxs_feat_select_sort,idxs_feat_plot));
figure(533),imagesc(table_feat_norm_sort(idxs_feat_plot_sort ,lines_endo)), colormap(cmap), axis image
 saveas(gcf, [dir_feat '/' sprintf('table_feat_select_endo.fig')])
    saveas(gcf, [dir_feat '/' sprintf('table_feat_select_endo.png')])
figure(534),imagesc(table_feat_norm_sort(idxs_feat_plot_sort,lines_meso)), colormap(cmap), axis image
 saveas(gcf, [dir_feat '/' sprintf('table_feat_select_meso.fig')])
    saveas(gcf, [dir_feat '/' sprintf('table_feat_select_meso.png')])
figure(535),imagesc(table_feat_norm_sort(idxs_feat_plot_sort,lines_ecto)), colormap(cmap), axis image
 saveas(gcf, [dir_feat '/' sprintf('table_feat_select_ecto.fig')])
    saveas(gcf, [dir_feat '/' sprintf('table_feat_select_ecto.png')])