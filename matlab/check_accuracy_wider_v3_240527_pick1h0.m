%% find the best epoch based on line validation

close all
clear
clc
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

save_fig =true;
save_png =true;


dir_out0 = '/data02/gkim/stem_cell_jwshin/outs';
dir_case = '23_SEC1H5_wider_v3_pick1h0_GM_germline_fishdeep10_b012_in_lr0.001';%%'23_SEC1H5_wider_v3_allh0_GM_germline_fishdeep10_b012_in_lr0.001_ens';
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

epoch_bestmdl_str = '00124'

targets_test = [];
scores_test = [];
feats_test = [];
paths_test = {};
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
            targets_test(iter_data) = 2;
        elseif contains(fname, 'Endo') && contains(fname,{'12h', '12H'})
            continue
            lines_test(iter_data) = 3;
            % types_test(iter_data) = 3;%2;
            targets_test(iter_data) = 3;
        elseif contains(fname, 'Endo') && contains(fname,{'24h', '24H'})
            continue
            lines_test(iter_data) = 4;
            % types_test(iter_data) = 3;
            targets_test(iter_data) = 11;
        elseif contains(fname, 'Endo') && contains(fname,{'48h', '48H'})
            lines_test(iter_data) = 3;
            % types_test(iter_data) = 3;
            targets_test(iter_data) = 3;
        elseif contains(fname, 'Meso') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 4;
            % types_test(iter_data) = 4;%4;
            targets_test(iter_data) = 1;
        elseif contains(fname,'Meso') && contains(fname,{'12h', '12H'})
            continue
            lines_test(iter_data) = 7;
            % types_test(iter_data) = 4;%4;
            targets_test(iter_data) = 6;
        elseif contains(fname,'Meso') && contains(fname,{'24h', '24H'})
            continue
            lines_test(iter_data) = 8;
            % types_test(iter_data) = 4;
            targets_test(iter_data) = 7;
        elseif contains(fname,'Meso') && contains(fname,{'48h', '48H'})
            continue
            lines_test(iter_data) = 9;
            % types_test(iter_data) = 4;
            targets_test(iter_data) = 8;
        elseif contains(fname, 'Ecto') && contains(fname,{'6h', '6H'})
            lines_test(iter_data) = 5;
            % types_test(iter_data) = 5;%4;
            targets_test(iter_data) = 0;
        elseif contains(fname,'Ecto') && contains(fname,{'12h', '12H'})
            continue
            lines_test(iter_data) = 11;
            % types_test(iter_data) = 5;%6;
            targets_test(iter_data) = 1;
        elseif contains(fname,'Ecto') && contains(fname,{'24h', '24H'})
            continue
            lines_test(iter_data) = 12;
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
            lines_test(iter_data) = 7;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 6;
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 6;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 5;
        else
            lines_test(iter_data) = 1;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 4;
        end

    elseif contains(fname, {'H9', 'h9', '230714.'}) 
        if contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 11;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 6;
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 10;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 5;
        else
            lines_test(iter_data) = 9;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 4;
        end

    elseif contains(fname, {'230427', '230713'})%contains(fname, 'JAX', 'Jax', '230713.') 
        % old jax
        continue

    elseif contains(fname,{'JAX','Jax','jax'})
        % late jax
        if contains(fname, {'24h', '24H'})
            lines_test(iter_data) = 15;
            % types_test(iter_data) = 7;
            targets_test(iter_data) = 6;
        elseif contains(fname, {'12h', '12H'})
            lines_test(iter_data) = 14;
            % types_test(iter_data) = 6;
            targets_test(iter_data) = 5;
        else
            lines_test(iter_data) = 13;
            % types_test(iter_data) = 1;
            targets_test(iter_data) = 4;
        end

    % high
    elseif contains(fname, {'.HD09.', 'HD09_'})
        lines_test(iter_data) = 17;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 7;
    elseif contains(fname, {'.HD11.', 'HD11_'})
        lines_test(iter_data) = 18;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 7;
    elseif contains(fname, {'.A2.', 'A2_'})
        lines_test(iter_data) = 19;
        % types_test(iter_data) = 7;
        targets_test(iter_data) = 8;
    elseif contains(fname, {'.A12.', 'A12_'})
        lines_test(iter_data) = 20;
        % types_test(iter_data) = 8;
        targets_test(iter_data) = 7;

    % low

    elseif contains(fname, {'.HD18.', 'HD18_'})
        lines_test(iter_data) = 21;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 8;
    elseif contains(fname, {'.B13.', 'B13_'})
        lines_test(iter_data) = 22;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 8;
    elseif contains(fname, {'.A19.', 'A19_'})
        lines_test(iter_data) = 23;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 8;
    elseif contains(fname, {'.A20.', 'A20_'})
        lines_test(iter_data) = 24;
        % types_test(iter_data) = 9;
        targets_test(iter_data) = 8;


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
%% plot prediction
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

% colors_cls = [
%     1 1 3/4;...
%     1 1 1/2;...
%     1 1 0;...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
%     1 3/4 1;...
%     1 1/2 1;...
%     1 0 1;...
%     0.8 0 0.8;...
%     3/4 1 1;...
%     1/2 1 1;...
%     0 1 1;...
%     0 0.8 0.8;...
%     3/4 3/4 3/4;...
%     ];


colors_cls = [
    1 1 3/4;...
    1 3/4 1;...
    3/4 1 1;...
    0 0.8 0.8;...
    3/4 3/4 3/4;...
    ];



labels_xtick = {'GM untreated',%1
        'GM + endo 6h',%2
%         'GM + endo 12h',%3
%         'GM + endo 24h',%4
        'GM + endo 48h',%5
        'GM + meso 6h',%6
%         'GM + meso 12h',%7
%         'GM + meso 24h',%8
%         'GM + meso 48h',%9
        'GM + ecto 6h',%10
%         'GM + ecto 12h',%11
%         'GM + ecto 24h',%12
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


idx_slash = strfind(dir_out,'/');
dir_plot = [dir_out0 '_present/' dir_case '/' dir_bestmdl];

mkdir(dir_plot) 

close all 
h_fig = figure(100);
h_fig.Position = [0 0 900 600];
h_fig.Color = [1 1 1];
hold on

for line_ = unique(lines_test(lines_test>0))'
        if line_ == 0 
            continue
        end
        idxs_thisline = find(lines_test == line_);

        y_ = preds_test(idxs_thisline);
        x_ = line_;

        y_cummul = length(idxs_thisline);
        
        for target_ = [2 3 1 0 4]%[13 9 10 11 12 5 6 7 8 0 1 2 3 4]
            %13:-1:0%sort(unique(targets_test)', 'ascend')
            bar(x_,y_cummul,'LineWidth',2,'FaceColor','none','FaceColor', colors_cls(target_+1,:));
            y_cummul = y_cummul-sum(y_==target_);
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

xticks(unique(lines_test(lines_test>0)));
xticklabels(labels_xtick);
set(gca, 'FontSize', 12)
title(strrep('prediction', '_', ' '));

saveas(gcf,[dir_plot, '/', 'select_prediction', '.fig']);
saveas(gcf,[dir_plot, '/', 'select_prediction', '.png']);

%% plot prediction_normalized

close all 
h_fig = figure(101);
h_fig.Position = [0 0 900 600];
h_fig.Color = [1 1 1];
hold on
for line_ = unique(lines_test(lines_test>0))'
        if line_ == 0 
            continue
        end
        idxs_thisline = find(lines_test == line_);

        y_ = preds_test(idxs_thisline);
        x_ = line_;

        y_cummul = 1;%length(idxs_thisline);
        
        for target_ = [2 3 1 0 4]%[13 9 10 11 12 5 6 7 8 0 1 2 3 4]
            %12:-1:0%sort(unique(targets_test)', 'ascend')
            bar(x_,y_cummul,'LineWidth',2,'FaceColor','none','FaceColor', colors_cls(target_+1,:));
            y_cummul = y_cummul-sum(y_==target_)/length(idxs_thisline);
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

xticks(unique(lines_test(lines_test>0)));
xticklabels(labels_xtick);
set(gca, 'FontSize', 12)
title('prediction normalized');

saveas(gcf,[dir_plot, '/', 'select_prediction_normalized', '.fig']);
saveas(gcf,[dir_plot, '/', 'select_prediction_normalized', '.png']);



%% confusion matrix within GM + germ layer
close all
conf_GL = zeros(size(scores_test,2),size(scores_test,2));

idx_cls_rearrange = [4 2 3 1 0];%[2 3 1 0 4];%[13 9 10 11 12 5 6 7 8 0 1 2 3 4]

for iter_data = 1:size(preds_test,1)
    if targets_test(iter_data) == -1
        continue
    elseif targets_test(iter_data) > 4
        continue
    end

    conf_GL(targets_test(iter_data)+1,preds_test(iter_data)+1) = ...
        conf_GL(targets_test(iter_data)+1,preds_test(iter_data)+1)+1;
    
end

conf_GL = conf_GL(idx_cls_rearrange+1,idx_cls_rearrange+1);

conf_GL_norm = conf_GL./repmat(sum(conf_GL,2), [1 size(conf_GL,2)]);

figure(55)
imagesc(conf_GL_norm), colormap magma, axis image


%%

lines_RA = [1 6 7 9 10 11 13 14 15];%unique(lines_test);%[26 27 28 29 30 31 32 33];
%lines_RA(lines_RA<8) = [];
%lines_RA(lines_RA>=17) = [];
conf_RA = zeros(length(lines_RA),size(scores_test,2)); 
for iter_data = 1:size(preds_test,1)
    if ~ismember(lines_test(iter_data), lines_RA)
        continue
    end

    conf_RA(find(lines_RA == lines_test(iter_data)),preds_test(iter_data)+1) = ...
        conf_RA(find(lines_RA == lines_test(iter_data)),preds_test(iter_data)+1)+1;
    
end

conf_RA = conf_RA(:,idx_cls_rearrange+1);
conf_RA_norm = conf_RA./repmat(sum(conf_RA,2), [1 size(conf_RA,2)]);

figure(56)
imagesc(conf_RA_norm), colormap magma, axis image



%%

lines_iPSC = unique(lines_test);%[26 27 28 29 30 31 32 33];
lines_iPSC(lines_iPSC<17) = [];
conf_iPSC = zeros(length(lines_iPSC),size(scores_test,2)); 
for iter_data = 1:size(preds_test,1)
    if ~ismember(lines_test(iter_data), lines_iPSC)
        continue
    end

    conf_iPSC(find(lines_iPSC == lines_test(iter_data)),preds_test(iter_data)+1) = ...
        conf_iPSC(find(lines_iPSC == lines_test(iter_data)),preds_test(iter_data)+1)+1;
    
end

conf_iPSC = conf_iPSC(:,idx_cls_rearrange+1);
conf_iPSC_norm = conf_iPSC./repmat(sum(conf_iPSC,2), [1 size(conf_iPSC,2)]);

figure(57)
imagesc(conf_iPSC_norm), colormap magma, axis image