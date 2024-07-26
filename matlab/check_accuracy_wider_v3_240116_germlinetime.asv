%% find the best epoch based on line validation

close all
clear
clc
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

save_fig =true;
save_png =true;


dir_out0 = '/data02/gkim/stem_cell_jwshin/outs';
dir_out = [dir_out0 '/' ...
    '23_SEC1H5_wider_v3_allh_GM_germline_fishdeep1_b012_in_lr0.001'];
dir_outi = [dir_out '_test'];

epochs = [];
accs_tr = [];
accs_va = [];
accs_te = [];

cd(dir_out)
list_mdl = dir('epoch*]');

for iter_mdl = 1:length(list_mdl)
    cd(dir_out)

    name_mdl = list_mdl(iter_mdl).name;
    idxs_lbra = strfind(name_mdl,'[');
    epochs = [epochs; str2num(name_mdl(idxs_lbra(1)+1:idxs_lbra(1)+5))];
    accs_tr = [accs_tr; str2num(name_mdl(idxs_lbra(2)+1:idxs_lbra(2)+5))];
    accs_va = [accs_va; str2num(name_mdl(idxs_lbra(3)+1:idxs_lbra(3)+5))];
    accs_te = [accs_te; str2num(name_mdl(idxs_lbra(4)+1:idxs_lbra(4)+5))];


    
    metrics = accs_va;%+accs_tr;
    idx_bestmdl = min(find(metrics == max(metrics)));
    dir_bestmdl = list_mdl(idx_bestmdl).name;
        idxs_lbra = strfind(dir_bestmdl,'[');
    epoch_bestmdl_str = (dir_bestmdl(idxs_lbra(1)+1:idxs_lbra(1)+5));
end

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

%%

preds_test = -ones(size(targets_test));
for iter_data = 1:length(targets_test)
    preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
end
preds_test = preds_test-1;

lines_test = zeros(size(targets_test)); % represents plot location and (potentially) markers 
types_test = zeros(size(targets_test)); % represents the color
for iter_data = 1:length(targets_test)
    fname = paths_test{iter_data};
    idx_slash = strfind(fname,'/');
    fname = fname(idx_slash(end)+1:end);
    if contains(fname, 'GM') || contains(fname, 'gm') || contains(fname, '230719.')  || contains(fname, '230720.') 
        if contains(fname, 'Endo') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 2;
            types_test(iter_data) = 3;%2;
        elseif contains(fname, 'Endo') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 3;
            types_test(iter_data) = 3;
        elseif contains(fname,'Meso') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 4;
            types_test(iter_data) = 5;%4;
        elseif contains(fname,'Meso') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 5;
            types_test(iter_data) = 5;
        elseif contains(fname,'Ecto') && contains(fname,{'12h', '12H'})
            lines_test(iter_data) = 6;
            types_test(iter_data) = 7;%6;
        elseif contains(fname,'Ecto') && contains(fname,{'24h', '24H'})
            lines_test(iter_data) = 7;
            types_test(iter_data) = 7;
        else
            lines_test(iter_data) = 1;
            types_test(iter_data) = 1;
        end

    elseif contains(fname, {'.HD02.', 'HD02_'})
        lines_test(iter_data) = 13 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD03.', 'HD03_'})
        lines_test(iter_data) = 14 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD04.', 'HD04_'})
        lines_test(iter_data) = 15 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD05.', 'HD05_'})
        lines_test(iter_data) = 16 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD07.', 'HD07_'})
        lines_test(iter_data) = 17 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD09.', 'HD09_'})
        lines_test(iter_data) = 18 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD11.', 'HD11_'})
        lines_test(iter_data) = 19 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD12.', 'HD12_'})
        lines_test(iter_data) = 20 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD14.', 'HD14_'})
        lines_test(iter_data) = 21 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD15.', 'HD15_'})
        lines_test(iter_data) = 22 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD18.', 'HD18_'})
        lines_test(iter_data) = 23 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.HD25.', 'HD25_'})
        lines_test(iter_data) = 24 - 4;
        types_test(iter_data) = 8;

    elseif contains(fname, {'.B1.', 'B1_','.BJ01.'})
        lines_test(iter_data) = 26 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.BJ04.', 'BJ04_'})
        lines_test(iter_data) = 27 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B7.', 'B7_'})
        lines_test(iter_data) = 28 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.BJ11.'})
        lines_test(iter_data) = 29 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B12.', 'B12_'})
        lines_test(iter_data) = 30 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B13.', 'B13_'})
        lines_test(iter_data) = 31 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.BJ14.'})
        lines_test(iter_data) = 32 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B17.', 'B17_', '.BJ17.'})
        lines_test(iter_data) = 33 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B18.', 'B18_'})
        lines_test(iter_data) = 34 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B21.', 'B21_'})
        lines_test(iter_data) = 35 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.BJ22.'})
        lines_test(iter_data) = 36 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.B23.', 'B23_'})
        lines_test(iter_data) = 37 - 4;
        types_test(iter_data) = 8;


    elseif contains(fname, {'.A2.', 'A2_'})
        lines_test(iter_data) = 39 - 4;%7;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.A12.', 'A12_'})
        lines_test(iter_data) = 40 - 4;%8;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.A19.', 'A19_'})
        lines_test(iter_data) = 41 - 4;
        types_test(iter_data) = 8;
    elseif contains(fname, {'.A20.', 'A20_'})
        lines_test(iter_data) = 42 - 4;
        types_test(iter_data) = 8;

    else
        continue
        error('No line found')
    end

end
%% plot prediction

colors_type = [
    1 0 1;...
    %     0.5 0.5 0;...
%     0 0.5 0.5;...
    0 1 1;...
%     0.5 0 0.5;...
    1 1 0;... 
    0 0 0;...
    0.5 0.5 0.5];

labels_xtick = {'GM',
        'GM + endo 12h',
        'GM + endo 24h',
        'GM + meso 12h',
        'GM + meso 24h',
        'GM + ecto 12h',
        'GM + ecto 24h',
        'HD 02',
        'HD 03',
        'HD 04',
        'HD 05',
        'HD 07',
        'HD 09',
        'HD 11',
        'HD 12',
        'HD 14',
        'HD 15',
        'HD 18',
        'HD 25',
        'BJ 01',
        'BJ 04',
        'BJ 07',
        'BJ 11',
        'BJ 12',
        'BJ 13',
        'BJ 14',
        'BJ 17',
        'BJ 18',
        'BJ 21',
        'BJ 22',
        'BJ 23',
        'PD 02',%'A2_',
        'PD 12',%'A12_',
        'PD 19',%'A19_',
        'PD 20',%'A20_',};
        };


dir_plot = '/data02/gkim/stem_cell_jwshin/data_present/240122_classification_germlayer';
idx_slash = strfind(dir_out,'/');
dir_plot = [dir_plot dir_out(idx_slash(end):end) '/' dir_bestmdl];

mkdir(dir_plot) 

close all 
h_fig = figure(100);
h_fig.Position = [0 0 1200 600];
h_fig.Color = [1 1 1];
hold on

for line_ = unique(lines_test(lines_test>0))'
        if line_ == 0 
            continue
        end
        idxs_thisline = find(lines_test == line_);

        y_ = preds_test(idxs_thisline);
        x_ = line_;

        num_cummul = [0];
        for target_ = sort(unique(targets_test)', 'ascend')
            %ecto meso endo ctl
            num_cummul = [num_cummul num_cummul(end)+sum(y_==target_)];
        end
        num_cummul(1) = [];
        for target_ = sort(unique(targets_test)', 'descend')
            bar(x_,num_cummul(target_+1),'LineWidth',2,'FaceColor','none','FaceColor', colors_type(target_+1,:));
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

saveas(gcf,[dir_plot, '/', 'prediction', '.fig']);
saveas(gcf,[dir_plot, '/', 'prediction', '.png']);

%% plot prediction_normalized

dir_plot = '/data02/gkim/stem_cell_jwshin/data_present/240122_classification_germlayer';
idx_slash = strfind(dir_out,'/');
dir_plot = [dir_plot dir_out(idx_slash(end):end) '/' dir_bestmdl];

mkdir(dir_plot) 

close all 
h_fig = figure(101);
h_fig.Position = [0 0 1200 600];
h_fig.Color = [1 1 1];
hold on

for line_ = unique(lines_test(lines_test>0))'
        if line_ == 0 
            continue
        end
        idxs_thisline = find(lines_test == line_);

        y_ = preds_test(idxs_thisline);
        x_ = line_;

        num_cummul = [0];
        for target_ = sort(unique(targets_test)', 'ascend')
            %ecto meso endo ctl
            num_cummul = [num_cummul num_cummul(end)+sum(y_==target_)];
        end
        num_cummul(1) = [];
        num_cummul = num_cummul/max(num_cummul);
        for target_ = sort(unique(targets_test)', 'descend')
            bar(x_,num_cummul(target_+1),'LineWidth',2,'FaceColor','none','FaceColor', colors_type(target_+1,:));
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

saveas(gcf,[dir_plot, '/', 'prediction_normalized', '.fig']);
saveas(gcf,[dir_plot, '/', 'prediction_normalized', '.png']);