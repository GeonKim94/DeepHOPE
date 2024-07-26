%% find the best epoch based on line validation

close all
clear
clc
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

addpath('/data02/gkim/Bacteria_Classification/src/codes_matlab/')

addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
addpath('/data01/gkim/Matlab_subcodes/AllInFocus/')

save_fig =true;
save_png =true;


dir_out0 = '/workspace01/gkim/stem_cell_jwshin/outs';
%dir_out0 = '/data02/gkim/stem_cell_jwshin/outs';
dir_case = '23_SEC1H5_wider_v3_pick3h0_GM_germline_fishdeep10_b012_in_lr0.001';%%'23_SEC1H5_wider_v3_allh0_GM_germline_fishdeep10_b012_in_lr0.001_ens';
dir_out = [dir_out0 '/' ...
    dir_case];
dir_outr = [dir_out '_testRA'];
dir_outi = [dir_out '_testiPSC'];
dir_outf = [dir_out '_testfl'];

epochs = [];
accs_tr = [];
accs_va = [];
accs_te = [];
accs_trw = [];
accs_vaw = [];
accs_tew = [];


cd(dir_out)
list_mdl = dir('epoch*');

for iter_mdl = 1:length(list_mdl)
    cd(dir_out)

    name_mdl = list_mdl(iter_mdl).name;
    idxs_lbra = strfind(name_mdl,'[');
    epochs = [epochs; str2num(name_mdl(idxs_lbra(1)+1:idxs_lbra(1)+5))];
    accs_tr = [accs_tr; str2num(name_mdl(idxs_lbra(2)+1:idxs_lbra(2)+5))];
    accs_va = [accs_va; str2num(name_mdl(idxs_lbra(3)+1:idxs_lbra(3)+5))];
    accs_te = [accs_te; str2num(name_mdl(idxs_lbra(4)+1:idxs_lbra(4)+5))];
    accs_trw = [accs_trw; str2num(name_mdl(idxs_lbra(5)+1:idxs_lbra(5)+5))];
    accs_vaw = [accs_vaw; str2num(name_mdl(idxs_lbra(6)+1:idxs_lbra(6)+5))];
    accs_tew = [accs_tew; str2num(name_mdl(idxs_lbra(7)+1:idxs_lbra(7)+5))];


    
end

metrics = accs_vaw+(accs_vaw-accs_va)*2;
idx_bestmdl = min(find(metrics == max(metrics)));
dir_bestmdl = list_mdl(idx_bestmdl).name;
    idxs_lbra = strfind(dir_bestmdl,'[');
epoch_bestmdl_str = (dir_bestmdl(idxs_lbra(1)+1:idxs_lbra(1)+5));

epoch_bestmdl_str = '00485'
'00639'
'00609'
'00594'
'00357'

'00371'
'00317'
'00356'
'00297'
'00372'
'00395'
'00416'
'00484'
'00485'


targets_test = [];
scores_test = [];
feats_test = [];
paths_test = {};
sets_test = [];
%% get fl test_result

cd(dir_outf)
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
    if contains(fname, {'.HD09.', 'HD09_'})
        lines_test(iter_data) = 1;
        targets_test(iter_data) = 12;
    elseif contains(fname, {'.A20.', 'A20_'})
        lines_test(iter_data) = 2;
        targets_test(iter_data) = 13;
    else
        continue
        error('No line found')
    end

end


%%
colors_cls = [
    1 1 3/4;...
    1 1 1/2;...
    1 1 0;...
%     0.8 0.8 0;...
%     0.6 0.6 0;...
    1 3/4 1;...
    1 1/2 1;...
    1 0 1;...
%     0.8 0 0.8;...
    3/4 1 1;...
    1/2 1 1;...
    0 1 1;...
%     0 0.8 0.8;...
    3/4 3/4 3/4;...
    ];


labels_xtick = {'HD09 + CDy1',
    'PD20 + CDy1'};

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
        
        for target_ = [9 6 7 8 3 4 5 0 1 2]
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

saveas(gcf,[dir_plot, '/', 'select_prediction_CDy1', '.fig']);
saveas(gcf,[dir_plot, '/', 'select_prediction_CDy1', '.png']);
%%

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
        
        for target_ = [9 6 7 8 3 4 5 0 1 2]
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

saveas(gcf,[dir_plot, '/', 'select_prediction_CDy1_normalized', '.fig']);
saveas(gcf,[dir_plot, '/', 'select_prediction_CDy1_normalized', '.png']);



%%

dir_data = '/workspace01/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_testfl';
dir_raw = '/workspace01/gkim/stem_cell_jwshin/data/240716_TCF';
dir_mask = '/workspace01/gkim/stem_cell_jwshin/data/23_mask_wider_v3_allh_onRA';


h_ = figure(1);
h_.Position = [0 100 1600, 800];
h_.Color = [1 1 1];

strs_cls = {'Ctl',...
    'Endo 6 h', 'Endo 12 h', 'Endo 24 h',...
    'Meso 6 h', 'Meso 12 h', 'Meso 24 h',...
    'Ecto 6 h', 'Ecto 12 h', 'Ecto 24 h',...
    };


idx_class_re = [10 7 8 9 4 5 6 1 2 3];
rate_area_CDy1s = [];
for iter_path = 1:length(paths_test)
    
    path_test = paths_test{iter_path};
    [dir_test, fname_test, ext_test] = fileparts(path_test);

    [path_find,found] = search_recursive_v2(dir_raw, [fname_test '.TCF'],false);
    
    hour = 0;


    n_m = 1.337;
    ri = h5read(path_test, '/ri');
%     ri = ReadLDMTCFHT(path_find, hour);
%     ri = single(ri*10000);
%     ri = permute(ri, [2 1 3]);
%     ri(ri == 0) = n_m*10000; 
    

    ri_mip = max(ri,[],3);
    info_temp = h5info(path_find, '/Data/3D');
    size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
    res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

    fl = ReadLDMTCFHT_FLCH0(path_find, hour);
    fl = single(fl);
    fl = permute(fl, [2 1 3]);

    fl_mip = max(fl,[],3);

    maxval_fl = 2000;
    
    info_temp = h5info(path_find, '/Data/3DFL');
    sizefl_ori = [info_temp.Attributes(10).Value info_temp.Attributes(11).Value info_temp.Attributes(12).Value];
    resfl_ori = [info_temp.Attributes(5).Value info_temp.Attributes(6).Value info_temp.Attributes(7).Value];

    size_rs = round((single(sizefl_ori).*resfl_ori)./res_ori);
    fl_mip_rs = imresize(fl_mip, size_rs(1:2));

    [path_find,found] = search_recursive_v2(dir_mask, [fname_test '.mat'],false);
    load(path_find);

    fl_mip_rs_crop = fl_mip_rs(x_crop_(1):x_crop_(2), y_crop_(1):y_crop_(2));


    mask_fl = fl_mip_rs_crop>500;
    mask_fl = mask_fl.*mask_cyto;


    img_ol0 = overlayimg4(...            
            ri_mip,...
            fl_mip_rs_crop,...%zeros(size(ri_rs_,1),size(ri_rs_,2)),...%
            zeros(size(ri_mip,1),size(ri_mip,2)),...
            zeros(size(ri_mip,1),size(ri_mip,2)),...
            13300,0,0,0,13800, maxval_fl, 10, 10);

    if size(img_ol0,1)<size(img_ol0,2)
        img_ol0 = permute(img_ol0,[2 1 3]);
        mask_cyto = mask_cyto';
        mask_fl = mask_fl';
    end

    rate_area_CDy1 = sum(sum(mask_fl))/sum(sum(mask_cyto));

    set(0,'CurrentFigure', h_)
    ax = subplot(1,2,1);
    imshow(img_ol0),axis image

    hold on, plot([size(img_ol0,2)*0.5/8 size(img_ol0,2)*0.5/8+20/res_ori(1)],...
        [size(img_ol0,1)*7.5/8 size(img_ol0,1)*7.5/8],'y',...
        'LineWidth', 4)
    title(sprintf('%s - RI (gray) & CDy1 FL (red)',fname_test))
    hold off

    ax = subplot(2,2,2);
    scores_ = scores_test(iter_path,idx_class_re);
    b = bar(scores_);
    b.FaceColor = 'flat';
    b.CData = colors_cls(idx_class_re,:);
    ylim([-15 15])
    ax.XTick = [1:10];
    ax.XTickLabels = strs_cls;
    title(sprintf('Prediction result: %s', strs_cls{find(scores_ == max(scores_))}))
    
    ax = subplot(2,4,7);
    imagesc(mask_cyto+mask_fl, [0 2]), axis image, axis off
    ax.Colormap = [0 0 0;
        1 1 1
        1 0 0];
    hold on, plot([size(img_ol0,2)*0.5/8 size(img_ol0,2)*0.5/8+20/res_ori(1)],...
            [size(img_ol0,1)*7.5/8 size(img_ol0,1)*7.5/8],'y',...
            'LineWidth', 2)

    title(sprintf('CDy1 area: %.2f%%', rate_area_CDy1*100))
    hold off
    
    mkdir([dir_plot, '/mipview/'])
    saveas(h_,[dir_plot, '/mipview/' fname_test,'.png'])

    rate_area_CDy1s = [rate_area_CDy1s; rate_area_CDy1];

end

%%
scores_contrast = scores_test(:,10)-mean(scores_test(:,1:9),2);
%%
x_ = min(rate_area_CDy1s):(max(rate_area_CDy1s)-min(rate_area_CDy1s))/255:max(rate_area_CDy1s)
P = polyfit(rate_area_CDy1s,scores_contrast,1);
h_ = figure(); h_.Color = [1 1 1];
plot(rate_area_CDy1s,scores_contrast, 'k.')
hold on
plot(x_, P(1)*x_ + P(2), 'r--'), axis square
xlabel('Area ratio of CDy1')
ylabel('Neural net output for Control')
saveas(h_,[dir_plot, '/mipview/curve.png'])
%%
h_ = figure(); h_.Color = [1 1 1];
idx_TP = intersect(find(lines_test == 2), find(preds_test ~= 9));
idx_FN = intersect(find(lines_test == 2), find(preds_test == 9));
idx_TN = intersect(find(lines_test == 1), find(preds_test == 9));
idx_FP = intersect(find(lines_test == 1), find(preds_test ~= 9));
hold on
bar(1, mean(rate_area_CDy1s(idx_TN)), 'FaceColor', [3/4 3/4 3/4])
errorbar(mean(rate_area_CDy1s(idx_TN)),std(rate_area_CDy1s(idx_TN)) ,'k')

bar(2, mean(rate_area_CDy1s(idx_FP)), 'FaceColor', [3/4 3/4 3/4])
errorbar(2,mean(rate_area_CDy1s(idx_FP)),std(rate_area_CDy1s(idx_FP)),'k')

bar(3, mean(rate_area_CDy1s(idx_FN)), 'FaceColor', [3/4 3/4 3/4])
errorbar(3,mean(rate_area_CDy1s(idx_FN)),std(rate_area_CDy1s(idx_FN)),'k')

bar(4, mean(rate_area_CDy1s(idx_TP)), 'FaceColor', [3/4 3/4 3/4])
errorbar(4,mean(rate_area_CDy1s(idx_TP)),std(rate_area_CDy1s(idx_TP)),'k')
ylim([0 inf])

ax = gca;
ax.XTick = [1 2 3 4];
ax.XTickLabel = {'HD09 -> Ctl', 'HD09 -> Non-ctl',...
    'PD20 -> Ctl', 'PD20 -> Non-ctl'};
ylabel('Area ratio of CDy1')
xlabel('Groups')

saveas(h_,[dir_plot, '/mipview/predcdy1.png'])


%%


dir_src = '/workspace01/gkim/stem_cell_jwshin/data/240716_TCF/00_train';

list_cls = dir('*_fl');
list_cls = list_cls([list_cls.isdir]);
for iter_cls = 1:length(list_cls)
    dir_cls = list_cls(iter_cls).name;
    cd(dir_src);cd(dir_cls);
    
    list_meas = dir('2*');
    list_meas = list_meas([list_meas.isdir]);
    
    for iter_meas = 1:length(list_meas)
        dir_meas = list_meas(iter_meas).name;
        cd(dir_src);cd(dir_cls);cd(dir_meas);
        
        list_TCF = dir('*.TCF');
        if isempty(list_TCF)
            continue
        end
        n_m = 1.337;
        hour = 0;
        fname_TCF = list_TCF(1).name;
%         fname_TCF = '240717.163612.HD09.032.Group1.A1.S032.TCF';
%         fname_TCF = '240716.232716.A20.020.Group1.A1.S020.TCF';
        info_temp = h5info(fname_TCF, '/Data/3D');
        ri = ReadLDMTCFHT(fname_TCF, hour);
        ri = single(ri*10000);
        ri = permute(ri, [2 1 3]);
        ri(ri == 0) = n_m*10000; 

        size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
        res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

        
        fl = ReadLDMTCFHT_FLCH0(fname_TCF, hour);
        fl = single(fl);
        fl = permute(fl, [2 1 3]);
        
        info_temp = h5info(fname_TCF, '/Data/3DFL');
        sizefl_ori = [info_temp.Attributes(10).Value info_temp.Attributes(11).Value info_temp.Attributes(12).Value];
        resfl_ori = [info_temp.Attributes(5).Value info_temp.Attributes(6).Value info_temp.Attributes(7).Value];




    end
end