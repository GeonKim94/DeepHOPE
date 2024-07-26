close all
clear
clc
addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
dir_recon = '/data02/gkim/stem_cell_jwshin/results_CLAM/VAE3D_230811+230502_3DH5_wider_v3_allh_onRA_lf1024_b016_lr0.0001_weighted2/epoch[00174]_mse[0.2058]_pcc[0.2309]_kld[49.1558]/recon/train';
dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_3DH5_wider_v3_allh_onRA/train';
h_ = figure(1);

cap_min = 13300;
cap_max = 14000;
h_.Position = [0 0 1200 600];
h_.Color = [1 1 1];
cd(dir_data)
list_cls = dir('0*');
for iter_cls = 1:length(list_cls)
    cd(dir_data)
    cd(list_cls(iter_cls).name)
    list_h5 = dir('*.h5');
    for iter_h5 = 1:length(list_h5)
        cd(dir_data)
        cd(list_cls(iter_cls).name)
        data = h5read(list_h5(iter_h5).name,'/ri');
        data = single(data);
        data = (data - cap_min)/(cap_max - cap_min);
    
        data(data<0.0) = 0.0;
        data(data>1.0) = 1.0;
        cd(dir_recon)
        cd(list_cls(iter_cls).name)
        recon = h5read(list_h5(iter_h5).name,'/recon');
        data = cencrop3d(data, 384, 384, 32);
        recon = permute(recon, [3 2 1]);
        subplot(1,2,1)
        imagesc(data(:,:,16), [0 1]), axis image
        subplot(1,2,2)
        imagesc(recon(:,:,16), [0 1]), axis image
        corrcoef(single(data),recon)
        mean(abs(data-recon).^2, "all")
        pause()
    end
end