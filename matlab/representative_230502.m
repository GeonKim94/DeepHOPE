addpath('/data02/gkim/RI2FL_Postech/src/matlab/subcodes')

%% low
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_24hour_treat/230426.200342.GM25256_24hour_treat.042.Group1.A1.S042.h5';

%% middle

fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_12hour_treat/230426.172309.GM25256_12hour_treat.023.Group1.A1.S023.h5';
%% high

fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_untreated/230426.152945.GM25256_untreated.042.Group1.A1.S042.h5';


%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_12hour_treat/230426.182150.GM25256_12hour_treat.063.Group1.A1.S063.h5';

res_3d = [0.15 0.15 0.9]; 

%%
idxs_slash = strfind(fname_3d,'/');
idxs_dot = strfind(fname_3d,'.');
fname_3d_ = fname_3d(idxs_slash(end)+1:idxs_dot(end)-1);
ri_3d = h5read(fname_3d,'/ri');

%%
cd('/data02/gkim/stem_cell_jwshin/data_present')

figure(1),imagesc(ri_3d(:,:,12)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_upper.fig'])
figure(2),imagesc(ri_3d(:,:,8)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_middle.fig'])
figure(3),imagesc(ri_3d(:,:,4)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_lower.fig'])

figure(4),imagesc(squeeze(ri_3d(round(end/2),:,:))', [1.337, 1.38]), axis image, colormap gray
daspect([3 1 1])
saveas(gcf,[fname_3d_, '_zmiddle.fig'])




figure(11),imagesc(ri_3d(1201:1584,401:784,12)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_ROI1_upper.fig'])
figure(12),imagesc(ri_3d(1201:1584,401:784,8)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_ROI1_middle.fig'])
figure(13),imagesc(ri_3d(1201:1584,401:784,4)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_ROI1_lower.fig'])



figure(11),imagesc(ri_3d(201:584,101:484,12)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_ROI2_upper.fig'])
figure(12),imagesc(ri_3d(201:584,101:484,8)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_ROI2_middle.fig'])
figure(13),imagesc(ri_3d(201:584,101:484,4)', [1.337, 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_ROI2_lower.fig'])

%%
fname_seg = '/data02/gkim/stem_cell_jwshin/outs/230518_segmentation/230426.182150.GM25256_12hour_treat.063.Group1.A1.h5';
info = h5info(fname_seg);

map_segmentation = h5read(fname_seg,'/exported_data');
map_segmentation = squeeze(map_segmentation);

%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_12hour_treat/230426.182150.GM25256_12hour_treat.063.Group1.A1.S063.h5';

idxs_slash = strfind(fname_3d,'/');
idxs_dot = strfind(fname_3d,'.');
fname_3d_ = fname_3d(idxs_slash(end)+1:idxs_dot(end)-1);

res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.340;

ri_3d = h5read(fname_3d,'/ri');

cmap = rand(256,3);
cmap(1,:) = [0 0 0];
figure(71)
imagesc(map_segmentation'), axis image, colormap(cmap)

cd('/data02/gkim/stem_cell_jwshin/data_present')
saveas(gcf,[fname_3d_, '_seg.fig'])

figure(72)
imagesc(max(ri_3d,[],3)', [1.337 1.38]), axis image, colormap gray
saveas(gcf,[fname_3d_, '_MIP.fig'])
