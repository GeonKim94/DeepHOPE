% path_h5 = ['/workspace01/gkim/stem_cell_jwshin/data/23_SEC1H5_wider_v3_allh_onRA/test/00_RA_12h/',...
%     '230426.181016.GM25256_12hour_treat.054.Group1.A1.S054.h5'];
% 
% dir_raw = '/data02/gkim/stem_cell_jwshin/data';
% [~,fname_h5,~] = fileparts(path_h5);
% [path_raw, found] = search_recursive_v2(dir_raw,[fname_h5 '.TCF'], false);
% 
% ri = h5read(path_h5,'/ri');

clear
clc
close all

size_zoom = 384;
plot_range = [13300, 13800];

h_1 = figure(1);
h_1.Color = [1 1 1];
h_1.Position = [0 100 900 900];

h_2 = figure(2);
h_2.Color = [1 1 1];
h_2.Position = [0 100 900 900];

h_3 = figure(3);
h_3.Color = [1 1 1];
h_3.Position = [0 100 900 900];

dir_mask = '/workspace01/gkim/stem_cell_jwshin/data/23_mask_wider_v3_allh_onRA';
addpath(genpath('/data01/gkim/Matlab_subcodes/'))

paths_data = {};
paths_data{end+1} = ['/data02/gkim/stem_cell_jwshin/data/230502_TCF/00_train/GM25256_untreated/230426.154125.GM25256_untreated.047.Group1.A1.S047',...
    '/','230426.154125.GM25256_untreated.047.Group1.A1.S047.TCF'];
paths_data{end+1} = ['/data02/gkim/stem_cell_jwshin/data/230502_TCF/00_train/GM25256_12hour_treat/230426.181016.GM25256_12hour_treat.054.Group1.A1.S054',...
    '/','230426.181016.GM25256_12hour_treat.054.Group1.A1.S054.TCF'];

paths_data{end+1} = ['/data02/gkim/stem_cell_jwshin/data/230502_TCF/00_train/GM25256_24hour_treat/230426.185637.GM25256_24hour_treat.009.Group1.A1.S009',...
    '/', '230426.185637.GM25256_24hour_treat.009.Group1.A1.S009.TCF'];


dir_save = '/workspace01/gkim/stem_cell_jwshin/data_present/';
mkdir(dir_save)
for iter_data = 1:length(paths_data)
path_data = paths_data{iter_data};
[~,fname_data,~] = fileparts(path_data);
[path_mask, found_mask] = search_recursive_v2(dir_mask,fname_data, false);
load(path_mask);

ri = h5read(path_data,'/Data/3D/000000');
ri = ri(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2),:);

resx_ri = h5readatt(path_data, '/Data/3D', 'ResolutionX');
resy_ri = h5readatt(path_data, '/Data/3D', 'ResolutionY');
resz_ri = h5readatt(path_data, '/Data/3D', 'ResolutionZ');

mipz_stitch = max(ri, [],3);
mipx_stitch = squeeze(max(ri, [],1));

n_thres = 13420; %13370

count_z = sum(single(mipx_stitch),1);
z_glass = gradient(count_z);
z_glass = find(z_glass == max(z_glass));

z_sample = z_glass:1:z_glass+11;


props = regionprops(mask_stitch,'Centroid');
[XX,YY] = meshgrid(1:size(mask_stitch,1),1:size(mask_stitch,2));
XX = XX';
YY = YY';
RR = (XX-props.Centroid(1)).^2+(YY-props.Centroid(2)).^2;
RR(~mask_stitch) = 0;
[x0_fov,y0_fov] = find(RR == max(max(RR)));
x0_fov = x0_fov-floor(size_zoom/2);
y0_fov = y0_fov-floor(size_zoom/2);
x0_fov = min(max(0,x0_fov),size(mask_stitch,1)-size_zoom+1);
y0_fov = min(max(0,y0_fov),size(mask_stitch,2)-size_zoom+1);



set(0, 'CurrentFigure',h_1);
hold off
imagesc(max(ri,[],3),plot_range), axis image, colormap gray
hold on, plot([size(ri,2)*0.5/8 size(ri,2)*0.5/8+20/resx_ri],...
    [size(ri,1)*7.5/8 size(ri,1)*7.5/8],'y',...
    'LineWidth', 2)

saveas(h_1, [dir_save, '/' fname_data '_mip.fig'])

set(0, 'CurrentFigure',h_2);
subplot(2,2,1)
hold off
imagesc(max(cencrop2d(ri(:,:,z_sample),size_zoom,size_zoom),[],3),plot_range), axis image, colormap gray, axis off

hold on, plot([size_zoom*0.5/8 size_zoom*0.5/8+20/resx_ri],...
    [size_zoom*7.5/8 size_zoom*7.5/8],'y',...
    'LineWidth', 2)
for iter_z = 1:3
subplot(2,2,iter_z+1)
hold off
imagesc(cencrop2d(ri(:,:,z_sample(ceil(length(z_sample)*iter_z/4))),size_zoom,size_zoom),plot_range), axis image, colormap gray, axis off

hold on, plot([size_zoom*0.5/8 size_zoom*0.5/8+20/resx_ri],...
    [size_zoom*7.5/8 size_zoom*7.5/8],'y',...
    'LineWidth', 2)
end
saveas(h_2, [dir_save, '/' fname_data '_cropcenter.fig'])

set(0, 'CurrentFigure',h_3);
subplot(2,2,1)
hold off
imagesc(max(ri(x0_fov:x0_fov+size_zoom-1,y0_fov:y0_fov+size_zoom-1,z_sample),[],3),plot_range), axis image, colormap gray, axis off

hold on, plot([size_zoom*0.5/8 size_zoom*0.5/8+20/resx_ri],...
    [size_zoom*7.5/8 size_zoom*7.5/8],'y',...
    'LineWidth', 2)

for iter_z = 1:3
subplot(2,2,iter_z+1)
hold off
imagesc(ri(x0_fov:x0_fov+size_zoom-1,y0_fov:y0_fov+size_zoom-1,z_sample(ceil(length(z_sample)*iter_z/4))),plot_range), axis image, colormap gray, axis off

hold on, plot([size_zoom*0.5/8 size_zoom*0.5/8+20/resx_ri],...
    [size_zoom*7.5/8 size_zoom*7.5/8],'y',...
    'LineWidth', 2)
end
saveas(h_3, [dir_save, '/' fname_data '_cropedge.fig'])

end