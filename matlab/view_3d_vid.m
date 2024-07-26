
%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3D_highmid_3lines/train/01_high/230426.152759.GM25256_untreated.041.Group1.A1.S041.h5';
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3D_highmid_3lines/train/01_high/230426.150725.GM25256_untreated.025.Group1.A1.S025.h5';
 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(101)
imagesc(max(ri_3d,[],3), [1.337 1.38]), axis image, colormap gray

%%


resx_rs = 0.15;%0.15;
resy_rs = 0.15;%0.15;
resz_rs = 0.9;%0.6;

h_ = figure(1);
h_.Position = [1920 0 1920 1080];
h_.Color = [1 1 1];
dir_save = '/data02/gkim/stem_cell_jwshin/';
mkdir(dir_save)
figure(1),
mkdir(dir_save)
%if exist([dir_save strrep(fname_result, '.mat', '_FOV1.avi')])
if exist([dir_save strrep(fname_3d(max(strfind(fname_3d,'/'))+1:end), '.h5', '.avi')])
    error
end
%v = VideoWriter([dir_save erase(fname_result, '.mat') '_FOV1']);
v = VideoWriter([dir_save erase(fname_3d(max(strfind(fname_3d,'/'))+1:end), '.h5')]);
v.FrameRate = 3;
open(v)
for iter_z = 1:size(ri_3d,3)
    set(0, 'CurrentFigure', h_)
    imagesc(ri_3d(:,:,iter_z), [1.337 1.38]), axis image, colormap gray
    hold on, plot([size(ri_3d,2)*0.5/8 size(ri_3d,2)*0.5/8+10/resx_rs],...
        [size(ri_3d,1)*7.5/8,size(ri_3d,1)*7.5/8],'y',...
        'LineWidth', 2)
    hold off
    axis off
    title(sprintf('RI, z = %0.2f um',resz_rs*(iter_z-1)))

    colorbar
    
    pause(0.1)
    %h_.Position = [1920 0 1920 1080];
    frame = getframe(h_);
    writeVideo(v,frame)
end
    close(v);

%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3D_highmid_3lines/train/00_low/230426.182150.GM25256_12hour_treat.063.Group1.A1.S063.h5';
res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(102)
imagesc(max(ri_3d,[],3)', [1.337 1.38]), axis image, colormap gray
%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_24hour_treat/230426.204618.GM25256_24hour_treat.063.Group1.A1.S063.h5';
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_24hour_treat/230426.203900.GM25256_24hour_treat.058.Group1.A1.S058.h5';
res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(103)
imagesc(max(ri_3d,[],3)', [1.337 1.38]), axis image, colormap gray
%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_24hour_treat/230426.204618.GM25256_24hour_treat.063.Group1.A1.S063.h5';
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_24hour_treat/230426.203900.GM25256_24hour_treat.058.Group1.A1.S058.h5';
res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(103)
imagesc(max(ri_3d,[],3)', [1.337 1.38]), axis image, colormap gray
