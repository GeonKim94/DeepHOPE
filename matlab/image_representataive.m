
clc
clear
close all

addpath('/data02/gkim/RI2FL_Postech/src/matlab/subcodes/')

cmap_act = gray;
cmap_act(:,[1 3]) = 0;

cmap_lipid = gray;
cmap_lipid(:,3) = 0;

cmap_mem = gray;
cmap_mem(:,1) = 0;

cmap_mito = gray;
cmap_mito(:,[2,3]) = 0;

cmap_nuc = gray;
cmap_nuc(:,[1,2]) = 0;

cmap_oli = gray;
cmap_oli(:,2) = 0;

resolutionZ = 0.4;
%% LOAD DATA HERE

dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_MIPH5_wider_v2_allh_onRA/train/01_high';

cd(dir_data);
list_img = dir('*GM*.h5');
iter_img = 42;
fname_data = list_img(iter_img).name;
info = h5info(fname_data);
input = h5read(fname_data, '/ri');

input_ = (input-1.33)/(1.38-1.337);
input_(input_ > 1) = 1;
input_ = input_';

figure(11),imshow(input_) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)

imwrite(uint16(input_*(2^16-1)),'/data02/gkim/GMS061.tiff')


%%
dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_MIPH5_wider_v2_allh_onRA/train/01_high';

fname_data = '230425.163727.H9_untreated.060.Group1.A1.S060.h5';
info = h5info(fname_data);
input = h5read(fname_data, '/ri');

input_ = (input-1.33)/(1.38-1.33);
input_(input_ > 1) = 1;
input_ = input_;

figure(11),imshow(input_) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)

size_crop = 384;

startx_FOV1 = 650;
endx_FOV1 = startx_FOV1+size_crop-1;
starty_FOV1 = 400;
endy_FOV1 = starty_FOV1+size_crop-1;

input_FOV1 = input_(starty_FOV1:endy_FOV1,startx_FOV1:endx_FOV1);

figure(12),imshow(input_FOV1) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)

imwrite(uint16(input_FOV1*(2^16-1)),'/data02/gkim/H9_S060_FOV1.tiff')



startx_FOV2 = 1150;
endx_FOV2 = startx_FOV2+size_crop-1;
starty_FOV2 = 350;
endy_FOV2 = starty_FOV2+size_crop-1;

input_FOV2 = input_(starty_FOV2:endy_FOV2,startx_FOV2:endx_FOV2);

figure(13),imshow(input_FOV2) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)



imwrite(uint16(input_FOV1*(2^16-1)),'/data02/gkim/H9_S060_FOV1.tiff')
%%

dir_data = '/data02/gkim/stem_cell_jwshin/data/230502_TCF/00_train/H9_untreated';

fname_data = '230425.163727.H9_untreated.060.Group1.A1.S060.TCF';
cd(dir_data)
cd(erase(fname_data,'.TCF'))
info_temp = h5info(fname_data,'/Data/3D');
input = h5read(fname_data, '/Data/3D/000000');
res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];
input_crop = input(250:1710,300:2475,:);
h_f = figure(55);
set(h_f, 'Position', [0 200 1680 1080]);
set(h_f, 'Color', [1 1 1]);

v = VideoWriter('/data02/gkim/230425.163727.H9_untreated.060.Group1.A1.S060.avi');
v.FrameRate = 3;
open(v);
for iter_z = 26:size(input,3)-5
    imagesc(input_crop(:,:,iter_z),[13300 13800]),axis image, axis off,colormap gray
    pos = get(gca,'Position');
    pos(1) = 0.00;
    pos(2) = 0.00;
    pos(3) = 1.0;
    pos(4) = 1.0;
    title(sprintf('z = %.2f µm', res_ori(3)*(iter_z-26)));

    drawnow
    pause(0.1)
    pause(0.1)
    figure(55)
    frame = getframe(h_f);
    writeVideo(v,frame)
end
close(v);


h_f = figure(55);
set(h_f, 'Position', [0 0 1080 900]);
set(h_f, 'Color', [1 1 1]);

size_crop = 256;
startx_FOV1 = 426;
endx_FOV1 = startx_FOV1+size_crop-1;
starty_FOV1 = 276;
endy_FOV1 = starty_FOV1+size_crop-1;

v = VideoWriter('/data02/gkim/230425.163727.H9_untreated.060.Group1.A1.S060_FOV1.avi');
v.FrameRate = 3;
open(v);
for iter_z = 26:size(input,3)-5
    imagesc(input_crop(starty_FOV1:endy_FOV1,...
        startx_FOV1:endx_FOV1,...
        iter_z),[13300 13800]),axis image, axis off,colormap gray
    pos = get(gca,'Position');
    pos(1) = 0.00;
    pos(2) = 0.00;
    pos(3) = 1.0;
    pos(4) = 1.0;
    title(sprintf('z = %.2f µm', res_ori(3)*(iter_z-26)));

    drawnow
    pause(0.1)
    pause(0.1)
    figure(55)
    frame = getframe(h_f);
    writeVideo(v,frame)
end
close(v);




h_f = figure(55);
set(h_f, 'Position', [0 0 1080 900]);
set(h_f, 'Color', [1 1 1]);

size_crop = 256;
startx_FOV2 = 801;
endx_FOV2 = startx_FOV2+size_crop-1;
starty_FOV2 = 501;
endy_FOV2 = starty_FOV2+size_crop-1;

v = VideoWriter('/data02/gkim/230425.163727.H9_untreated.060.Group1.A1.S060_FOV2.avi');
v.FrameRate = 3;
open(v);
for iter_z = 26:size(input,3)-5
    imagesc(input_crop(starty_FOV2:endy_FOV2,...
        startx_FOV2:endx_FOV2,...
        iter_z),[13300 13800]),axis image, axis off,colormap gray
    pos = get(gca,'Position');
    pos(1) = 0.00;
    pos(2) = 0.00;
    pos(3) = 1.0;
    pos(4) = 1.0;
    title(sprintf('z = %.2f µm', res_ori(3)*(iter_z-26)));

    drawnow
    pause(0.1)
    pause(0.1)
    figure(55)
    frame = getframe(h_f);
    writeVideo(v,frame)
end
close(v);



%%

input_ = single(input_crop-13300)/(13800-13300);
input_(input_ > 1) = 1;
input_ = input_;


imwrite(uint16(max(input_,[],3)*(2^16-1)),'/data02/gkim/H9_S060_crop.tiff')

figure(11),imshow(uint16(max(input_,[],3)*(2^16-1))) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)

size_crop = 384;

imwrite(uint16(input_FOV1*(2^16-1)),'/data02/gkim/H9_S060_FOV1.tiff')
startx_FOV1 = 650;
endx_FOV1 = startx_FOV1+size_crop-1;
starty_FOV1 = 400;
endy_FOV1 = starty_FOV1+size_crop-1;

input_FOV1 = input_(starty_FOV1:endy_FOV1,startx_FOV1:endx_FOV1);

figure(12),imshow(input_FOV1) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)

imwrite(uint16(input_FOV1*(2^16-1)),'/data02/gkim/H9_S060_FOV1.tiff')



startx_FOV2 = 1150;
endx_FOV2 = startx_FOV2+size_crop-1;
starty_FOV2 = 350;
endy_FOV2 = starty_FOV2+size_crop-1;

input_FOV2 = input_(starty_FOV2:endy_FOV2,startx_FOV2:endx_FOV2);

figure(13),imshow(input_FOV2) %imagesc(input(:,:,z1), [0 0.8113]), axis image
set(gca, 'Colormap', gray)



imwrite(uint16(input_FOV1*(2^16-1)),'/data02/gkim/H9_S060_FOV1.tiff')