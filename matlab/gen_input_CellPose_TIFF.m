close all
clear
clc
addpath('/data02/gkim/RI2FL_Postech/src/matlab/subcodes')

fname_seg = '/data02/gkim/stem_cell_jwshin/outs/230518_segmentation/230426.182150.GM25256_12hour_treat.063.Group1.A1.h5';
info = h5info(fname_seg);

cmap = rand(256,3);
cmap(1,:) = [0 0 0];
map_segmentation = h5read(fname_seg,'/exported_data');
map_segmentation = squeeze(map_segmentation);
figure(1)
imagesc(map_segmentation), axis image, colormap(cmap)



% imwrite(uint16(map_segmentation),'/data02/gkim/stem_cell_jwshin/data/_cellpose/label/StemCell_MIP_mask_02.png')
%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3D_highmid_3lines/train/00_low/230426.182150.GM25256_12hour_treat.063.Group1.A1.S063.h5';
res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(2)
imagesc(max(ri_3d,[],3), [1.337 1.38]), axis image, colormap gray

ri_mip = max(ri_3d,[],3);
ri_mip(ri_mip<1.337) = 1.337;
ri_mip = uint16(ri_mip*10000);

for iter_z = 1:size(ri_3d,3)
    ri_slice = uint16(ri_3d(:,:,iter_z)*10000);
    if iter_z == 1
        imwrite(ri_slice,'/data02/gkim/stem_cell_jwshin/data/_cellpose/input_tiff/StemCell_3D_ri_02.TIFF')
    else
        imwrite(ri_slice,'/data02/gkim/stem_cell_jwshin/data/_cellpose/input_tiff/StemCell_3D_ri_02.TIFF','WriteMode','append')
    end
end


for iter_z = 1:size(ri_3d,3)
    if iter_z == 1
        imwrite(map_segmentation,'/data02/gkim/stem_cell_jwshin/data/_cellpose/label_tiff/StemCell_3D_mask_02.TIFF')
    else
        imwrite(map_segmentation,'/data02/gkim/stem_cell_jwshin/data/_cellpose/label_tiff/StemCell_3D_mask_02.TIFF','WriteMode','append')
    end
end
%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3D_highmid_3lines/train/01_high/230426.150725.GM25256_untreated.025.Group1.A1.S025.h5';
res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(2)
imagesc(max(ri_3d,[],3), [1.337 1.38]), axis image, colormap gray

ri_mip = max(ri_3d,[],3);
ri_mip(ri_mip<1.337) = 1.337;
ri_mip = uint16(ri_mip*10000);
imwrite(ri_mip,'/data02/gkim/stem_cell_jwshin/data/_cellpose/input/StemCell_MIP_ri_01.png')
%%
fname_3d = '/data02/gkim/stem_cell_jwshin/data/230502_3DH5/00_train/GM25256_24hour_treat/230426.203900.GM25256_24hour_treat.058.Group1.A1.S058.h5';
res_3d = [0.15 0.15 0.9]; 
res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
thres_ri = 1.3400;
thres_ri_lip = 1.3675;

ri_3d = h5read(fname_3d,'/ri');
figure(2)
imagesc(max(ri_3d,[],3), [1.337 1.38]), axis image, colormap gray

ri_mip = max(ri_3d,[],3);
ri_mip(ri_mip<1.337) = 1.337;
ri_mip = uint16(ri_mip*10000);
imwrite(ri_mip,'/data02/gkim/stem_cell_jwshin/data/_cellpose/input/StemCell_MIP_ri_03.png')
