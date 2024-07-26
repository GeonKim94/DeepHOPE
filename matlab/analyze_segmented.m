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

%%
idxs_cell = unique(map_segmentation);
idxs_cell = idxs_cell(idxs_cell ~= 0);

%% colony level analysis
mipz_col = max(ri_3d, [],3);
mask_mipz = mipz_col > thres_ri;
[lbl,num] = bwlabeln(mask_mipz, 4);
pixels_lbl = [];
for idx_mask = 1:num
    pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
end
mask_mipz = (lbl==find(pixels_lbl == max(pixels_lbl)));
mask_mipz = imfill(mask_mipz,'holes');

se = strel('disk',round(20*res_ori(1)/res_3d(1)));
mask_interior = imerode(mask_mipz,se);
mask_exterior = imdilate(mask_mipz,se);

mipx_col = squeeze(max(single(ri_3d).*repmat(mask_mipz,[1 1 size(ri_3d,3)]),[],1))';
mipy_col = squeeze(max(single(ri_3d).*repmat(mask_mipz,[1 1 size(ri_3d,3)]),[],2))';

mask_mipx = mipx_col > thres_ri;
[mask_mipx, num_obj] = bwlabeln(mask_mipx);
size_obj = [];
for iter_obj = 1:num_obj
    size_obj(iter_obj) = sum(sum(sum(mask_mipx == iter_obj)));
end
mask_mipx = mask_mipx == find(size_obj == max(size_obj));

mask_mipy = mipy_col > thres_ri;
mask_mipy = bwlabeln(mask_mipy);
[mask_mipy, num_obj] = bwlabeln(mask_mipy);
size_obj = [];
for iter_obj = 1:num_obj
    size_obj(iter_obj) = sum(sum(sum(mask_mipy == iter_obj)));
end
mask_mipy = mask_mipy == find(size_obj == max(size_obj));

botx = [];
topx = [];
boty = [];
topy = [];
for iter_h = 1:size(mask_mipx,2)
    botx = [botx min(find(mask_mipx(:,iter_h)))];
    topx = [topx max(find(mask_mipx(:,iter_h)))];
end
for iter_h = 1:size(mask_mipy,2)
    boty = [boty min(find(mask_mipy(:,iter_h)))];
    topy = [topy max(find(mask_mipy(:,iter_h)))];
end

coor_bot = mode([botx boty]);

mask_3d = (single(ri_3d).*repmat(mask_mipz,[1 1 size(ri_3d,3)])) > thres_ri;
mask_3d = mask_3d.*repmat(permute(mask_mipx', [3 1 2]),[size(ri_3d,1) 1 1]);
mask_3d = mask_3d.*repmat(permute(mask_mipy', [1 3 2]),[1 size(ri_3d,2) 1]);
mask_3d(:,:, 1:coor_bot-1) = 0;

mask_lip = mipz_col > thres_ri_lip;
mask_lip = mask_exterior.*mask_lip;
se3 = strel('disk',4);
mask_lip = mask_lip - imdilate(imdilate(imerode(mask_lip,se3),se3),se3);
mask_lip(mask_lip<0) = 0;
[label_lip, num_lip] = bwlabel(mask_lip);



mask_3d_lip = mask_3d.*(ri_3d>thres_ri_lip);
%                     se4 = strel('sphere',4);
%                     mask_3D_lip = mask_3D_lip - imdilate(imdilate(imerode(mask_3D_lip,se4),se4),se4);
%                     mask_3D_lip(mask_3D_lip<0) = 0;
mask_3d_lip = mask_3d_lip.*repmat(mask_lip,[1,1,size(ri_3d,3)]);
[label_lip, num_lip] = bwlabeln(mask_3d_lip,6);

mask_3d_nolip = mask_3d-(mask_3d_lip);
mask_3d_nolip(mask_3d_nolip<0) = 0;

map_thick = sum(single(mask_3d),3);
props_col = regionprops(mask_mipz,'Centroid', 'Orientation', 'Circularity', 'Eccentricity', ...
    'MajorAxisLength', 'MinorAxisLength', 'Area');
coor_crd = props_col.Centroid;
coor_crd = fliplr(coor_crd);
%% cell level analysis

feats_cell = [];
for iter_cell = 1:length(idxs_cell)
    idx_cell = idxs_cell(iter_cell);
    map_cell = (map_segmentation == idx_cell);
    [lbl, num] = bwlabeln(map_cell);
    pixels_lbl = [];
    for idx_mask = 1:num
        pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
    end
    map_cell = (lbl==find(pixels_lbl == max(pixels_lbl)));
    
    props_cell = regionprops(map_cell, 'Centroid', 'Orientation', 'Circularity', 'Eccentricity', ...
    'MajorAxisLength', 'MinorAxisLength', 'Area');

    ri_cell = ri_3d.*mask_3d_nolip.*repmat(map_cell,[1 1 size(ri_3d,3)]);

    dist_cell = sqrt(sum(abs(fliplr(props_cell.Centroid)-coor_crd).^2));
    vol_cell = sum(sum(sum(mask_3d.*repmat(map_cell,[1 1 size(ri_3d,3)]))))*res_3d(1)*res_3d(2)*res_3d(3);
    dmdensity_cell = (mean(ri_cell(ri_cell>0)-1.337))/0.0018;
    drymass_cell = dmdensity_cell*vol_cell;

    vol_lip_cell = sum(sum(sum(mask_3d_lip.*repmat(map_cell,[1 1 size(ri_3d,3)]))))*res_3d(1)*res_3d(2)*res_3d(3);

    [label_lip_cell, num_lip_cell] = bwlabeln(mask_3d_lip.*repmat(map_cell,[1 1 size(ri_3d,3)]),6);

    area_cell = sum(sum(map_cell))*res_3d(1)*res_3d(2);
    thick_cell = vol_cell/area_cell;
    eccen_cell = props_cell.Eccentricity;
    circu_cell = props_cell.Circularity;
    orien_cell = props_cell.Orientation;
    axratio_cell =  props_cell.MinorAxisLength/props_cell.MajorAxisLength;
    lipratio_cell = vol_lip_cell/vol_cell;
    aspratio_cell = thick_cell/sqrt(area_cell);

    feats_cell = [feats_cell;
        [dist_cell, area_cell, thick_cell, vol_cell, vol_lip_cell, num_lip_cell, drymass_cell, dmdensity_cell, eccen_cell, circu_cell, axratio_cell, orien_cell, lipratio_cell, aspratio_cell]];

%     figure(1001), imagesc(max(ri_cell,[],3), [1.337 1.380]), axis image

end

min_dist = min(feats_cell(:,1)');
max_dist = max(feats_cell(:,1)');
%%

for iter_feat = 1:size(feats_cell,2)-1
    figure(iter_feat+100)
    plot(feats_cell(:,1),feats_cell(:,iter_feat+1), 'k.');
end

%%
perp = 10;
stan = false;
lr = 200;
feats_tsne = tsne(feats_cell(:,[2:8 10 13]), 'perplexity', perp, 'Standardize', stan, 'LearnRate', lr);



%% t-sne plot with distance color
hdl_fig = figure('Position', [0 0 600 600]);
set(gcf,'Color', [1 1 1])
set(gca, 'position', [0 0 1 1]);
set(gcf, 'paperpositionmode', 'auto');
hold on

cmap = parula;
for iter_data = 1:size(feats_tsne)

    idx_color = 1+round(255*(feats_cell(iter_data,1)-min_dist)/(max_dist-min_dist));
    color_plot = cmap(idx_color,:);

    sign_plot = '.';


    plot(feats_tsne(iter_data,1),feats_tsne(iter_data,2), 'Marker', sign_plot, 'Color', color_plot)
  
end
minx = min(feats_tsne(:,1)) - 0.1*(max(feats_tsne(:,1)) - min(feats_tsne(:,1))); 
maxx = max(feats_tsne(:,1)) + 0.1*(max(feats_tsne(:,1)) - min(feats_tsne(:,1))); 
miny = min(feats_tsne(:,2)) - 0.1*(max(feats_tsne(:,2)) - min(feats_tsne(:,2))); 
maxy = max(feats_tsne(:,2)) + 0.1*(max(feats_tsne(:,2)) - min(feats_tsne(:,2))); 
axis image off
xlim([minx maxx]);
ylim([miny maxy]);
hold off
%% t-sne plot with cluster color

idx_cluster = kmeans(feats_tsne,3);
hdl_fig = figure('Position', [0 0 600 600]);
set(gcf,'Color', [1 1 1])
set(gca, 'position', [0 0 1 1]);
set(gcf, 'paperpositionmode', 'auto');
hold on

cmap = [[1 0 0];[0 1 0];[0 0 1]];
for iter_data = 1:size(feats_tsne)

    color_plot = cmap(idx_cluster(iter_data),:);
    sign_plot = '.';


    plot(feats_tsne(iter_data,1),feats_tsne(iter_data,2), 'Marker', sign_plot, 'Color', color_plot)
  
end
minx = min(feats_tsne(:,1)) - 0.1*(max(feats_tsne(:,1)) - min(feats_tsne(:,1))); 
maxx = max(feats_tsne(:,1)) + 0.1*(max(feats_tsne(:,1)) - min(feats_tsne(:,1))); 
miny = min(feats_tsne(:,2)) - 0.1*(max(feats_tsne(:,2)) - min(feats_tsne(:,2))); 
maxy = max(feats_tsne(:,2)) + 0.1*(max(feats_tsne(:,2)) - min(feats_tsne(:,2))); 
axis image off
xlim([minx maxx]);
ylim([miny maxy]);
hold off
%% show 2d map according to cluster
map_feat = zeros(size(mipz_col,1),size(mipz_col,2),3);
for iter_cell = 1:length(idxs_cell)
    idx_cell = idxs_cell(iter_cell);
    map_cell = (map_segmentation == idx_cell);

    map_feat(:,:,idx_cluster(iter_cell)) = map_feat(:,:,idx_cluster(iter_cell))+map_cell;
end
figure(5), imshow(permute(map_feat,[2 1 3])), axis image

%% show 2d map according to features
%[dist_cell, area_cell, thick_cell, vol_cell, vol_lip_cell, num_lip_cell, drymass_cell, dmdensity_cell, eccen_cell, circu_cell, axratio_cell, orien_cell]];

{'dist_cell', 'area_cell', 'thick_cell', 'vol_cell', 'vol_lip_cell', ...
    'num_lip_cell', 'drymass_cell', 'dmdensity_cell', 'eccen_cell', 'circu_cell',...
    'axratio_cell', 'orien_cell', 'lipratio_cell', 'aspratio_cell'};
close all
for idx_feat = 8%1:size(feats_cell,2)

map_feat = zeros(size(mipz_col,1),size(mipz_col,2));
for iter_cell = 1:length(idxs_cell)
    idx_cell = idxs_cell(iter_cell);
    map_cell = (map_segmentation == idx_cell);

    map_feat = map_feat+map_cell.*feats_cell(iter_cell, idx_feat);
end


if idx_feat == 12
    figure(300+idx_feat), imagesc(map_feat', [-90 90]), axis image
    cmap = [flipud(hot); hot];
    colormap(cmap)
else
    figure(300+idx_feat), imagesc(map_feat', [0 inf]), axis image
    cmap = flipud(hot);
%     cmap(1,:) = [0 0 0];
    colormap(cmap)
end

end
