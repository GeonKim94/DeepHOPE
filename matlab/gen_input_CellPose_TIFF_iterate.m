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
dir_save = '/data02/gkim/stem_cell_jwshin/data/_cellpose/input_tiff';
dir_data = '/data02/gkim/stem_cell_jwshin/data';
cd(dir_data)
list_exp = dir('*3D*');
for iter_exp = 1:length(list_exp)
    cd(dir_data)
    cd(list_exp(iter_exp).name)
    list_set = dir('*');
    list_set = list_set(3:end);
    list_set = list_set(find([list_set.isdir]));
    
    for iter_set = 1:length(list_set)
        cd(dir_data)
        cd(list_exp(iter_exp).name)
        cd(list_set(iter_set).name)
        list_cls = dir('*_*');

        for iter_cls = 1:length(list_cls)
            cd(dir_data)
            cd(list_exp(iter_exp).name)
            cd(list_set(iter_set).name)
            cd(list_cls(iter_cls).name)
            list_h5 = dir('*.h5');
            
            for iter_h5 = 1:length(list_h5)
                
                fname_3d = list_h5(iter_h5).name;

                mkdir([dir_save, '/', ...
                            list_exp(iter_exp).name, '/', ...
                            list_set(iter_set).name, '/', ...
                            list_cls(iter_cls).name, '/']);

                path_save = [dir_save, '/', ...
                            list_exp(iter_exp).name, '/', ...
                            list_set(iter_set).name, '/', ...
                            list_cls(iter_cls).name, '/', ...
                            strrep(fname_3d,'.h5','.tiff')];

                if exist(path_save)
                    continue
                end

                res_3d = [0.15 0.15 0.9]; 
                res_ori = [0.155432865023613 0.155432865023613 0.949573814868927];
                ri_m = 1.337;
%                 thres_ri = 1.3400;
%                 thres_ri_lip = 1.3675;
                
                ri_3d = h5read(fname_3d,'/ri');
                figure(2)
                imagesc(max(ri_3d,[],3), [1.337 1.38]), axis image, colormap gray
                
                ri_3d(ri_3d<ri_m) = ri_m;
                

             
                for iter_z = 1:size(ri_3d,3)
                    ri_slice = uint16(ri_3d(:,:,iter_z)*10000);
                    if iter_z == 1
                        imwrite(ri_slice,path_save)
                    else
                        imwrite(ri_slice,path_save,'WriteMode','append')
                    end
                end


            end

            
        end
        
    end

end
