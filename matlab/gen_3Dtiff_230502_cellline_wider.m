% check size
% dataset division is 8:1:1
% for processing standard colony data (230323 + 230407)

clear
clc
close all

addpath('/data01/gkim/Matlab_subcodes/gk')

dir_source = '/data02/gkim/stem_cell_jwshin/data/230502_TCF'; % 220822_TCF';%220715_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/_cellpose/input_tiff/230502_tiff_wider'; %'/data02/gkim/stem_cell_jwshin/data/230502_3DH5';
dir_img = '/data02/gkim/stem_cell_jwshin/data/_cellpose/MIPpreview/230502_tiff_wider';%%230221_PNG'; %220902_PNG';%220803_PNG';

cd(dir_source)
dir_set = dir('0*');

size_pad = 20;% (um) % was 5 before "wider" version 

hours = 0;%0:1:24;

h_ = figure(1);
h_.Position = [0 0 1800 900];
h_.Color = [1 1 1];
for hour = hours

for iter_set = length(dir_set):-1:1
    cd(dir_source)
    cd(dir_set(iter_set).name)
    dir_cls = dir('*_*');
    
    if iter_set == 3
        stride = 0.5;%1;
    else
        stride = 0.5;
    end
    
    for iter_cls = 1:length(dir_cls)%1:length(dir_cls)
        
        cd(dir_source)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_mat = dir('23*');
        for iter_stitch = length(dir_mat):-1:1
            
            iter_x = 0;
            iter_y = 0;

            cd(dir_source)
            cd(dir_set(iter_set).name)
            cd(dir_cls(iter_cls).name)
            cd(dir_mat(iter_stitch).name);

            dir_tcf = dir('*.TCF');

            if length(dir_tcf) == 0 
                continue
            end
            
            fname = dir_tcf(1).name;
            mkdir([dir_img, '/', ...
                        dir_set(iter_set).name, '/', ...
                        dir_cls(iter_cls).name, '/']);

            mkdir([dir_save, '/', ...
                        dir_set(iter_set).name, '/', ...
                        dir_cls(iter_cls).name, '/']);

            path_save = [dir_save, '/', ...
                        dir_set(iter_set).name, '/', ...
                        dir_cls(iter_cls).name, '/', ...
                        strrep(fname,'.TCF','.tiff')];

            if exist(path_save)
                continue
            end
            %% comment this if you're fixing the data for the above condition
%             if exist([dir_img, '/', dir_set(iter_set).name, '/',...
%                 dir_cls(iter_cls).name, sprintf('_hr%02d', hour), '/', '00_colony_' fname_save, '.png'])
%                 'skipping... colony data already exist'
%                 continue
%             end
            %%
            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);
            tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            tomo_stitch(tomo_stitch == 0) = n_m; 
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];
 
            size_new = [512 512 64];
            size_cen = [384 384 64]; % the s
            % ize that preprocess3d.py will result in

            res_new_ = [0.15 0.15 0.9]; %0.3];
            %size_crop = round(size_new.*res_new_./res_ori);
            mipx_stitch = squeeze(max(tomo_stitch, [],1));
            mipz_stitch = max(tomo_stitch, [],3);

            n_thres = 13420; %13370


%             mask_stitch = mipz_stitch >= n_thres;
%             se = strel('disk', round(2/res_ori(1)));
%             mask_stitch = imdilate(mask_stitch,se);
%             mask_stitch = imfill(mask_stitch,'holes');
%             mask_stitch = imerode(mask_stitch,se);
% 
%             mask_stitch = bwlabel(mask_stitch, 4);
%             label_colony = mask_stitch(round(size_ori(1)/2), round(size_ori(2)/2));
%             if label_colony == 0
%                 rprop = regionprops(mask_stitch);
%                 label_colony = find([rprop.Area] == max([rprop.Area]));
%             end
%             mask_stitch = (mask_stitch==label_colony);
            mip_stitch = max(tomo_stitch,[],3);
            mask_stitch = mip_stitch > 13420;
            [lbl,num] = bwlabeln(mask_stitch, 4);
            pixels_lbl = [];
            for idx_mask = 1:num
                pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
            end
            mask_stitch = (lbl==find(pixels_lbl == max(pixels_lbl)));
            mask_stitch = imfill(mask_stitch,'holes');
            
            x_crop_ = find(sum(mask_stitch,2)>0);
            x_crop_ = [max(min(x_crop_)-round(size_pad/res_ori(1)),1) min(max(x_crop_)+round(size_pad/res_ori(1)),size_ori(1))];
            x_crop_ = single(x_crop_);
            
            y_crop_ = find(sum(mask_stitch,1)>0);
            y_crop_ = [max(min(y_crop_)-round(size_pad/res_ori(1)),1) min(max(y_crop_)+round(size_pad/res_ori(1)),size_ori(2))];
            y_crop_ = single(y_crop_);

            tomo_stitch = tomo_stitch(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2),:);
            mask_stitch = mask_stitch(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2));

            data = single(tomo_stitch);%/10000;
            data = imresize3(data, round(size(data).*res_ori./res_new_), "cubic");
            mask_stitch = imresize(mask_stitch, round(size(mask_stitch).*res_ori(1:2)./res_new_(1:2)), "bicubic");

            thickness_min = 24;%(um)

            count_z = squeeze(squeeze(sum(sum(single(data>n_thres),1),2)));
            z_glass = gradient(count_z);
            z_glass = find(z_glass == max(z_glass));
           
            if z_glass-2 < 1
                data = cat(3, 13370*ones(size(data,1),size(data,2),2-z_glass+1),data );
                z_glass = 3;
            end

            if (size(data,3)-z_glass+2) >= round(thickness_min/res_new_(3))
                data = data(:,:, z_glass-2 : z_glass-2 + round(thickness_min/res_new_(3)));
            else
                data = cat(3, data, 13370*ones(size(data,1),size(data,2), round(thickness_min/res_new_(3)) - (size(data,3)-z_glass+2)));
                data = data(:,:, z_glass-2 : end);
            end

            for iter_z = 1:size(data,3)
                data_slice = uint16(data(:,:,iter_z));
                if iter_z == 1
                    imwrite(data_slice,path_save)
                else
                    imwrite(data_slice,path_save,'WriteMode','append')
                end
            end


            set(0, 'CurrentFigure', h_), hold off
            if size(data,1)<size(data,2)
                imagesc(squeeze(max(data,[],3)), [13370 13800]), axis image, colormap gray
            else
                imagesc(squeeze(max(data,[],3))', [13370 13800]), axis image, colormap gray
            end
            h_.Position = [0 0 1800 900];
            pause(0.2)
            saveas(h_, [dir_img, '/', ...
                        dir_set(iter_set).name, '/', ...
                        dir_cls(iter_cls).name, '/' strrep(fname,'.TCF','.png')]);


        end
    end
    
end

end