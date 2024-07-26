% check size
% dataset division is 8:1:1
% for processing standard colony data (230323 + 230407)
% v2 difference: center-prioritizing crop to be curated (+0306 margin 15->20 fixed)

clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')

dir_source = '/data02/gkim/stem_cell_jwshin/data/231117_TCF'; 
dir_save = '/data02/gkim/stem_cell_jwshin/data/231117_MIPH5_wider_v2'; 
dir_img = '/data02/gkim/stem_cell_jwshin/data/231117_MIPPNG_wider_v2';

cd(dir_source)
dir_set = dir('0*');
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

size_pad = 20; 

hours = 0;%0:1:24;

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
            
            mkdir([dir_save, '/',...
            	dir_set(iter_set).name, '/',...
            	dir_cls(iter_cls).name]);%, sprintf('_hr%02d', hour)]);
            
            mkdir([dir_img, '/',...
            	dir_set(iter_set).name, '/',...
            	dir_cls(iter_cls).name]);%, sprintf('_hr%02d', hour)]);
            
            fname = dir_tcf(1).name;
            idx_dot = strfind(fname, '.');
            fname_save = fname(1:idx_dot(end)-1);
            %% comment this if you're fixing the data for the above condition
            if exist([dir_img, '/', dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name, sprintf('_hr%02d', hour), '/', '00_colony_' fname_save, '.png'])
                'skipping... colony data already exist'
                continue
            end
            %%

            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);

            idx_slash = strfind(dir_source,'/');
            idx_slash = idx_slash(end);
            if str2num(dir_source(idx_slash+1:idx_slash+6)) < 230701
                tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 
            else
                tomo_stitch = ReadLDMTCFHT(fname, hour);
                tomo_stitch = single(tomo_stitch*10000);
                tomo_stitch = permute(tomo_stitch, [2 1 3]);
                tomo_stitch(tomo_stitch == 0) = n_m; 
            end
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];
 
            size_new = [512 512 64];
            size_cen = [384 384 64]; % the s
            % ize that preprocess3d.py will result in

            res_new_ = [0.15 0.15 0.3];
            size_crop = round(size_new.*res_new_./res_ori);
            mipx_stitch = squeeze(max(tomo_stitch, [],1));
            mipz_stitch = max(tomo_stitch, [],3);

            n_thres = 13420; %13370

            count_z = sum(single(mipx_stitch),1);
            z_glass = gradient(count_z);
            z_glass = find(z_glass == max(z_glass));
            z_crop_max = min(z_glass+size_crop(3) - 2, size_ori(3));
            z_crop = [z_crop_max-size_crop(3)+1 z_crop_max];

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
            se = strel('disk', round(1/res_ori(1)));
            mip_stitch = max(tomo_stitch,[],3);
            mask_stitch = mip_stitch > n_thres;
            mask_stitch = imdilate(mask_stitch,se);
            mask_stitch = imfill(mask_stitch,'holes');
            mask_stitch = imerode(mask_stitch,se);
            [lbl,num] = bwlabeln(mask_stitch, 4);
            if mask_stitch(round(end/2),round(end/2)) ~= 0 && sum(sum(lbl == mask_stitch(round(end/2),round(end/2))))*res_ori(1)^2>100*100
                lbl_colony = lbl(round(end/2),round(end/2));
            else
                pixels_lbl = [];
                for idx_mask = 1:num
                    pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
                end
                lbl_colony = find(pixels_lbl == max(pixels_lbl));
            end
            mask_stitch = (lbl==lbl_colony);
            mask_stitch = imfill(mask_stitch,'holes');
            
            x_crop_ = find(sum(mask_stitch,2)>0);
            x_crop_ = [max(min(x_crop_)-round(size_pad/res_ori(1)),1) min(max(x_crop_)+round(size_pad/res_ori(1)),size_ori(1))];
            x_crop_ = single(x_crop_);
            
            y_crop_ = find(sum(mask_stitch,1)>0);
            y_crop_ = [max(min(y_crop_)-round(size_pad/res_ori(1)),1) min(max(y_crop_)+round(size_pad/res_ori(1)),size_ori(2))];
            y_crop_ = single(y_crop_);

            tomo_stitch = tomo_stitch(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2),:);
            mask_stitch = mask_stitch(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2));
            
            data = max(single(tomo_stitch)/10000,[],3);
            data = imresize(data, round(size(data).*res_ori(1:2)./res_new_(1:2)), "bicubic");
%             mask_stitch = imresize(mask_stitch, round(size(mask_stitch).*res_ori(1:2)./res_new_(1:2)), "bicubic");
            set(0, 'CurrentFigure', h_), hold off
            subplot(1,2,1), imagesc(max(data,[],3), [1.3300 1.3800]), axis image, colormap gray
%             subplot(1,2,2), imagesc(mask_stitch, [0 1]), axis image, colormap gray
            subplot(1,2,2), imagesc(mipz_stitch, [13300 13800]), axis image, colormap gray
            drawnow
            pause(0.1)

            cd([dir_save, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);

%             save([fname_save, '.mat'],...%sprintf('_hr%02d.mat', hour)],...
%                 'data','-v7.3');

            h5create([fname_save,'.h5'], '/ri', size(data));
            h5write([fname_save,'.h5'], '/ri', data);

            cd([dir_img, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            saveas(h_, [fname_save, '.png'])
        end
    end
    
end

end

%% make test folder 