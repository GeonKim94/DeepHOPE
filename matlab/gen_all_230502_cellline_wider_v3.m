% check size
% dataset division is 8:1:1
% for processing standard colony data (230323 + 230407)
% v2 difference: center-prioritizing crop to be curated (+0306 margin 15->20 fixed)
% v3 difference: doesn't resample the image for efficiency & integrity


clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')


h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

% dir_source = '/data02/gkim/stem_cell_jwshin/data/231228_TCF'; 
% dir_feat = '/data02/gkim/stem_cell_jwshin/data/231228_feat_wider_v3'; 
% dir_mip = '/data02/gkim/stem_cell_jwshin/data/231228_MIPH5_wider_v3'; 
% dir_sec1 = '/data02/gkim/stem_cell_jwshin/data/231228_SEC1H5_wider_v3'; 
% dir_img = '/data02/gkim/stem_cell_jwshin/data/231228_SEC1PNG_wider_v3';


% dir_source = '/data02/gkim/stem_cell_jwshin/data/231228_TCF'; 
% dir_feat = '/data02/gkim/stem_cell_jwshin/data/231228_feat_wider_v3'; 
% dir_mip = '/data02/gkim/stem_cell_jwshin/data/231228_MIPH5_wider_v3'; 
% dir_sec1 = '/data02/gkim/stem_cell_jwshin/data/231228_SEC1H5_wider_v3'; 
% dir_img = '/data02/gkim/stem_cell_jwshin/data/231228_SEC1PNG_wider_v3';


dir_source = '/data01/gkim/stem_cell_jwshin/data/231229_TCF'; 
dir_feat = '/data01/gkim/stem_cell_jwshin/data/231229_feat_wider_v3'; 
dir_mip = '/data01/gkim/stem_cell_jwshin/data/231229_MIPH5_wider_v3'; 
dir_sec1 = '/data01/gkim/stem_cell_jwshin/data/231229_SEC1H5_wider_v3'; 
dir_img = '/data01/gkim/stem_cell_jwshin/data/231229_SEC1PNG_wider_v3';


cd(dir_source)
dir_set = dir('0*');

size_pad = 20; 

hours = 0;%0:1:24;

sizes = [];

for hour = hours

for iter_set = length(dir_set):-1:1
    cd(dir_source)
    cd(dir_set(iter_set).name)
    dir_cls = dir('*Meso*48h*');
    
    if iter_set == 3
        stride = 0.5;%1;
    else
        stride = 0.5;
    end
    
    for iter_cls = 1:length(dir_cls)%[1 2 3 7 8 9 13 14 15 16 17 18]%1:length(dir_cls)%3:3:9%1:length(dir_cls)
        
        cd(dir_source)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_mat = dir('2*');
        for iter_stitch = 1:length(dir_mat)
            
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
            idx_dot = strfind(fname, '.');
            fname_save = fname(1:idx_dot(end)-1);
            %% comment this if you're fixing the data for the above condition
%             if exist([dir_img, '/', dir_set(iter_set).name, '/',...
%                 dir_cls(iter_cls).name, sprintf('_hr%02d', hour), '/', '00_colony_' fname_save, '.png'])
%                 'skipping... colony data already exist'
%                 continue
%             end
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

            mipz_stitch = max(tomo_stitch, [],3);
            mipx_stitch = squeeze(max(tomo_stitch, [],1));

            n_thres = 13420; %13370

            count_z = sum(single(mipx_stitch),1);
            z_glass = gradient(count_z);
            z_glass = find(z_glass == max(z_glass));

            z_sample = z_glass:1:z_glass+11;

            
            se = strel('disk', round(1/res_ori(1)));
            mask_stitch = mipz_stitch > n_thres;
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
            
            %% save mip img

            data = max((tomo_stitch),[],3);

            mkdir([dir_mip, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            
            cd([dir_mip, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);

            h5create([fname_save,'.h5'], '/ri', size(data));
            h5write([fname_save,'.h5'], '/ri', data);


            %% save sec1 img
            data = tomo_stitch(:,:,z_sample);

            mkdir([dir_sec1, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            cd([dir_sec1, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);

            h5create([fname_save,'.h5'], '/ri', size(data));
            h5write([fname_save,'.h5'], '/ri', data);

            set(0, 'CurrentFigure', h_), hold off

            subplot(1,2,1), imagesc(data(:,:,1), [13300 13800]), axis image, colormap gray
            subplot(1,2,2), imagesc(data(:,:,end), [13300 13800]), axis image, colormap gray

            mkdir([dir_img, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            cd([dir_img, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            saveas(h_, [fname_save, '.png'])

            %% feature extraction

            mipz_stitch = max(tomo_stitch,[],3);
            mipx_stitch = squeeze(max(single(tomo_stitch).*repmat(mask_stitch,[1 1 size(tomo_stitch,3)]),[],1))';
            mipy_stitch = squeeze(max(single(tomo_stitch).*repmat(mask_stitch,[1 1 size(tomo_stitch,3)]),[],2))';
            maskx_stitch = mipx_stitch > 13420;
            [maskx_stitch, num_obj] = bwlabeln(maskx_stitch);
            size_obj = [];
            for iter_obj = 1:num_obj
                size_obj(iter_obj) = sum(sum(sum(maskx_stitch == iter_obj)));
            end
            maskx_stitch = maskx_stitch == find(size_obj == max(size_obj));
    
            masky_stitch = mipy_stitch > 13420;
            masky_stitch = bwlabeln(masky_stitch);
            [masky_stitch, num_obj] = bwlabeln(masky_stitch);
            size_obj = [];
            for iter_obj = 1:num_obj
                size_obj(iter_obj) = sum(sum(sum(masky_stitch == iter_obj)));
            end
            masky_stitch = masky_stitch == find(size_obj == max(size_obj));


            mask_3D = (single(tomo_stitch).*repmat(mask_stitch,[1 1 size(tomo_stitch,3)])) > 13420;
            mask_3D = mask_3D.*repmat(permute(maskx_stitch', [3 1 2]),[size(tomo_stitch,1) 1 1]);
            mask_3D = mask_3D.*repmat(permute(masky_stitch', [1 3 2]),[1 size(tomo_stitch,2) 1]);

            mask_3D(:,:, 1:z_glass-5) = 0;
            mask_3D = logical(mask_3D);
            map_thick = sum(single(mask_3D),3);
            stats = regionprops(single(mask_stitch),'Centroid', 'Area');
            area_colony = stats.Area*res_ori(1)*res_ori(2);

            area_cyto = sum(sum(sum(mask_3D,3)>0))*res_ori(1)*res_ori(2);
            coor_crd = stats.Centroid;
            coor_crd = fliplr(coor_crd);      

            stats = regionprops(single(mask_stitch),sum(single(tomo_stitch).*single(mask_3D)/10000,3),'WeightedCentroid');
            coor_com = stats.WeightedCentroid;
            coor_com = fliplr(coor_com);     

            [XX, YY] = find(mask_stitch);
            XY = [XX YY];
            RR = sqrt((XX-coor_crd(1)).^2+(YY-coor_crd(2)).^2);

            RI_avg_colony = mean(single(tomo_stitch(mask_3D>0))/10000);
            RI_std_colony = std(single(tomo_stitch(mask_3D>0))/10000,0,1);
            volume_colony = sum(sum(sum(mask_3D)))*res_ori(1)*res_ori(2)*res_ori(3);
            thick_avg = volume_colony/area_colony;            


            se = strel('disk',50);
            mask_gap = (mipz_stitch<=13420).*imerode(mask_stitch,se);
            se = strel('disk',1);
            mask_gap = imerode(mask_gap,se);
            mask_gap = imdilate(mask_gap,se);
            area_gap = sum(sum(mask_gap))*res_ori(1)*res_ori(2);


            n_thres_lip1 = 13800;
            n_thres_lip2 = 14000;
            mask_lip = mipz_stitch > n_thres_lip1;
            mask_lip = mask_stitch.*mask_lip;
            se = strel('disk',10);
            mask_lip = mask_lip - imdilate(imdilate(imerode(mask_lip,se),se),se);
            mask_lip(mask_lip<0) = 0;

            mask_lip_3D = tomo_stitch>n_thres_lip1;
            mask_lip_3D = mask_lip_3D.*repmat(mask_lip,[1,1,size(tomo_stitch,3)]);
            mask_lip_3D = (mask_lip_3D + (tomo_stitch > n_thres_lip2))>0;

            mask_lip = sum(mask_lip_3D,3)>0;
            
            volume_lip = sum(sum(sum(mask_lip_3D)))*res_ori(1)*res_ori(2)*res_ori(3);
            area_lip = sum(sum(sum(mask_lip_3D,3)>0))*res_ori(1)*res_ori(2);
            RI_avg_lip = mean(single(tomo_stitch(mask_lip_3D>0))/10000);
            RI_std_lip = std(single(tomo_stitch(mask_lip_3D>0))/10000,0,1);

            volume_cyto = volume_colony-volume_lip;
            RI_avg_cyto = mean(single(tomo_stitch((mask_3D-mask_lip_3D)>0))/10000);
            RI_std_cyto = std(single(tomo_stitch((mask_3D-mask_lip_3D)>0))/10000,0,1);

            len_bound = 0;
            coor_bound = bwboundaries(mask_stitch);

            if length(coor_bound) > 1
                len_max = -inf;
                idx_max = 0;
                for iter_mask = 1:length(coor_bound)
                    len_current = length(coor_bound{iter_mask});
                    if len_current > len_max 
                        len_max = len_current;
                        idx_max = iter_mask;
                    end
                end
                coor_bound = coor_bound{idx_max};
            else
                coor_bound = coor_bound{1};
            end
            for iter_coor = 1:size(coor_bound,1)-1
                dr = coor_bound(iter_coor,:)-coor_bound(iter_coor+1,:);
                dr = sqrt(sum(abs(dr).^2));
                len_bound = len_bound+dr;
            end
            roundness = 4*pi*sum(sum(mask_stitch))/len_bound^2;
            stats = regionprops(mask_stitch,....
                'MinorAxisLength', 'MajorAxisLength', 'Solidity');
            eccentricity = sqrt(1-(stats.MinorAxisLength/stats.MajorAxisLength)^2);
            solidity = stats.Solidity;

            se = strel('disk',10);
            mask_boundin = mask_stitch-imerode(mask_stitch,se);
            mask_boundout = imdilate(mask_stitch,se)-mask_stitch;
            n_boundin = mean(single(mipz_stitch(mask_boundin>0))/10000);
            n_boundout = mean(single(mipz_stitch(mask_boundout>0))/10000);
            ncont_bound = n_boundin-n_boundout;
            
            thick_avg = sum(sum(map_thick.*mask_stitch))/sum(sum(mask_stitch)).*res_ori(3);
            thick_std = std(map_thick(mask_stitch>0),0,1)*res_ori(3);     
            
            center_disp = sum(((coor_crd-coor_com).*[res_ori(1), res_ori(2)]).^2);
    
            spread_thick = sqrt(sum(map_thick(mask_stitch).*(RR).^2)/sum(map_thick(mask_stitch)));
            
            skew_thick = [(sum(map_thick(mask_stitch).*(XX - coor_crd(1)).^3)/sum(map_thick(mask_stitch)))...
                (sum(map_thick(mask_stitch).*(YY - coor_crd(2)).^3)/sum(map_thick(mask_stitch)))]...
                /spread_thick^3;
    
            kurt_thick = (sum(map_thick(mask_stitch).*(RR ).^4)/sum(map_thick(mask_stitch)))...
                /spread_thick^4;


            map_dm = sum(single(tomo_stitch).*((mask_3D-mask_lip_3D)>0) ,3);
            spread_dm = sqrt(sum(map_dm(mask_stitch).*(RR).^2)/sum(map_dm(mask_stitch)));
            
            skew_dm = [(sum(map_dm(mask_stitch).*(XX - coor_crd(1)).^3)/sum(map_dm(mask_stitch)))...
                (sum(map_dm(mask_stitch).*(YY - coor_crd(2)).^3)/sum(map_dm(mask_stitch)))]...
                /spread_dm^3;
    
            kurt_dm = (sum(map_dm(mask_stitch).*(RR ).^4)/sum(map_dm(mask_stitch)))...
                /spread_dm^4;

            map_lip = sum(mask_lip_3D ,3);
            spread_lip = sqrt(sum(map_lip(mask_stitch).*(RR).^2)/sum(map_lip(mask_stitch)));
            
            skew_lip = [(sum(map_lip(mask_stitch).*(XX - coor_crd(1)).^3)/sum(map_lip(mask_stitch)))...
                (sum(map_lip(mask_stitch).*(YY - coor_crd(2)).^3)/sum(map_lip(mask_stitch)))]...
                /spread_lip^3;
    
            kurt_lip = (sum(map_lip(mask_stitch).*(RR ).^4)/sum(map_lip(mask_stitch)))...
                /spread_dm^4;


            mkdir([dir_feat, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);

            save([dir_feat, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name, '/',...
                fname_save '.mat'],...
                'volume_colony','area_colony','RI_avg_colony','RI_std_colony',...
                'volume_lip','area_lip','RI_avg_lip','RI_std_lip',...# volume_cyto = volume_colony-volume_lip
                'volume_cyto','area_cyto','RI_avg_cyto','RI_std_cyto',...
                'area_gap','len_bound','roundness','solidity','eccentricity',...
                'n_boundin','n_boundout','ncont_bound',...
                'thick_avg','thick_std', 'spread_thick', 'skew_thick', 'kurt_thick',...
                'spread_dm', 'skew_dm', 'kurt_dm',...
                'spread_lip', 'skew_lip', 'kurt_lip');

            sizes = [sizes; size(mask_stitch)];

         
        end
    end
    
end

end