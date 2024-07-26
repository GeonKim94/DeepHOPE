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

% dir_source = '/data02/gkim/stem_cell_jwshin/data/231228_TCF'; 
% dir_feat = '/data02/gkim/stem_cell_jwshin/data/231228_feat_wider_v3'; 
% dir_mip = '/data02/gkim/stem_cell_jwshin/data/231228_MIPH5_wider_v3'; 
% dir_sec1 = '/data02/gkim/stem_cell_jwshin/data/231228_SEC1H5_wider_v3'; 
% dir_img = '/data02/gkim/stem_cell_jwshin/data/231228_SEC1PNG_wider_v3';

hour = 0;

dir_raw = '/data02/gkim/stem_cell_jwshin/data';
cd(dir_raw)
list_source = dir('2*_TCF'); 
list_source = list_source(~contains({list_source.name}, 'RI2FL'));
for iter_source = 1:length(list_source)
dir_source = [dir_raw, '/' list_source(iter_source).name];
dir_feat = [strrep(strrep(strrep(dir_source,'/data02','/workspace01'),'_TCF', '_mask_wider_v3'),'/data/','/data/23_mask_wider_v3_allh_onRA/')];

cd(dir_source)
dir_set = dir('0*');

size_pad = 20; 

sizes = [];

for iter_set = length(dir_set):-1:1
    cd(dir_source)
    cd(dir_set(iter_set).name)
    dir_cls = dir('*_*');
    
    if iter_set == 3
        stride = 0.5;%1;
    else
        stride = 0.5;
    end
    
    for iter_cls = 1:length(dir_cls)%[1 2 3 7 8 9 13 14 15 16 17 18]%1:length(dir_cls)%3:3:9%1:length(dir_cls)
        
        cd(dir_source)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        list_mat = dir('2*');
        list_mat=list_mat(find([list_mat.isdir]));
        for iter_stitch = 1:length(list_mat)
            
            iter_x = 0;
            iter_y = 0;

            cd(dir_source)
            cd(dir_set(iter_set).name)
            cd(dir_cls(iter_cls).name)
            cd(list_mat(iter_stitch).name);

            dir_tcf = dir('*.TCF');

            if length(dir_tcf) == 0 
                continue
            end
            
            
            fname = dir_tcf(1).name;
            idx_dot = strfind(fname, '.');
            fname_save = fname(1:idx_dot(end)-1);
            %% comment this if you're fixing the data for the above condition
            if ~exist([dir_feat, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name, '/',...
                fname_save '.mat'])
                'skipping... colony data already exist'
                continue
            end
            %%
            try 
            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);


            catch
                continue
            end

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
            
            %% mask_extraction
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
            mask_cyto = sum(mask_3D,3)>0;
            map_thick = sum(single(mask_3D),3);
            load([dir_feat, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name, '/',...
                fname_save '.mat'],...
                'mask_stitch','mask_cyto','mask_gap','mask_lip','mask_boundin','mask_boundout',...
                'x_crop_','y_crop_')
            save([dir_feat, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name, '/',...
                fname_save '.mat'],...
                'mask_stitch','mask_cyto','mask_gap','mask_lip','mask_boundin','mask_boundout',...
                'x_crop_','y_crop_','map_thick');

      
        end
    end
    
end

end