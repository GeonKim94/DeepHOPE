%% check size

clear
clc
close all

addpath('/data01/gkim/Matlab_subcodes/gk')

dir_source = '/data02/gkim/stem_cell_jwshin/data/220715_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220715_3D';
dir_img = '/data02/gkim/stem_cell_jwshin/data/220715_PNG';

cd(dir_source)
dir_set = dir('0*');

for iter_set = 1:length(dir_set)
    cd(dir_source)
        cd(dir_set(iter_set).name)
    dir_cls = dir('0*');
    
    if iter_set == 3
        stride = 1;
    else
        stride = 2;
    end
    
    for iter_cls = 1:length(dir_cls)
        
        cd(dir_source)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_mat = dir('2*');
        for iter_stitch = 1:length(dir_mat)
            

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
            	dir_cls(iter_cls).name]);
            
            mkdir([dir_img, '/',...
            	dir_set(iter_set).name, '/',...
            	dir_cls(iter_cls).name]);
            

            fname = dir_tcf(1).name;
            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);
            tomo_stitch = h5read(fname, '/Data/3D/000000');
            tomo_stitch(tomo_stitch == 0) = n_m; 
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];


            size_new = [384 384 32];
            res_new_ = [0.15 0.15 0.6];
            size_crop = round(size_new.*res_new_./res_ori);

            mipx_stitch = squeeze(max(tomo_stitch, [],1));
            mipz_stitch = max(tomo_stitch, [],3);

            n_thres = 13390;

            count_z = sum(single(mipx_stitch),1);
            z_glass = gradient(count_z);
            z_glass = find(z_glass == max(z_glass));
            z_crop_max = min(z_glass+size_crop(3) - 2, size_ori(3));
            z_crop = [z_crop_max-size_crop(3)+1 z_crop_max];

            mask_stitch = mipz_stitch >= n_thres;
            se = strel('disk', round(2/res_ori(1)));
            mask_stitch = imdilate(mask_stitch,se);
            mask_stitch = imfill(mask_stitch,'holes');
            mask_stitch = imerode(mask_stitch,se);

            [mask_stitch, areas] = bwlabel(mask_stitch, 4);
            label_colony = mask_stitch(round(size_ori(1)/2), round(size_ori(2)/2));
            mask_stitch = (mask_stitch==label_colony);

            figure(1)
            subplot(1,2,1), imagesc(mipz_stitch, [13370 13690]), axis image, colormap gray
            subplot(1,2,2), imagesc(mask_stitch, [0 1]), axis image, colormap gray

            x_crop_ = find(sum(mask_stitch,2)>0);
            x_crop_ = [min(x_crop_) max(x_crop_)];
            y_crop_ = find(sum(mask_stitch,1)>0);
            y_crop_ = [min(y_crop_) max(y_crop_)];
            
            num_crop = [ceil((x_crop_(2)-x_crop_(1)+1)/size_crop(1)),...
                ceil((y_crop_(2)-y_crop_(1)+1)/size_crop(2))];
            if num_crop(1)*size_crop(1) > size_ori(1)
                tomo_stitch = cenpad2d(tomo_stitch, num_crop(1)*size_crop(1),size(tomo_stitch,2), 13370);
                mask_stitch = cenpad2d(mask_stitch, num_crop(1)*size_crop(1),size(mask_stitch,2), 0);
            end
            if num_crop(2)*size_crop(2) > size_ori(2)
                tomo_stitch = cenpad2d(tomo_stitch,size(tomo_stitch,1), num_crop(2)*size_crop(2), 13370);
                mask_stitch = cenpad2d(mask_stitch,size(mask_stitch,1), num_crop(2)*size_crop(2), 0);
            end

            x_crop_ = find(sum(mask_stitch,2)>0);
            x_crop_ = [min(x_crop_) max(x_crop_)];
            y_crop_ = find(sum(mask_stitch,1)>0);
            y_crop_ = [min(y_crop_) max(y_crop_)];

            if mean(x_crop_) > floor(size(mask_stitch,1)/2+1)
                x_crop_max = min(size(tomo_stitch,1), floor(mean(x_crop_))+floor(num_crop(1)*size_crop(1)/2));
                x_crop = [x_crop_max - num_crop(1)*size_crop(1)+1 x_crop_max];
            else
                x_crop_min = max(1, floor(mean(x_crop_))-ceil(num_crop(1)*size_crop(1)/2));
                x_crop = [x_crop_min x_crop_min + num_crop(1)*size_crop(1)-1];
            end

            if mean(y_crop_) > floor(size(mask_stitch,2)/2+1)
                y_crop_max = min(size(tomo_stitch,2), floor(mean(y_crop_))+floor(num_crop(2)*size_crop(2)/2));
                y_crop = [y_crop_max - num_crop(2)*size_crop(2)+1 y_crop_max];
            else
                y_crop_min = max(1, floor(mean(y_crop_)) - ceil(num_crop(2)*size_crop(2)/2));
                y_crop = [y_crop_min y_crop_min + num_crop(2)*size_crop(2)-1];
            end


            idx_dot = strfind(fname, '.');
            fname_save = fname(idx_dot(end-2)+1:idx_dot(end)-1);
            h = figure(5);
            h.Position = [595 327 645 651];
            h.Color = [1 1 1];
            for iter_x = 1:stride*num_crop(1)-1
                for iter_y = 1:stride*num_crop(2)-1


                    score_mask = mask_stitch(round(x_crop(1)+(iter_x-1)*size_crop(1)/stride)...
                        :round(x_crop(1)+(iter_x-1)*size_crop(1)/stride)+size_crop(1)-1,...
                        round(y_crop(1)+(iter_y-1)*size_crop(2)/stride)...
                        :round(y_crop(1)+(iter_y-1)*size_crop(2)/stride)+size_crop(2)-1);
                    if sum(sum(score_mask)) < 1/3*size_crop(1)*size_crop(2)
                        continue
                    end

                    tomo = single(tomo_stitch(round(x_crop(1)+(iter_x-1)*size_crop(1)/stride)...
                        :round(x_crop(1)+(iter_x-1)*size_crop(1)/stride)+size_crop(1)-1,...
                        round(y_crop(1)+(iter_y-1)*size_crop(2)/stride)...
                        :round(y_crop(1)+(iter_y-1)*size_crop(2)/stride)+size_crop(2)-1,...
                        z_crop(1):z_crop(2)))/10000;
                    
                    cd([dir_save, '/',...
                        dir_set(iter_set).name, '/',...
                        dir_cls(iter_cls).name]);
                    
                    data = imresize3(tomo,size_new, 'linear');
                    res_new = (res_ori.*size_crop)./size_new;
                    resx = res_new(1);
                    resy = res_new(2);
                    resz = res_new(3);
                    
                    save([fname_save, sprintf('_%02d_%02d.mat',iter_x, iter_y)], 'data', 'resx', 'resy', 'resz','-v7.3');
                    
                    cd([dir_img, '/',...
                        dir_set(iter_set).name, '/',...
                        dir_cls(iter_cls).name]);
                  
                    
                    set(0, 'CurrentFigure', h)
                    subplot(2,2,1), imagesc(max(data,[],3), [1.337 1.36]),axis image, colormap gray
                    title('MIP - XY')
                    subplot(2,2,2), imagesc(squeeze(max(data,[],2)), [1.337 1.36]),axis image, colormap gray
                    title('MIP - YZ')
                    colorbar
                    subplot(2,2,3), imagesc(squeeze(max(data,[],1))', [1.337 1.36]),axis image, colormap gray
                    title('MIP - XZ')

                    
                    saveas(h, [fname_save, sprintf('_%02d_%02d.png',iter_x, iter_y)])
                    
                    
                    
                end
            end
        end
    end
    
end

%% compress

clear
clc
close all

addpath('/data01/gkim/Matlab_subcodes/gk')

dir_source = '/data02/gkim/stem_cell_jwshin/data/220715_3D';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220715_3D_comp';

cd(dir_source)
dir_set = dir('*');
dir_set = dir_set(3:end);

for iter_set = 1:length(dir_set)
    cd(dir_source)
    cd(dir_set(iter_set).name)
    dir_cls = dir('0*');

    for iter_cls = 1:length(dir_cls)
        
        cd(dir_source)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_mat = dir('*.mat');
        for iter_stitch = 1:length(dir_mat)
            

            cd(dir_source)
            cd(dir_set(iter_set).name)
            cd(dir_cls(iter_cls).name)
            load(dir_mat(iter_stitch).name);

         
                    
            mkdir(dir_save)
            cd(dir_save)
            mkdir(dir_set(iter_set).name)
            cd(dir_set(iter_set).name)
            mkdir(dir_cls(iter_cls).name)
            cd(dir_cls(iter_cls).name)
            save(dir_mat(iter_stitch).name, 'data', 'resx', 'resy', 'resz');




        end
    end
    
end
