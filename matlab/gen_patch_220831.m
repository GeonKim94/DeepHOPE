%% check size
% dataset division is 8:1:1

clear
clc
close all

addpath('/data01/gkim/Matlab_subcodes/gk')

dir_source = '/data02/gkim/stem_cell_jwshin/data/220822_TCF';%220715_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220831_3D';%220803_3D';
dir_img = '/data02/gkim/stem_cell_jwshin/data/220831_PNG';%220803_PNG';

cd(dir_source)
dir_set = dir('0*');
h_ = figure(1);
h_.Position = [0 0 1800 900];
h_.Color = [1 1 1];
h = figure(5);
h.Position = [595 327 950 1000];
h.Color = [1 1 1];


size_pad = 0; %(um)

for iter_set = 1:length(dir_set)
    cd(dir_source)
        cd(dir_set(iter_set).name)
    dir_cls = dir('0*');
    
    if iter_set == 3
        stride = 0.5;%1;
    else
        stride = 0.5;
    end
    
    for iter_cls = 1:length(dir_cls)%1:length(dir_cls)
        
        cd(dir_source)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_mat = dir('2*');
        for iter_stitch = length(dir_mat):-1:1
            

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

            idx_dot = strfind(fname, '.');
            fname_save = fname(idx_dot(end-5)+1:idx_dot(end)-1);

            size_new = [320 320 64];
            res_new_ = [0.15 0.15 0.3];
            size_crop = round(size_new.*res_new_./res_ori);

            mipx_stitch = squeeze(max(tomo_stitch, [],1));
            mipz_stitch = max(tomo_stitch, [],3);

            n_thres = 13410; %13370

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

            mask_stitch = bwlabel(mask_stitch, 4);
            label_colony = mask_stitch(round(size_ori(1)/2), round(size_ori(2)/2));
            if label_colony == 0
                rprop = regionprops(mask_stitch);
                label_colony = find([rprop.Area] == max([rprop.Area]));
            end
            mask_stitch = (mask_stitch==label_colony);

            cropran_x = find(sum(mask_stitch,2)>0);
            cropran_x = [min(cropran_x) max(cropran_x)];
            cropcen_x = ceil(0.5*max(cropran_x)+0.5*min(cropran_x)+0.5);
            croplen_x = max(cropran_x)-min(cropran_x)+1;
            leftover_x = croplen_x-floor(croplen_x/size_crop(1)/stride)*size_crop(1)*stride;
            if min(cropran_x) < 1
                if max(cropran_x) == size_ori(1)
                    croplen_x = round(floor(croplen_x/size_crop(1)/stride)...
                        *size_crop(1)*stride);
                    cropran_x(1) = cropcen_x - floor(croplen_x/2+0.5);
                    cropran_x(2) = cropran_x(1) + croplen_x - 1;
                    
                    xs_start_ = round(cropran_x(1):(size_crop(1)*stride):(cropran_x(2)-size_crop(1)+5));
                    cropran_x(2) = max(xs_start_)+size_crop(1);
                    croplen_x = cropran_x(2)-croplen_x+1;
                    
                else
                    croplen_x = (ceil(croplen_x/size_crop(1)/stride) + (leftover_x>(0.5*(stride))))...
                        *size_crop(1)*stride;
                    cropran_x(1) = 1;
                    cropran_x(2) = cropran_x(1) + croplen_x - 1;
                    
                    xs_start_ = round(cropran_x(1):(size_crop(1)*stride):(cropran_x(2)-size_crop(1)+5));
                    cropran_x(2) = max(xs_start_)+size_crop(1);
                    croplen_x = cropran_x(2)-croplen_x+1;
                    
                    if cropran_x(2) > size_ori(1)
                        tomo_stitch = padarray(tomo_stitch, [(cropran_x(2) - size_ori(1)), 0],13370,'post');
                        mask_stitch = padarray(mask_stitch, [(cropran_x(2) - size_ori(1)), 0],0,'post');
                    end
                    
                end
            else    
                if max(cropran_x) > size_ori(1)
                    croplen_x = (ceil(croplen_x/size_crop(1)/stride) + (leftover_x>(0.5*(stride))))...
                        *size_crop(1)*stride;
                    cropran_x(2) = size_ori(1);
                    cropran_x(1) = cropran_x(1) - croplen_x + 1;
                    
                    xs_start_ = round(cropran_x(1):(size_crop(1)*stride):(cropran_x(2)-size_crop(1)+5));
                    cropran_x(2) = max(xs_start_)+size_crop(1);
                    croplen_x = cropran_x(2)-croplen_x+1;
                    
                    if cropran_x(1) <1
                        tomo_stitch = padarray(tomo_stitch, [(cropran_x(2) - size_ori(1)), 0],13370,'pre');
                        mask_stitch = padarray(mask_stitch, [(cropran_x(2) - size_ori(1)), 0],0,'pre');
                    end
                    
                    cropran_x(2) = size(tomo_stitch,1);
                    cropran_x(1) = cropran_x(2)-croplen_x+1;
                else
                    croplen_x = (ceil(croplen_x/size_crop(1)/stride) + (leftover_x>(0.5*(stride))))...
                        *size_crop(1)*stride;
                    cropran_x(1) = cropcen_x - floor(croplen_x/2+0.5);
                    cropran_x(2) = cropran_x(1) + croplen_x - 1;
                    
                    xs_start_ = round(cropran_x(1):(size_crop(1)*stride):(cropran_x(2)-size_crop(1)+5));
                    cropran_x(2) = max(xs_start_)+size_crop(1);
                    croplen_x = cropran_x(2)-cropran_x(1)+1;
                    
                    if (cropran_x(2) > size_ori(1)) || (cropran_x(1) < 1)
                        tomo_stitch = cenpad2d(tomo_stitch, croplen_x, size(tomo_stitch,2),13370);
                        mask_stitch = cenpad2d(mask_stitch, croplen_x, size(mask_stitch,2),0);
                    end
                    
                    cropran_x(2) = cropran_x(2) + round((size(tomo_stitch,1)-size_ori(1))/2);
                    cropran_x(1) = cropran_x(2)-croplen_x+1;
                    
                end
            end
            
            
            cropran_y = find(sum(mask_stitch,1)>0);
            cropran_y = [min(cropran_y) max(cropran_y)];
            cropcen_y = ceil(0.5*max(cropran_y)+0.5*min(cropran_y)+0.5);
            croplen_y = max(cropran_y)-min(cropran_y)+1;
            leftover_y = croplen_y-floor(croplen_y/size_crop(2)/stride)*size_crop(2)*stride;
            if min(cropran_y) < 1
                if max(cropran_y) == size_ori(2)
                    croplen_y = (floor(croplen_y/size_crop(2)/stride))...
                        *size_crop(2)*stride;
                    cropran_y(1) = cropcen_y - floor(croplen_y/2+0.5);
                    cropran_y(2) = cropran_y(1) + croplen_y - 1;
                else
                    croplen_y = (ceil(croplen_y/size_crop(2)/stride) + (leftover_y>(0.5*(stride))))...
                        *size_crop(2)*stride;
                    cropran_y(1) = 1;
                    cropran_y(2) = cropran_y(1) + croplen_y - 1;
                    
                    ys_start_ = round(cropran_y(1):(size_crop(2)*stride):(cropran_y(2)-size_crop(2)+5));
                    cropran_y(2) = max(ys_start_)+size_crop(2);
                    croplen_y = cropran_y(2)-croplen_y+1;
                    
                    if cropran_y(2) > size_ori(2)
                        tomo_stitch = padarray(tomo_stitch, [0, (cropran_y(2) - size_ori(2))],13370,'post');
                        mask_stitch = padarray(mask_stitch, [0, (cropran_y(2) - size_ori(2))],0,'post');
                    end
                    
                end
            else    
                if max(cropran_y) > size_ori(2)
                    croplen_y = (ceil(croplen_y/size_crop(2)/stride) + (leftover_y>(0.5*(stride))))...
                        *size_crop(2)*stride;
                    cropran_y(2) = size_ori(2);
                    cropran_y(1) = cropran_y(2) - croplen_y + 1;
                    
                    ys_start_ = round(cropran_y(1):(size_crop(2)*stride):(cropran_y(2)-size_crop(2)+5));
                    cropran_y(2) = max(ys_start_)+size_crop(2);
                    croplen_y = cropran_y(2)-croplen_y+1;
                    
                    if cropran_y(1) <1
                        tomo_stitch = padarray(tomo_stitch, [0, (cropran_y(2) - size_ori(2))],13370,'pre');
                        mask_stitch = padarray(mask_stitch, [0, (cropran_y(2) - size_ori(2))],0,'pre');
                    end
                    
                    cropran_y(2) = size(tomo_stitch,2);
                    cropran_y(1) = cropran_y(2)-croplen_x+1;
                else
                    croplen_y = (ceil(croplen_y/size_crop(2)/stride) + (leftover_y>(0.5*(stride))))...
                        *size_crop(2)*stride;
                    cropran_y(1) = cropcen_y - floor(croplen_y/2+0.5);
                    cropran_y(2) = cropran_y(1) + croplen_y - 1;
                    
                    ys_start_ = round(cropran_y(1):(size_crop(2)*stride):(cropran_y(2)-size_crop(2)+5));
                    cropran_y(2) = max(ys_start_)+size_crop(2);
                    croplen_y = cropran_y(2)-croplen_y+1;
                    
                    if (cropran_y(2) > size_ori(2)) || (cropran_y(1) < 1)
                        tomo_stitch = cenpad2d(tomo_stitch, size(tomo_stitch,1), croplen_y,13370);
                        mask_stitch = cenpad2d(mask_stitch, size(mask_stitch,1), croplen_y,0);
                    end
                    
                    cropran_y(2) = cropran_y(2) + round((size(tomo_stitch,1)-size_ori(1))/2);
                    cropran_y(1) = cropran_y(2)-croplen_x+1;
                end
            end
            
            
%             set(0, 'CurrentFigure', h_), hold off
%             subplot(1,2,1), imagesc(mipz_stitch, [13370 13690]), axis image, colormap gray
%             subplot(1,2,2), imagesc(mask_stitch, [0 1]), axis image, colormap gray
%             cd([dir_img, '/',...
%                         dir_set(iter_set).name, '/',...
%                         dir_cls(iter_cls).name]);
%             saveas(h_, ['00_colony_' fname_save, '.png'])
            
      
            set(0, 'CurrentFigure', h_)
            subplot(1,2,1), hold off
            imagesc(max(tomo_stitch,[],3), [13370 13800]),axis image, colormap gray
            subplot(1,2,2), hold off
            imagesc(mask_stitch, [0 1]),axis image, colormap gray
            
            for iter_x = 1:num_crop_x/stride-(1/stride-1)
                x_start = round(x_crop(1)+(iter_x-1)*size_crop(1)*stride);
                x_end = round(x_crop(1)+(iter_x-1)*size_crop(1)*stride)+size_crop(1)-1;


                mask_stitch_x = mask_stitch;
                mask_stitch_x(1:x_start-1,:) = 0;
                mask_stitch_x(x_end+1:end,:) = 0;
                y_crop_ = find(sum(mask_stitch_x,1)>0);
                y_crop_ = [max(min(y_crop_)-round(size_pad/res_ori(1)),0) min(max(y_crop_)+round(size_pad/res_ori(1)),size_ori(2))];
                
                
                if y_crop_(1) == 1
                    if y_crop_(2) == size_ori(2)
                        num_crop_y = floor((y_crop_(2)-y_crop_(1)+1)/size_crop(2)/stride)*stride;
                    else
                        num_crop_y = ceil((y_crop_(2)-y_crop_(1)+1)/size_crop(2)/stride)*stride;
                    end
                else
                    if y_crop_(2) == size_ori(2)
                        num_crop_y = ceil((y_crop_(2)-y_crop_(1)+1)/size_crop(2)/stride)*stride;
                    else
                        num_crop_y = ceil((y_crop_(2)-y_crop_(1)+1)/size_crop(2)/stride)*stride;%+1;
                    end
                end
                
                num_crop_y = ceil((y_crop_(2)-y_crop_(1)+1)/size_crop(2));

                if mean(y_crop_) > floor(size(mask_stitch,2)/2+1)
                   y_crop_max = min(size(tomo_stitch,2), floor(mean(y_crop_))+floor(num_crop_y*size_crop(2)/2));
                   y_crop = [y_crop_max - num_crop_y*size_crop(2)+1 y_crop_max];
                else
                   y_crop_min = max(1, floor(mean(y_crop_)) - ceil(num_crop_y*size_crop(2)/2));
                   y_crop = [y_crop_min y_crop_min + num_crop_y*size_crop(2)-1];
                end
                    
                for iter_y = 1:num_crop_y/stride-(1/stride-1)

                    
                    y_start = round(y_crop(1)+(iter_y-1)*size_crop(2)*stride);
                    y_end = round(y_crop(1)+(iter_y-1)*size_crop(2)*stride)+size_crop(2)-1;
%                     z_start = z_crop(1);
%                     z_end = z_crop(2);
%                     
%                     score_mask = mask_stitch(x_start:x_end,y_start:y_end);
%                     if sum(sum(score_mask)) < 1/10*size_crop(1)*size_crop(2)
%                         continue
%                     end
% 
%                     tomo = single(tomo_stitch(x_start:x_end,y_start:y_end,z_start:z_end))/10000;
%                     
%                     cd([dir_save, '/',...
%                         dir_set(iter_set).name, '/',...
%                         dir_cls(iter_cls).name]);
%                     
%                     data = imresize3(tomo,size_new, 'linear');
%                     res_new = (res_ori.*size_crop)./size_new;
%                     resx = res_new(1);
%                     resy = res_new(2);
%                     resz = res_new(3);
%                     
%                     save([fname_save, sprintf('_%02d_%02d.mat',iter_x, iter_y)],...
%                         'data', 'resx', 'resy', 'resz','fname','x_start','x_end',...
%                         'y_start','y_end','z_start','z_end','-v7.3');
%                     
%                     cd([dir_img, '/',...
%                         dir_set(iter_set).name, '/',...
%                         dir_cls(iter_cls).name]);
%                     set(0, 'CurrentFigure', h)
%                     subplot(2,2,1), imagesc(max(data,[],3), [1.337 1.38]),axis image, colormap gray
%                     title('MIP - XY')
%                     subplot(2,2,2), imagesc(squeeze(max(data,[],2)), [1.337 1.38]),axis image, colormap gray
%                     title('MIP - YZ')
%                     colorbar
%                     subplot(2,2,3), imagesc(squeeze(max(data,[],1))', [1.337 1.38]),axis image, colormap gray
%                     title('MIP - XZ')
%                     subplot(2,2,4), imagesc(max(tomo_stitch,[],3), [13370 13800]), axis image, colormap gray
%                     subplot(2,2,4), hold on
%                     plot([y_start,y_end], [x_start,x_start],'r-')
%                     plot([y_start,y_end],[x_end,x_end], 'r-')
%                     plot([y_start,y_start],[x_start,x_end], 'r-')
%                     plot([y_end,y_end],[x_start,x_end], 'r-')
%                     title(fname_save)
%                     drawnow
%                     
%                     pause(0.1)
%                     hold off
%                     

                    set(0, 'CurrentFigure', h_),subplot(1,2,2),hold on
                    plot([y_start,y_end], [x_start,x_start],'r-')
                    plot([y_start,y_end],[x_end,x_end], 'r-')
                    plot([y_start,y_start],[x_start,x_end], 'r-')
                    plot([y_end,y_end],[x_start,x_end], 'r-')
                    saveas(h_, ['00_colony_' fname_save, '.png'])
                end
            end
%             pause()
        end
    end
    
end