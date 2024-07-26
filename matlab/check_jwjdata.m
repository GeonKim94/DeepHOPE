
clear
clc
close all

h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')

dir_source = '/data02/gkim/RI2FL_Postech/data/231207_RI2FL_TCF/';
dir_feat = '/workspace01/gkim/stem_cell_jwshin/data/231207_RI2FL_feat_wider_v3'; 
dir_mip = '/workspace01/gkim/stem_cell_jwshin/data/231207_RI2FL_MIPH5_wider_v3'; 
dir_sec1 = '/workspace01/gkim/stem_cell_jwshin/data/231207_RI2FL_SEC1H5_wider_v3'; 
dir_img = '/workspace01/gkim/stem_cell_jwshin/data/231207_RI2FL_SEC1PNG_wider_v3';


cd(dir_source)
list_cls = dir('sh*');
processing = true;

classes = [];
sizes = [];
fnames = {};
z_size = 16;
hour = 0;
for iter_cls = 1:length(list_cls)
    dir_cls = list_cls(iter_cls).name;
    cd(dir_source)
    cd(dir_cls)
    list_date = dir('2*');

    for iter_date = 1:length(list_date)
        dir_date = list_date(iter_date).name;
        cd(dir_source)
        cd(dir_cls)
        cd(dir_date)
        list_meas = dir('2*');
        list_meas = list_meas(find([list_meas.isdir]));

        for iter_meas = 1:length(list_meas)
            dir_meas = list_meas(iter_meas).name;
            cd(dir_source)
            cd(dir_cls)
            cd(dir_date)
            cd(dir_meas)
            
            list_tcf = dir('*.TCF');
            if isempty(list_tcf)
                continue
            end

            fname_tcf = list_tcf(1).name;
            if ~isfile(list_tcf(1).name)
                continue
            end
        
            if list_tcf(1).bytes < 10^6
                continue
            end

            try
                info_temp = h5info(fname_tcf, '/Info/Device');
                n_m = uint16(info_temp.Attributes(3).Value*10000);
    %             tomo_stitch = h5read(fname_tcf, sprintf('/Data/3D/%06d', hour));
                
                info_temp = h5info(fname_tcf, '/Data/3D');         
                size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
                res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];
    
                info_temp = h5info(fname_tcf, '/Info/Device');
                n_m = uint16(info_temp.Attributes(3).Value*10000);
    
             
                tomo_stitch = h5read(fname_tcf, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 

                classes = [classes; iter_cls];
                sizes = [sizes; size_ori];

                fnames{end+1} = fname_tcf;

            catch
                continue
            end
               
            if ~processing
                continue
            end

            
            idx_dot = strfind(fname_tcf, '.');
            fname_save = fname_tcf(1:idx_dot(end)-1);

            mipz_stitch = max(tomo_stitch, [],3);
            mipx_stitch = squeeze(max(tomo_stitch, [],1));

            n_thres = 13420; %13370

            count_z = sum(single(mipx_stitch),1);
                
%             z_glass = gradient(count_z);
%             z_glass = find(z_glass == max(z_glass));

            count_z_sumz0 = [];
            z0s = [];
            for iter_z = 1:size(tomo_stitch,3)-z_size+1
                count_z_sumz0(iter_z) = sum(count_z(iter_z:iter_z+z_size-1));
            end
            
            z_glass = find(count_z_sumz0 == max(count_z_sumz0));




            z_sample = z_glass:1:z_glass+z_size-1;

            
%             se = strel('disk', round(1/res_ori(1)));
%             mask_stitch = mipz_stitch > n_thres;
%             mask_stitch = imdilate(mask_stitch,se);
%             mask_stitch = imfill(mask_stitch,'holes');
%             mask_stitch = imerode(mask_stitch,se);
%             [lbl,num] = bwlabeln(mask_stitch, 4);
%             if mask_stitch(round(end/2),round(end/2)) ~= 0 && sum(sum(lbl == mask_stitch(round(end/2),round(end/2))))*res_ori(1)^2>100*100
%                 lbl_colony = lbl(round(end/2),round(end/2));
%             else
%                 pixels_lbl = [];
%                 for idx_mask = 1:num
%                     pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
%                 end
%                 lbl_colony = find(pixels_lbl == max(pixels_lbl));
%             end
%             mask_stitch = (lbl==lbl_colony);
%             mask_stitch = imfill(mask_stitch,'holes');
%             
%             x_crop_ = find(sum(mask_stitch,2)>0);
%             x_crop_ = [max(min(x_crop_)-round(size_pad/res_ori(1)),1) min(max(x_crop_)+round(size_pad/res_ori(1)),size_ori(1))];
%             x_crop_ = single(x_crop_);
%             
%             y_crop_ = find(sum(mask_stitch,1)>0);
%             y_crop_ = [max(min(y_crop_)-round(size_pad/res_ori(1)),1) min(max(y_crop_)+round(size_pad/res_ori(1)),size_ori(2))];
%             y_crop_ = single(y_crop_);
% 
%             tomo_stitch = tomo_stitch(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2),:);
%             mask_stitch = mask_stitch(x_crop_(1):x_crop_(2),y_crop_(1):y_crop_(2));
            
            %% save mip img

            data = max((tomo_stitch),[],3);
            

            mkdir([dir_mip , '/',...
            dir_cls , '/',...
            dir_date, '/',...
            ])

            cd([dir_mip , '/',...
            dir_cls , '/',...
            dir_date, '/',...
            ]);

            h5create([fname_save,'.h5'], '/ri', size(data));
            h5write([fname_save,'.h5'], '/ri', data);


            %% save sec1 img
            data = tomo_stitch(:,:,z_sample);


            mkdir([dir_sec1 , '/',...
            dir_cls , '/',...
            dir_date, '/',...
            ])

            cd([dir_sec1 , '/',...
            dir_cls , '/',...
            dir_date, '/',...
            ]);


            h5create([fname_save,'.h5'], '/ri', size(data));
            h5write([fname_save,'.h5'], '/ri', data);

            set(0, 'CurrentFigure', h_), hold off

            subplot(1,2,1), imagesc(data(:,:,1), [13300 13800]), axis image, colormap gray
            subplot(1,2,2), imagesc(data(:,:,end), [13300 13800]), axis image, colormap gray

            mkdir([dir_img, '/',...
            dir_cls , '/',...
            dir_date, '/',...
            ])
            cd([dir_img, '/',...
            dir_cls , '/',...
            dir_date, '/',...
            ])
            saveas(h_, [fname_save, '.png'])
        end


    end

end

fnames = fnames';


%% split sec1

for iter_cls = 1:length(unique(classes))
    
    %% find test & val data
    class_ = classes(iter_cls);
    num_cls = sum(classes == class_);
    idxs_cls = find(classes == class_);

    num_testval = floor(num_cls/10);

    idxs_test = idxs_cls(randperm(num_cls, num_testval*2));
    while max(max(sizes(idxs_test,1:2))) > 1024*3
        idxs_test = idxs_cls(randperm(num_cls, num_testval*2));
    end

    idxs_val = idxs_test(1:num_testval);
    idxs_test = idxs_test(num_testval+1:2*num_testval);

    

    idxs_train = setdiff(idxs_cls, [idxs_test;idxs_val]);

    %% move folders

    for iter_train = 1:length(idxs_train)

        idx_train = idxs_train(iter_train);
        [path_h5, found] = search_recursive_v2(dir_sec1,erase(fnames{idx_train},'.TCF'),false);        
        dir_h5_new = [dir_sec1 '/00_train/' ...
            sprintf('%02d_', classes(idx_train)-1) list_cls(classes(idx_train)).name '/'];
        mkdir(dir_h5_new)
        movefile(path_h5,dir_h5_new);

    end

    for iter_val = 1:length(idxs_val)

        idx_val = idxs_val(iter_val);
        [path_h5, found] = search_recursive_v2(dir_sec1,erase(fnames{idx_val},'.TCF'),false);        
        dir_h5_new = [dir_sec1 '/01_val/' ...
            sprintf('%02d_', classes(idx_val)-1) list_cls(classes(idx_val)).name '/'];
        mkdir(dir_h5_new)
        movefile(path_h5,dir_h5_new);

    end

    for iter_test = 1:length(idxs_test)

        idx_test = idxs_test(iter_test);
        [path_h5, found] = search_recursive_v2(dir_sec1,erase(fnames{idx_test},'.TCF'),false);        
        dir_h5_new = [dir_sec1 '/02_test/' ...
            sprintf('%02d_', classes(idx_test)-1) list_cls(classes(idx_test)).name '/'];
        mkdir(dir_h5_new)
        movefile(path_h5,dir_h5_new);

    end


end
%% split mip

for iter_cls = 1:length(unique(classes))
    
    %% find test & val data
    class_ = classes(iter_cls);
    num_cls = sum(classes == class_);
    idxs_cls = find(classes == class_);

    num_testval = floor(num_cls/10);

    idxs_test = idxs_cls(randperm(num_cls, num_testval*2));
    while max(max(sizes(idxs_test,1:2))) > 1024*3
        idxs_test = idxs_cls(randperm(num_cls, num_testval*2));
    end

    idxs_val = idxs_test(1:num_testval);
    idxs_test = idxs_test(num_testval+1:2*num_testval);

    

    idxs_train = setdiff(idxs_cls, [idxs_test;idxs_val]);

    %% move folders

    for iter_train = 1:length(idxs_train)

        idx_train = idxs_train(iter_train);
        [path_h5, found] = search_recursive_v2(dir_mip,erase(fnames{idx_train},'.TCF'),false);        
        dir_h5_new = [dir_mip '/00_train/' ...
            sprintf('%02d_', classes(idx_train)-1) list_cls(classes(idx_train)).name '/'];
        mkdir(dir_h5_new)
        movefile(path_h5,dir_h5_new);

    end

    for iter_val = 1:length(idxs_val)

        idx_val = idxs_val(iter_val);
        [path_h5, found] = search_recursive_v2(dir_mip,erase(fnames{idx_val},'.TCF'),false);        
        dir_h5_new = [dir_mip '/01_val/' ...
            sprintf('%02d_', classes(idx_val)-1) list_cls(classes(idx_val)).name '/'];
        mkdir(dir_h5_new)
        movefile(path_h5,dir_h5_new);

    end

    for iter_test = 1:length(idxs_test)

        idx_test = idxs_test(iter_test);
        [path_h5, found] = search_recursive_v2(dir_mip,erase(fnames{idx_test},'.TCF'),false);        
        dir_h5_new = [dir_mip '/02_test/' ...
            sprintf('%02d_', classes(idx_test)-1) list_cls(classes(idx_test)).name '/'];
        mkdir(dir_h5_new)
        movefile(path_h5,dir_h5_new);

    end


end