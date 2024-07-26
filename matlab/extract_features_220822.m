%% extract features

dir_source = '/data02/gkim/stem_cell_jwshin/data/220822_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220822_feature';
cd(dir_source);
dir_set = dir('0*');
nc_ratio_ = [];
smoothness_ = [];
ncont_bound_ = [];
set_ = [];
cls_ = [];

for iter_set = 1:length(dir_set)
    cd(dir_source);
    cd(dir_set(iter_set).name);
    dir_cls = dir('0*');
    
    for iter_cls = 1:length(dir_cls)
        cd(dir_source)
        cd(dir_set(iter_set).name);
        cd(dir_cls(iter_cls).name);
        
        dir_tcf = dir('2*');
        
            for iter_tcf = 1:length(dir_tcf)
                cd(dir_source)
                cd(dir_set(iter_set).name);
                cd(dir_cls(iter_cls).name);
                cd(dir_tcf(iter_tcf).name);
                fname = [dir_tcf(iter_tcf).name '.TCF'];
                tomo_temp = h5read(fname, '/Data/3D/000000');
                
                mip_temp = max(tomo_temp,[],3);
                mask_mip = mip_temp > 13420;
                [lbl,num] = bwlabeln(mask_mip, 4);
                pixels_lbl = [];
                for idx_mask = 1:num
                    pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
                end
                mask_mip = (lbl==find(pixels_lbl == max(pixels_lbl)));
                mask_mip = imfill(mask_mip,'holes');
                se = strel('disk',20);
                mask_interior = imerode(mask_mip,se);
                mask_exterior = imdilate(mask_mip,se);
                
                
                n_low = mip_temp;
                n_low(~mask_interior) = 0;
                mask_nuc = n_low<13450;
                se2 = strel('disk',1);
                mask_nuc = imerode(mask_nuc,se2);
                mask_nuc = imdilate(mask_nuc,se2);
                mask_nuc = mask_nuc.*mask_interior;

                figure(1)
                subplot(1,3,1), imagesc(mip_temp, [13370 13800]), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,3,2), imagesc(mask_mip), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,3,3), imagesc(mask_nuc), axis image;
                set(gca, 'Colormap',gray)

                nc_ratio = sum(sum(mask_nuc))/sum(sum(mask_mip));
                
                num_bound = sum(sum(boundarymask(mask_mip)));%% see data (MIP + slice)
                num_smoothbound = sum(sum(boundarymask(imerode(mask_exterior,se))));%% see data (MIP + slice)
                
                smoothness = num_smoothbound/num_bound;

                n_boundin = sum(sum(single(mip_temp)/10000.*(mask_mip-mask_interior)))/sum(sum((mask_mip-mask_interior)));
                n_boundout = sum(sum(single(mip_temp)/10000.*(mask_exterior-mask_mip)))/sum(sum((mask_exterior-mask_mip)));

                ncont_bound = n_boundin-n_boundout;
                
                mkdir(dir_save);
                cd(dir_save);
                mkdir(dir_set(iter_set).name);
                cd(dir_set(iter_set).name);
                mkdir(dir_cls(iter_cls).name);
                cd(dir_cls(iter_cls).name);
                
                save(strrep(fname, '.TCF','.mat'), 'nc_ratio', 'smoothness', 'ncont_bound');
            
                nc_ratio_ = [nc_ratio_; nc_ratio];
                smoothness_ = [smoothness_; smoothness];
                ncont_bound_ = [ncont_bound_; ncont_bound];
                set_ = [set_; iter_set];
                cls_ = [cls_; iter_cls];
                pause(0.1)
        end
    end
end


%% see the example of feature extraction


dir_source = '/data02/gkim/stem_cell_jwshin/data/220822_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220822_feature';
cd(dir_source);
dir_set = dir('0*');
nc_ratio_ = [];
smoothness_ = [];
ncont_bound_ = [];
set_ = [];
cls_ = [];

for iter_set = 1%1:length(dir_set)
    cd(dir_source);
    cd(dir_set(iter_set).name);
    dir_cls = dir('0*');
    
    for iter_cls = 1%1:length(dir_cls)
        cd(dir_source)
        cd(dir_set(iter_set).name);
        cd(dir_cls(iter_cls).name);
        
        dir_tcf = dir('2*');
        
            for iter_tcf = 66%1:length(dir_tcf)
                cd(dir_source)
                cd(dir_set(iter_set).name);
                cd(dir_cls(iter_cls).name);
                cd(dir_tcf(iter_tcf).name);
                fname = [dir_tcf(iter_tcf).name '.TCF'];
                tomo_temp = h5read(fname, '/Data/3D/000000');
                
                mip_temp = max(tomo_temp,[],3);
                mask_mip = mip_temp > 13420;
                [lbl,num] = bwlabeln(mask_mip, 4);
                pixels_lbl = [];
                for idx_mask = 1:num
                    pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
                end
                mask_mip = (lbl==find(pixels_lbl == max(pixels_lbl)));
                mask_mip = imfill(mask_mip,'holes');
                se = strel('disk',20);
                mask_interior = imerode(mask_mip,se);
                mask_exterior = imdilate(mask_mip,se);
                
                
                n_low = mip_temp;
                n_low(~mask_interior) = 0;
                mask_nuc = n_low<13450;
                se2 = strel('disk',1);
                mask_nuc = imerode(mask_nuc,se2);
                mask_nuc = imdilate(mask_nuc,se2);
                mask_nuc = mask_nuc.*mask_interior;

                figure(1)
                subplot(1,3,1), imagesc(mip_temp, [13370 13800]), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,3,2), imagesc(mask_mip), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,3,3), imagesc(mask_nuc), axis image;
                set(gca, 'Colormap',gray)
                

                nc_ratio = sum(sum(mask_nuc))/sum(sum(mask_mip));
                
                num_bound = sum(sum(boundarymask(mask_mip)));%% see data (MIP + slice)
                num_smoothbound = sum(sum(boundarymask(imerode(mask_exterior,se))));%% see data (MIP + slice)
                
                smoothness = num_smoothbound/num_bound;

                n_boundin = sum(sum(single(mip_temp)/10000.*(mask_mip-mask_interior)))/sum(sum((mask_mip-mask_interior)));
                n_boundout = sum(sum(single(mip_temp)/10000.*(mask_exterior-mask_mip)))/sum(sum((mask_exterior-mask_mip)));

                ncont_bound = n_boundin-n_boundout;
                
                mkdir(dir_save);
                cd(dir_save);
                mkdir(dir_set(iter_set).name);
                cd(dir_set(iter_set).name);
                mkdir(dir_cls(iter_cls).name);
                cd(dir_cls(iter_cls).name);
                
                %save(strrep(fname, '.TCF','.mat'), 'nc_ratio', 'smoothness', 'ncont_bound');
            
                nc_ratio_ = [nc_ratio_; nc_ratio];
                smoothness_ = [smoothness_; smoothness];
                ncont_bound_ = [ncont_bound_; ncont_bound];
                set_ = [set_; iter_set];
                cls_ = [cls_; iter_cls];
                pause()
        end
    end
end
%% extract features

dir_source = '/data02/gkim/stem_cell_jwshin/data/220822_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220822_feature';
cd(dir_source);
dir_set = dir('0*');
nc_ratio_ = [];
smoothness_ = [];
ncont_bound_ = [];
set_ = [];
cls_ = [];

for iter_set = 1:length(dir_set)
    cd(dir_source);
    cd(dir_set(iter_set).name);
    dir_cls = dir('0*');
    
    for iter_cls = 1:length(dir_cls)
        cd(dir_source)
        cd(dir_set(iter_set).name);
        cd(dir_cls(iter_cls).name);
        
        dir_tcf = dir('2*');
        
            for iter_tcf = 1:length(dir_tcf)
                cd(dir_source)
                cd(dir_set(iter_set).name);
                cd(dir_cls(iter_cls).name);
                cd(dir_tcf(iter_tcf).name);
                fname = [dir_tcf(iter_tcf).name '.TCF'];
                tomo_temp = h5read(fname, '/Data/3D/000000');
                
                mip_temp = max(tomo_temp,[],3);
                mask_mip = mip_temp > 13420;
                [lbl,num] = bwlabeln(mask_mip, 4);
                pixels_lbl = [];
                for idx_mask = 1:num
                    pixels_lbl = [pixels_lbl sum(sum(lbl == idx_mask))];
                end
                mask_mip = (lbl==find(pixels_lbl == max(pixels_lbl)));
                mask_mip = imfill(mask_mip,'holes');
                se = strel('disk',20);
                mask_interior = imerode(mask_mip,se);
                mask_exterior = imdilate(mask_mip,se);
                
                
                n_low = mip_temp;
                n_low(~mask_interior) = 0;
                mask_nuc = n_low<13450;
                se2 = strel('disk',1);
                mask_nuc = imerode(mask_nuc,se2);
                mask_nuc = imdilate(mask_nuc,se2);
                mask_nuc = mask_nuc.*mask_interior;

                figure(1)
                subplot(1,3,1), imagesc(mip_temp, [13370 13800]), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,3,2), imagesc(mask_mip), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,3,3), imagesc(mask_nuc), axis image;
                set(gca, 'Colormap',gray)

                mkdir(dir_save);
                cd(dir_save);
                mkdir(dir_set(iter_set).name);
                cd(dir_set(iter_set).name);
                mkdir(dir_cls(iter_cls).name);
                cd(dir_cls(iter_cls).name);
                

                nc_ratio = sum(sum(mask_nuc))/sum(sum(mask_mip));
                
                num_bound = sum(sum(boundarymask(mask_mip)));%% see data (MIP + slice)
                num_smoothbound = sum(sum(boundarymask(imerode(mask_exterior,se))));%% see data (MIP + slice)
                
                smoothness = num_smoothbound/num_bound;

                n_boundin = sum(sum(single(mip_temp)/10000.*(mask_mip-mask_interior)))/sum(sum((mask_mip-mask_interior)));
                n_boundout = sum(sum(single(mip_temp)/10000.*(mask_exterior-mask_mip)))/sum(sum((mask_exterior-mask_mip)));

                ncont_bound = n_boundin-n_boundout;

                figure(2)
                subplot(1,4,1), imagesc(mask_exterior-mask_mip), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,4,2), imagesc(mask_mip-mask_interior), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,4,3), imagesc(boundarymask(mask_mip)), axis image;
                set(gca, 'Colormap',gray)
                subplot(1,4,4), imagesc(boundarymask(imerode(mask_exterior,se))), axis image;
                set(gca, 'Colormap',gray)

                saveas(gcf,strrep(fname, '.TCF','.fig'));
                pause(0.1)
        end
    end
end
