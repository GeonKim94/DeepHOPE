%% cehck size
dir_source = '/data02/gkim/stem_cell_jwshin/data/220623_TCF';
cd(dir_source);
dir_set = dir('0*');
sizes_tomo = [];
vals = [];
ns = [];

lfits = [];

for iter_set = 1:length(dir_set)
    cd(dir_source);
    cd(dir_set(iter_set).name);
    dir_cls = dir('0*');
    
    for iter_cls = 1:length(dir_cls)
        cd(dir_source)
        cd(dir_set(iter_set).name);
        cd(dir_cls(iter_cls).name);
        
        dir_tcf = dir('*.TCF');
        
        for iter_tcf = 1:length(dir_tcf)
            fname = dir_tcf(iter_tcf).name;
            
            info_temp = h5info(fname, '/Data/3D/000000');
            
            sizes_tomo = [sizes_tomo; info_temp.Dataspace.Size];
            
            n_min = info_temp.Attributes(7).Value;
            n_max = info_temp.Attributes(6).Value ;
            
            tomo_temp = h5read(fname, '/Data/3D/000000');
            val_min = min(min(min(tomo_temp)));
            val_max = max(max(max(tomo_temp)));
            
            ns = [ns; [n_min n_max]];
            vals = [vals; [val_min val_max]];
            
            
            size(sizes_tomo,1);
        end
    end
end


cd(dir_source)
save('info_img.mat', 'sizes_tomo', 'ns', 'vals');

%% see data (MIP + slice)
dir_source = '/data02/gkim/stem_cell_jwshin/data/220623_TCF';
cd(dir_source);
dir_set = dir('0*');
sizes_tomo = [];
for iter_set = 1:length(dir_set)
    cd(dir_source);
    cd(dir_set(iter_set).name);
    dir_cls = dir('0*');
    
    for iter_cls = 1:length(dir_cls)
        cd(dir_source)
        cd(dir_set(iter_set).name);
        cd(dir_cls(iter_cls).name);
        
        dir_tcf = dir('*.TCF');
        
            for iter_tcf = 1:length(dir_tcf)
                cd(dir_source)
                cd(dir_set(iter_set).name);
                cd(dir_cls(iter_cls).name);
                fname = dir_tcf(iter_tcf).name;

                tomo_temp = h5read(fname, '/Data/3D/000000');

                figure(101)
                subplot(1,2,1)
                imagesc(tomo_temp(:,:,105), [min(vals(:,1)) max(vals(:,2))]), axis image, colormap jet
                title('slice')
                subplot(1,2,2)
                imagesc(max(tomo_temp,[],3), [min(vals(:,1)) max(vals(:,2))]), axis image, colormap jet
                title('MIP')

                cd(dir_source)
                mkdir('imgs');
                cd('imgs');
                saveas(gcf,strrep(fname, '.TCF', '.png'));
            
        end
    end
end


%% extract features

dir_source = '/data02/gkim/stem_cell_jwshin/data/220623_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/220623_feature';
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
        
        dir_tcf = dir('*.TCF');
        
            for iter_tcf = 1:length(dir_tcf)
                cd(dir_source)
                cd(dir_set(iter_set).name);
                cd(dir_cls(iter_cls).name);
                fname = dir_tcf(iter_tcf).name;

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
                
                
                n_low = mip_temp(mask_interior);
                n_low = sum(n_low<13420);
                nc_ratio = n_low/sum(sum(mask_interior));
                
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
        end
    end
end



%%
fname = '/data02/gkim/stem_cell_jwshin/20220502/20220502.115219.843.A19_1-014Stitching.TCF';
tomo = h5read(fname, '/Data/3D/000000');
%%

min_plot = min(min(min(tomo)));
max_plot = max(max(max(tomo)));
figure(303), imagesc(squeeze(tomo(:,:,:))', [min_plot max_plot]), axis image, colormap jet


%% 