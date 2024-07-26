%% extract features

dir_source = '/data02/gkim/stem_cell_jwshin/data/220822_TCF';%220822_TCF';
dir_save = '/data02/gkim/stem_cell_jwshin/data/230220_feature';
cd(dir_source);
dir_set = dir('0*');


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
                resx = h5readatt(fname, '/Data/3D', 'ResolutionX');
                resy = h5readatt(fname, '/Data/3D', 'ResolutionY');
                resz = h5readatt(fname, '/Data/3D', 'ResolutionZ');
                
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

                                mip_tempv1 = squeeze(max(single(tomo_temp).*repmat(mask_mip,[1 1 size(tomo_temp,3)]),[],1))';
                mip_tempv2 = squeeze(max(single(tomo_temp).*repmat(mask_mip,[1 1 size(tomo_temp,3)]),[],2))';
                mask_mipv1 = mip_tempv1 > 13420;
                [mask_mipv1, num_obj] = bwlabeln(mask_mipv1);
                size_obj = [];
                for iter_obj = 1:num_obj
                    size_obj(iter_obj) = sum(sum(sum(mask_mipv1 == iter_obj)));
                end
                mask_mipv1 = mask_mipv1 == find(size_obj == max(size_obj));

                mask_mipv2 = mip_tempv2 > 13420;
                mask_mipv2 = bwlabeln(mask_mipv2);t
                [mask_mipv2, num_obj] = bwlabeln(mask_mipv2);
                size_obj = [];
                for iter_obj = 1:num_obj
                    size_obj(iter_obj) = sum(sum(sum(mask_mipv2 == iter_obj)));
                end
                mask_mipv2 = mask_mipv2 == find(size_obj == max(size_obj));

                bot_v1 = [];
                top_v1 = [];
                bot_v2 = [];
                top_v2 = [];
                for iter_h = 1:size(mask_mipv1,2)
                    bot_v1 = [bot_v1 min(find(mask_mipv1(:,iter_h)))];
                    top_v1 = [top_v1 max(find(mask_mipv1(:,iter_h)))];
                end
                for iter_h = 1:size(mask_mipv2,2)
                    bot_v2 = [bot_v2 min(find(mask_mipv2(:,iter_h)))];
                    top_v2 = [top_v2 max(find(mask_mipv2(:,iter_h)))];
                end

                coor_bot = mode([bot_v1 bot_v2]);

                mask_3D = (single(tomo_temp).*repmat(mask_mip,[1 1 size(tomo_temp,3)])) > 13420;
                mask_3D = mask_3D.*repmat(permute(mask_mipv1', [3 1 2]),[size(tomo_temp,1) 1 1]);
                mask_3D = mask_3D.*repmat(permute(mask_mipv2', [1 3 2]),[1 size(tomo_temp,2) 1]);
                mask_3D(:,:, 1:coor_bot-1) = 0;
                    
                mask_3D = logical(mask_3D);

                map_thick = sum(single(mask_3D),3);
   
                stats = regionprops(mask_mip, 'Centroid', 'MajorAxisLength', ...
                    'MinorAxisLength');
                coor_crd = stats.Centroid;
                coor_crd = fliplr(coor_crd);      

                [XX, YY] = find(mask_mip);
                XY = [XX YY];
%                 surfit = @(B,XY)  B(1)+B(2)*XY(:,1)+B(3)*XY(:,2); 
%                 B = lsqcurvefit(surfit, [mean_thick 0 0], XY, map_thick(mask_mip));

                coor_com = [sum(XX.*map_thick(mask_mip))/sum(map_thick(mask_mip))...
                    sum(YY.*map_thick(mask_mip))/sum(map_thick(mask_mip))];

                RR = sqrt((XX-coor_com(1)).^2+(YY-coor_com(2)).^2);
                
                n_low = mip_temp;
                n_low(~mask_interior) = 0;

                mask_nuc = n_low<13450;
                se2 = strel('disk',1);
                mask_nuc = imerode(mask_nuc,se2);
                mask_nuc = imdilate(mask_nuc,se2);
                mask_nuc = mask_nuc.*mask_interior;

                mask_lip = mip_temp > 13675;
                mask_lip = mask_exterior.*mask_lip;
                se3 = strel('disk',4);
                mask_lip = mask_lip - imdilate(imdilate(imerode(mask_lip,se3),se3),se3);
                mask_lip(mask_lip<0) = 0;
                [label_lip, num_lip] = bwlabel(mask_lip);

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
                
                thick_mean = sum(sum(map_thick.*mask_mip))/sum(sum(mask_mip)).*resz;
                thick_std = std(map_thick(mask_mip>0),0,1)*resz;     
                
                center_disp = sum(((coor_crd-coor_com).*[resx resy]).^2);

                spread_thick = sqrt(sum(map_thick(mask_mip).*(RR).^2)/sum(map_thick(mask_mip)));
                
                skew_thick = [(sum(map_thick(mask_mip).*(XX - coor_com(1)).^3)/sum(map_thick(mask_mip)))...
                    (sum(map_thick(mask_mip).*(YY - coor_com(2)).^3)/sum(map_thick(mask_mip)))]...
                    /spread_thick^3;

                kurt_thick = (sum(map_thick(mask_mip).*(RR ).^4)/sum(map_thick(mask_mip)))...
                    /spread_thick^4;

                spread_thick = spread_thick*resx;

                list_thick = sort(map_thick(mask_mip),'descend');
                thick_peak = mean(list_thick(1:round(0.05*end)))*resz;

                RI_mean = mean(single(tomo_temp(mask_3D))/10000);
                RI_std = std(single(tomo_temp(mask_3D))/10000,0,1);
                
                area_col = sum(sum(mask_mip))*resx*resy;
                area_nuc = sum(sum(mask_nuc))*resx*resy;
                area_lip = sum(sum(mask_lip))*resx*resy;
                r_major = stats.MajorAxisLength*resx;
                r_minor = stats.MinorAxisLength*resx;

                mkdir(dir_save);
                cd(dir_save);
                mkdir(dir_set(iter_set).name);
                cd(dir_set(iter_set).name);
                mkdir(dir_cls(iter_cls).name);
                cd(dir_cls(iter_cls).name);

%                 figure(1)
%                 subplot(1,2,1), imagesc(mip_temp, [13370 13800]), axis image;
%                 set(gca, 'Colormap',gray)
%                 subplot(1,4,2), imagesc(mask_mip), axis image;
%                 set(gca, 'Colormap',gray)
%                 subplot(1,4,3), imagesc(mask_nuc), axis image;
%                 set(gca, 'Colormap',gray)
%                 subplot(1,2,2), imagesc(mask_lip), axis image;
%                 set(gca, 'Colormap',gray)
                
                save(strrep(fname, '.TCF','.mat'), 'smoothness', 'ncont_bound',...
                    'area_col', 'area_nuc', 'area_lip','num_lip',...
                    'thick_mean','thick_std','thick_peak',...
                    'center_disp', 'spread_thick', 'skew_thick', 'kurt_thick', ...
                    'RI_mean', 'RI_std');

        end
    end
end
