%% untreated line GM
clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
hour = 0;
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

fname = ['/data02/gkim/stem_cell_jwshin/data/230502_TCF/',...
    '00_train/GM25256_untreated/230426.150725.GM25256_untreated.025.Group1.A1.S025/',...
    '230426.150725.GM25256_untreated.025.Group1.A1.S025.TCF'];


            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);

            idx_slash = strfind(fname,'/');
            idx_slash = idx_slash(5);
            if str2num(fname(idx_slash+1:idx_slash+6)) < 230701
                tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 
            else
                tomo_stitch = ReadLDMTCFHT(fname, hour);
                tomo_stitch = single(tomo_stitch*10000);
                tomo_stitch = permute(tomo_stitch, [2 1 3]);
                tomo_stitch(tomo_stitch == 0) = n_m; 
            end
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

x_crop = 1024;
y_crop = 1536;
tomo_stitch = tomo_stitch(ceil(end/2+1/2)-x_crop*5/8+1:ceil(end/2+1/2)+x_crop*3/8,...
    ceil(end/2+1/2)-y_crop/2:ceil(end/2+1/2)+y_crop/2-1,...
    :);

set(0, 'CurrentFigure', h_), hold off
subplot(1,2,1)
imagesc(max(tomo_stitch,[],3)', [13300 13800]), axis image, colormap gray


roi = 256;

subplot(3,2,2);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+2),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,4);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+6),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,6);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+10),roi,roi)', [13300 13800]), axis image, colormap gray

saveas(h_,'img_untreated_poster.fig')

%% 12h line GM
clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
hour = 0;
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

fname = ['/data02/gkim/stem_cell_jwshin/data/230502_TCF/',...
    '00_train/GM25256_12hour_treat/230426.180831.GM25256_12hour_treat.053.Group1.A1.S053/',...
    '230426.180831.GM25256_12hour_treat.053.Group1.A1.S053.TCF'];


            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);

            idx_slash = strfind(fname,'/');
            idx_slash = idx_slash(5);
            if str2num(fname(idx_slash+1:idx_slash+6)) < 230701
                tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 
            else
                tomo_stitch = ReadLDMTCFHT(fname, hour);
                tomo_stitch = single(tomo_stitch*10000);
                tomo_stitch = permute(tomo_stitch, [2 1 3]);
                tomo_stitch(tomo_stitch == 0) = n_m; 
            end
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

x_crop = 1024;
y_crop = 1536;
tomo_stitch = tomo_stitch(ceil(end/2+1/2)-x_crop*4/8+1:ceil(end/2+1/2)+x_crop*4/8,...
    ceil(end/2+1/2)-y_crop/2:ceil(end/2+1/2)+y_crop/2-1,...
    :);

set(0, 'CurrentFigure', h_), hold off
subplot(1,2,1)
imagesc(max(tomo_stitch,[],3)', [13300 13800]), axis image, colormap gray


roi = 256;

subplot(3,2,2);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+3),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,4);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+7),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,6);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+11),roi,roi)', [13300 13800]), axis image, colormap gray


saveas(h_,'img_treated12_poster.fig')

%% 24h line GM
clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
hour = 0;
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

fname = ['/data02/gkim/stem_cell_jwshin/data/230502_TCF/',...
    '00_train/GM25256_24hour_treat/230426.195203.GM25256_24hour_treat.036.Group1.A1.S036/',...
    '230426.195203.GM25256_24hour_treat.036.Group1.A1.S036.TCF'];


            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);

            idx_slash = strfind(fname,'/');
            idx_slash = idx_slash(5);
            if str2num(fname(idx_slash+1:idx_slash+6)) < 230701
                tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 
            else
                tomo_stitch = ReadLDMTCFHT(fname, hour);
                tomo_stitch = single(tomo_stitch*10000);
                tomo_stitch = permute(tomo_stitch, [2 1 3]);
                tomo_stitch(tomo_stitch == 0) = n_m; 
            end
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

x_crop = 1024;
y_crop = 1536;
tomo_stitch = tomo_stitch(ceil(end/2+1/2)-x_crop*4/8+1:ceil(end/2+1/2)+x_crop*4/8,...
    ceil(end/2+1/2)-y_crop*7/16:ceil(end/2+1/2)+y_crop*9/16-1,...
    :);

set(0, 'CurrentFigure', h_), hold off
subplot(1,2,1)
imagesc(max(tomo_stitch,[],3)', [13300 13800]), axis image, colormap gray


roi = 256;

subplot(3,2,2);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+1),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,4);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+5),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,6);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+9),roi,roi)', [13300 13800]), axis image, colormap gray

saveas(h_,'img_treated24_poster.fig')
%% A2 iPSC
clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
hour = 0;
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

fname = ['/data02/gkim/stem_cell_jwshin/data/230306_TCF/',...
    '00_train/A2_/230116.110645.A2.011.Group1.A1.S011/',...
    '230116.110645.A2.011.Group1.A1.S011.TCF'];


            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);

            idx_slash = strfind(fname,'/');
            idx_slash = idx_slash(5);
            if str2num(fname(idx_slash+1:idx_slash+6)) < 230701
                tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 
            else
                tomo_stitch = ReadLDMTCFHT(fname, hour);
                tomo_stitch = single(tomo_stitch*10000);
                tomo_stitch = permute(tomo_stitch, [2 1 3]);
                tomo_stitch(tomo_stitch == 0) = n_m; 
            end
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

x_crop = 1024;
y_crop = 1536;
tomo_stitch = tomo_stitch(ceil(end/2+1/2)-x_crop*4/8+1:ceil(end/2+1/2)+x_crop*4/8,...
    ceil(end/2+1/2)-y_crop*9/16:ceil(end/2+1/2)+y_crop*7/16-1,...
    :);

set(0, 'CurrentFigure', h_), hold off
subplot(1,2,1)
imagesc(max(tomo_stitch,[],3)', [13300 13800]), axis image, colormap gray


roi = 256;

subplot(3,2,2);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+4),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,4);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+8),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,6);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+12),roi,roi)', [13300 13800]), axis image, colormap gray

saveas(h_,'img_A2_poster.fig')

%% A19 iPSC
clear
clc
close all


addpath('/data01/gkim/Matlab_subcodes/gk')
addpath('/data01/gkim/Matlab_subcodes')
hour = 0;
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

fname = ['/data02/gkim/stem_cell_jwshin/data/230306_TCF/',...
    '00_train/A19_2/230111.112402.A19_2.015.Group1.A1.S015/',...
    '230111.112402.A19_2.015.Group1.A1.S015.TCF'];


            info_temp = h5info(fname, '/Info/Device');
            n_m = uint16(info_temp.Attributes(3).Value*10000);

            idx_slash = strfind(fname,'/');
            idx_slash = idx_slash(5);
            if str2num(fname(idx_slash+1:idx_slash+6)) < 230701
                tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
                tomo_stitch(tomo_stitch == 0) = n_m; 
            else
                tomo_stitch = ReadLDMTCFHT(fname, hour);
                tomo_stitch = single(tomo_stitch*10000);
                tomo_stitch = permute(tomo_stitch, [2 1 3]);
                tomo_stitch(tomo_stitch == 0) = n_m; 
            end
%             tomo_stitch = h5read(fname, sprintf('/Data/3D/%06d', hour));
            
            info_temp = h5info(fname, '/Data/3D');         
            size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
            res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];

x_crop = 1024;
y_crop = 1536;
tomo_stitch = tomo_stitch(ceil(end/2+1/2)-x_crop*7/16+1:ceil(end/2+1/2)+x_crop*9/16,...
    ceil(end/2+1/2)-y_crop*8/16:ceil(end/2+1/2)+y_crop*8/16-1,...
    :);

set(0, 'CurrentFigure', h_), hold off
subplot(1,2,1)
imagesc(max(tomo_stitch,[],3)', [13300 13800]), axis image, colormap gray


roi = 256;

subplot(3,2,2);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+2),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,4);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+6),roi,roi)', [13300 13800]), axis image, colormap gray

subplot(3,2,6);
imagesc(cencrop2d(tomo_stitch(:,:,round(end/2)+10),roi,roi)', [13300 13800]), axis image, colormap gray



saveas(h_,'img_A19_poster.fig')