%% name of RI file
fname = '/data02/gkim/stem_cell_jwshin/data_render/220808.143158.mDP.001.Group1.A1.S001.TCF';

%% load RI
info_temp = h5info(fname, '/Info/Device');
n_m = uint16(info_temp.Attributes(3).Value*10000);
tomo_stitch = h5read(fname, '/Data/3D/000000');
tomo_stitch(tomo_stitch == 0) = n_m; 
info_temp = h5info(fname, '/Data/3D');         
size_ori = [info_temp.Attributes(7).Value info_temp.Attributes(8).Value info_temp.Attributes(9).Value];
res_ori = [info_temp.Attributes(2).Value info_temp.Attributes(3).Value info_temp.Attributes(4).Value];
tomo_stitch_MIP = max(tomo_stitch, [], 3);


figure(1), imagesc(tomo_stitch_MIP, [13360 13650]), axis image, colormap gray 
%% crop RI


tomo_stitch_MIP_crop = tomo_stitch_MIP(4255:6489,1498:3718);
figure(2), imagesc(tomo_stitch_MIP_crop, [13330 13800]), axis image, colormap gray 


%% save crop RI image
figure(2), saveas(gcf,'/data02/gkim/stem_cell_jwshin/data_render/figure_neuron_RI_MIP.fig');


%% name of FL file
fname = '/data02/gkim/stem_cell_jwshin/data_render/220811.103126.mDP.022.Group1.A1.S022.TCF';

%% load FL
info_temp = h5info(fname, '/Info/Device');
n_m = uint16(info_temp.Attributes(3).Value*10000);
FL0_stitch = single(h5read(fname, '/Data/3DFL/CH0/000000')); %blue channel
FL1_stitch = single(h5read(fname, '/Data/3DFL/CH1/000000')); %green channel
FL2_stitch = single(h5read(fname, '/Data/3DFL/CH2/000000')); %red channel


val0_min = 0; %blue minimum
val1_min = 0; %green minimum
val2_min = 10; %red minimum
val0_max = 70; %blue maximum
val1_max = 50; %green maximum
val2_max = 12; %red maximum


FL_stitch = cat(3, (FL2_stitch-val2_min)/(val2_max-val2_min), ...
    cat(3, (FL1_stitch-val1_min)/(val1_max-val1_min), ...
    (FL0_stitch-val0_min)/(val0_max-val0_min) ));

FL_stitch = min(FL_stitch,1);

info_temp = h5info(fname, '/Data/3DFL');     
size_ori = [info_temp.Attributes(10).Value info_temp.Attributes(11).Value info_temp.Attributes(12).Value];
res_ori = [info_temp.Attributes(5).Value info_temp.Attributes(6).Value info_temp.Attributes(7).Value];


figure(3), imshow(FL_stitch, [0 1]), axis image
%% crop FL


FL_stitch_crop = FL_stitch(6197:9393,2606:5816,:);
figure(4), imshow(FL_stitch_crop, [0 1]), axis image


%% save crop FL image
figure(4), saveas(gcf,'/data02/gkim/stem_cell_jwshin/data_render/figure_neuron_FL_brighter.fig');