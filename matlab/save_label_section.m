clear
clc

dir_input_3d = '/data02/gkim/stem_cell_jwshin/data/_cellpose/input_tiff';
cd(dir_input_3d)
dir_date = '230502_tiff_wider';
cd(dir_date)
dir_set = '00_train';
cd(dir_set)
dir_cls = 'GM25256_12hour_treat';
cd(dir_cls)

dir_input_section = '/data02/gkim/stem_cell_jwshin/data/_cellpose/input_tiff_section';

mkdir([dir_input_section '/' dir_date '/' dir_set '/' dir_cls])

fname_input = '230426.181846.GM25256_12hour_treat.061.Group1.A1.S061.tiff';
% label_manual = h5read(fname_input,'/exported_data');
% label_manual = squeeze(permute(label_manual,[3,2,4,1]));
ri = tiffreadVolume(fname_input);

for iter_z = 1:size(ri,3)
    ri_section = ri(:,:,iter_z);
    cd([dir_input_section '/' dir_date '/' dir_set '/' dir_cls])
    imwrite(ri_section, strrep(fname_input,'.tiff', sprintf('_%02d.tiff', iter_z)));
end


%%
dir_label_3d = '/data02/gkim/stem_cell_jwshin/data/_cellpose/label_manual_tiff';
cd(dir_label_3d)
dir_date = '230502_tiff_wider';
cd(dir_date)
dir_set = '00_train';
cd(dir_set)
dir_cls = 'GM25256_12hour_treat';
cd(dir_cls)

dir_label_section = '/data02/gkim/stem_cell_jwshin/data/_cellpose/label_manual_tiff_section';

mkdir([dir_label_section '/' dir_date '/' dir_set '/' dir_cls])

fname_label = '230426.181846.GM25256_12hour_treat.061.Group1.A1.S061.h5';
label_manual = h5read(fname_label,'/exported_data');
label_manual = squeeze(permute(label_manual,[3,2,4,1]));

for iter_z = 1:size(label_manual,3)
    label_manual_section = label_manual(:,:,iter_z);
    cd([dir_label_section '/' dir_date '/' dir_set '/' dir_cls])
    imwrite(label_manual_section, strrep(fname_label,'.h5', sprintf('_%02d.tiff', iter_z)));
end