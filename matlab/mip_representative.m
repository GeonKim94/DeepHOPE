%% endo 12h.fig'), size(data,2)

data = h5read(['/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA'...
    '/train'...
    '/04_endo/'...
    '231211.075504.GM25256_Endoderm_12h.045.Group1.A1.S045.h5'],...
    '/ri');


close all
h_ = figure(1);
h_.Position = [0 200 1000 1000];
h_.Color = [1 1 1];

imagesc(data, [13300 13800]), axis image, colormap gray
saveas(h_,'/data02/gkim/stem_cell_jwshin/endo12hmip.fig'), size(data,2)
%% endo 24h

data = h5read(['/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA'...
    '/train'...
    '/04_endo/'...
    '231211.153343.GM25256_Endoderm_24h.030.Group1.A1.S030.h5'],...
    '/ri');


close all
h_ = figure(1);
h_.Position = [0 200 1000 1000];
h_.Color = [1 1 1];

imagesc(data, [13300 13800]), axis image, colormap gray
saveas(h_,'/data02/gkim/stem_cell_jwshin/endo24hmip.fig'), size(data,2)


%% meso 12h

data = h5read(['/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA'...
    '/train'...
    '/03_meso/'...
    '231211.093747.GM25256_Mesoderm_12h.040.Group1.A1.S040.h5'],...
    '/ri');


close all
h_ = figure(1);
h_.Position = [0 200 1000 1000];
h_.Color = [1 1 1];

imagesc(data, [13300 13800]), axis image, colormap gray
saveas(h_,'/data02/gkim/stem_cell_jwshin/meso12hmip.fig'), size(data,2)

%% meso 24h

data = h5read(['/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA'...
    '/train'...
    '/03_meso/'...
    '231211.194646.GM25256_Mesoderm_24h.052.Group1.A1.S052.h5'],...
    '/ri');


close all
h_ = figure(1);
h_.Position = [0 200 1000 1000];
h_.Color = [1 1 1];

imagesc(data, [13300 13800]), axis image, colormap gray
saveas(h_,'/data02/gkim/stem_cell_jwshin/meso24hmip.fig'), size(data,2)


%% ecto 12h

data = h5read(['/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA'...
    '/train'...
    '/02_ecto/'...
    '231211.111212.GM25256_Ectoderm_12h.028.Group1.A1.S028.h5'],...
    '/ri');


close all
h_ = figure(1);
h_.Position = [0 200 1000 1000];
h_.Color = [1 1 1];

imagesc(data, [13300 13800]), axis image, colormap gray
saveas(h_,'/data02/gkim/stem_cell_jwshin/ecto12hmip.fig'), size(data,2)

%% ecto 24h

data = h5read(['/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA'...
    '/train'...
    '/02_ecto/'...
    '231211.205841.GM25256_Ectoderm_24h.024.Group1.A1.S024.h5'],...
    '/ri');


close all
h_ = figure(1);
h_.Position = [0 200 1000 1000];
h_.Color = [1 1 1];

imagesc(data, [13300 13800]), axis image, colormap gray
saveas(h_,'/data02/gkim/stem_cell_jwshin/ecto24hmip.fig'), size(data,2)