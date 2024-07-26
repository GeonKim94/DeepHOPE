img = h5read('/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA/test/05_ctl/230921.144000.GM25256_untreated.009.Group1.A1.S009.h5','/ri');
img = h5read('/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA/test/02_ecto/231211.125427.GM25256_Ectoderm_12h.045.Group1.A1.S045.h5','/ri');
img = h5read('/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA/test/03_meso/231211.090435.GM25256_Mesoderm_12h.020.Group1.A1.S020.h5','/ri');
img = h5read('/data02/gkim/stem_cell_jwshin/data/23_MIPH5_wider_v3_allh_onRA/test/04_endo/231211.072507.GM25256_Endoderm_12h.031.Group1.A1.S031.h5','/ri');


close all

imagesc(img, [13300 14000]), axis image, colormap gray;