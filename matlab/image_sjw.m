homedir='/data02/gkim/stem_cell_jwshin/data/230306_TCF/00_train/A2_2/230116.114309.A2_2.007.Group1.A1.S007';
cd(homedir)
tcfdata=dir('*.TCF');
data=h5read(tcfdata.name,'/Data/3D/000000');
bf=h5read(tcfdata.name,'/Data/BF/000000');
resx = h5readatt(tcfdata.name, '/Data/3D', 'ResolutionX');
resy = h5readatt(tcfdata.name, '/Data/3D', 'ResolutionY');
resz = h5readatt(tcfdata.name, '/Data/3D', 'ResolutionZ');
orthosliceViewer(data, 'DisplayRange',[13300 13800])
data2=data(150:450, 500:800,25:55);
bf2=bf(150/size(data,1)*size(bf,1):150/size(data,1)*size(bf,1)+300/size(data,1)*size(bf,1), 500/size(data,1)*size(bf,1):500/size(data,1)*size(bf,1)+300/size(data,1)*size(bf,1),:);
figure(1), imagesc(squeeze(max(data,[],3)), [13300 13800]), axis image, colormap gray, axis off
hold on
rectangle('Position',[500/size(data,1)*size(bf,1) 150/size(data,1)*size(bf,1) 300 300], 'EdgeColor','w','LineWidth',1.5)
figure(2), 
yzi=squeeze(max(data2,[],1));
[h,w]=size(yzi);
yzii=imresize(yzi, [h, w*6.278]);
xzi=squeeze(max(data2,[],2));
[h,w]=size(xzi);
xzii=imresize(xzi, [h, w*6.278]);
subplot(2,2,3); imagesc(squeeze(max(data2,[],3)), [13300 13800]), axis image, colormap gray, axis off
subplot(2,2,1); imagesc(yzii, [13300 13800]), axis image, colormap gray, axis off
subplot(2,2,4); imagesc(xzii, [13300 13800]), axis image, colormap gray, axis off