clear
close all
clc

dir_save = '/data02/gkim/stem_cell_jwshin/data/230811_MIPH5_wider'; %220902_3D';%220803_3D';
dir_fig = '/data02/gkim/stem_cell_jwshin/data/230811_MIPFIG_wider';
cd(dir_save)
dir_set = dir('0*');
h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];
h = figure(5);
h.Position = [595 327 950 1000];
h.Color = [1 1 1];

size_pad = 20; 

hours = 0;%0:1:24;

for hour = hours

for iter_set = length(dir_set):-1:1
    cd(dir_save)
    cd(dir_set(iter_set).name)
    dir_cls = dir('*_*');
    
    if iter_set == 3
        stride = 0.5;%1;
    else
        stride = 0.5;
    end
    
    for iter_cls = 1:length(dir_cls)%1:length(dir_cls)
        
        cd(dir_save)
        cd(dir_set(iter_set).name)
        cd(dir_cls(iter_cls).name)
        dir_h5 = dir('2*');
        for iter_h5 = 3:3:length(dir_h5)
            
            iter_x = 0;
            iter_y = 0;

            cd(dir_save)
            cd(dir_set(iter_set).name)
            cd(dir_cls(iter_cls).name)
            data = h5read(dir_h5(iter_h5).name, '/ri');

            set(0, 'CurrentFigure', h_), hold off
            imagesc(data, [1.33 1.38]), axis image, colormap gray
            drawnow
            pause(0.1)
            

            mkdir([dir_fig, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            cd([dir_fig, '/',...
                dir_set(iter_set).name, '/',...
                dir_cls(iter_cls).name]);
            saveas(h_, strrep(dir_h5(iter_h5).name, '.h5','.fig'))
        end
    end
    
end

end