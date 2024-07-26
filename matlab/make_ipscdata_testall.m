%%
close all
clear
clc

addpath('/data01/gkim/Matlab_subcodes/gk/')
% SEC1 <=> MIP
dir_data = '/data02/gkim/stem_cell_jwshin/data/230117iPSC_SEC1H5_wider_v3_testall';
dir_source = '/data02/gkim/stem_cell_jwshin/data/230306_SEC1H5_wider_v3';
dir_source = '/data02/gkim/stem_cell_jwshin/data/230920_SEC1H5_wider_v3';
dir_source = '/data02/gkim/stem_cell_jwshin/data/231117_SEC1H5_wider_v3';
path_anyinput = ['/data02/gkim/stem_cell_jwshin/data/230502_SEC1H5_wider_v3/00_train/GM25256_12hour_treat/' ...
    '230426.163526.GM25256_12hour_treat.001.Group1.A1.S001.h5'];

cd(dir_source)
cd('00_train')

list_cls = dir('*_*');

for iter_cls = 1:length(list_cls)
    
    cd(dir_source)
    cd('00_train')
    cd(list_cls(iter_cls).name)

    list_h5 = dir('*.h5');

    for iter_h5 = 1:length(list_h5)
    
            str_set = '/test/';
%             str_set = '/train/';
%             str_set = '/val/';

            str_cls = '/01_high/';
%             str_cls = '/00_low/';


        mkdir([dir_data str_set str_cls])
        
        %copyfile if u got space
        movefile([dir_source '/00_train/' ...
            list_cls(iter_cls).name '/'...
            list_h5(iter_h5).name],...
            [dir_data str_set str_cls])
          
    end

end

%%
str_set = '/test/';
% str_set = '/train/';
% str_set = '/val/';

% str_cls = '/01_high/';
str_cls = '/00_low/';

mkdir([dir_data str_set str_cls])
copyfile(path_anyinput,...
    [dir_data str_set str_cls]);


% str_set = '/test/';
str_set = '/train/';
% str_set = '/val/';

% str_cls = '/01_high/';
str_cls = '/00_low/';

mkdir([dir_data str_set str_cls])
copyfile(path_anyinput,...
    [dir_data str_set str_cls]);


% str_set = '/test/';
str_set = '/train/';
% str_set = '/val/';

str_cls = '/01_high/';
% str_cls = '/00_low/';

mkdir([dir_data str_set str_cls])
copyfile(path_anyinput,...
    [dir_data str_set str_cls]);


% str_set = '/test/';
% str_set = '/train/';
str_set = '/val/';

% str_cls = '/01_high/';
str_cls = '/00_low/';

mkdir([dir_data str_set str_cls])
copyfile(path_anyinput,...
    [dir_data str_set str_cls]);


% str_set = '/test/';
% str_set = '/train/';
str_set = '/val/';

str_cls = '/01_high/';
% str_cls = '/00_low/';

mkdir([dir_data str_set str_cls])
copyfile(path_anyinput,...
    [dir_data str_set str_cls]);

