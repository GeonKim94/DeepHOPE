


%%


dir_out = 'G:\stem_cell_jwshin\outs_present2';

cd(dir_out)
list_arch = dir('2*');
list_arch = list_arch([list_arch.isdir]);

for iter_arch = 1:length(list_arch)
    dir_arch = list_arch(iter_arch).name;
    cd(dir_out)
    cd(dir_arch)
    
    list_epoch = dir('epoch*');
    list_epoch = list_epoch([list_epoch.isdir]);
    for iter_epoch = 1:length(list_epoch)

        dir_epoch = list_epoch(iter_epoch).name;
        cd(dir_out)
        cd(dir_arch)
        cd(dir_epoch)
        copyfile('select_prediction_normalized.png',[dir_out '\' dir_arch '_' dir_epoch(1:12) '.png'])
    end
end

