%% onRA all
list_source = {'/data02/gkim/stem_cell_jwshin/data/230811_3DH5_wider_v3',...
    '/data02/gkim/stem_cell_jwshin/data/230502_3DH5_wider_v3'};

dir_ref = '/data02/gkim/stem_cell_jwshin/data/230811+230502_MIPH5_wider_v2_allh_onRA';

dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_3DH5_wider_v3_allh_onRA';

list_set = {'train', 'test', 'val'};

for iter_source = 1:length(list_source)

    dir_source = list_source{iter_source};

    cd(dir_source)
    cd('00_train')
    list_cls = dir('*_*');
    
    for iter_cls = 1:length(list_cls)

        cd(dir_source)
        cd('00_train')
        cd(list_cls(iter_cls).name)

        if contains(list_cls(iter_cls).name, '_untreated')
            str_cls = '01_high';
        else
            str_cls = '00_low';
        end

        list_h5 = dir('*.h5');

        for iter_h5 = 1:length(list_h5)
            
            fname_h5 = list_h5(iter_h5).name;

            for iter_set = 1:length(list_set)
                
                if isfile([dir_ref '/' list_set{iter_set} '/' str_cls '/'...
                        fname_h5])
                    
                    mkdir([dir_data '/' list_set{iter_set} '/' str_cls])
                    copyfile(fname_h5,...
                        [dir_data '/' list_set{iter_set} '/' str_cls]);

                    break

                end

            end


        end

    end

end

%% onRA all noJAX
list_source = {'/data02/gkim/stem_cell_jwshin/data/230811_3DH5_wider_v3',...
    '/data02/gkim/stem_cell_jwshin/data/230502_3DH5_wider_v3'};

dir_ref = '/data02/gkim/stem_cell_jwshin/data/230811+230502_MIPH5_wider_v2_allh_onRA_noJAX';

dir_data = '/data02/gkim/stem_cell_jwshin/data/230811+230502_3DH5_wider_v3_allh_onRA_noJAX';

list_set = {'train', 'test', 'val'};

for iter_source = 1:length(list_source)

    dir_source = list_source{iter_source};

    cd(dir_source)
    cd('00_train')
    list_cls = dir('*_*');
    
    for iter_cls = 1:length(list_cls)

        cd(dir_source)
        cd('00_train')
        cd(list_cls(iter_cls).name)

        if contains(list_cls(iter_cls).name, '_untreated')
            str_cls = '01_high';
        else
            str_cls = '00_low';
        end

        list_h5 = dir('*.h5');

        for iter_h5 = 1:length(list_h5)
            
            fname_h5 = list_h5(iter_h5).name;

            for iter_set = 1:length(list_set)
                
                if isfile([dir_ref '/' list_set{iter_set} '/' str_cls '/'...
                        fname_h5])
                    
                    mkdir([dir_data '/' list_set{iter_set} '/' str_cls])
                    copyfile(fname_h5,...
                        [dir_data '/' list_set{iter_set} '/' str_cls]);

                    break

                end

            end


        end

    end

end