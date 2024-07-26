dir_raw = 'D:\HTXTest\yoonlab\20231211';
dir_mip= 'D:\HTXTest\yoonlab\20231211_miptiff';

cd(dir_raw)
list_cls = dir('GM*');

for iter_cls = 1:length(list_cls)
    cd(dir_raw)
    dir_cls = list_cls(iter_cls).name;
    cd(dir_cls)
    list_data = dir('2*');

    for iter_data = 1:length(list_data)
        cd(dir_raw)
        cd(dir_cls)
        dir_data = list_data(iter_data).name;
        cd(dir_data)

        list_tcf = dir('*.TCF');

        if length(dir_tcf) == 0 
            continue
        end
        fname = list_tcf(1).name;

        info_temp = h5info(fname, '/Info/Device');
        n_m = uint16(info_temp.Attributes(3).Value*10000);
        tomo_stitch = ReadLDMTCFHT(fname, hour);
        tomo_stitch = single(tomo_stitch*10000);
        tomo_stitch = permute(tomo_stitch, [2 1 3]);
        tomo_stitch(tomo_stitch == 0) = n_m; 

        mip_stitch = single(max(tomo_stitch,[],3)); 
        
        mkdir(dir_mip)
        cd(dir_mip)
        mkdir(dir_cls)
        cd(dir_cls)

        desired_min = 13370;
        desired_max = 13900;
        
        % Rescale the image values to maximize visibility between the desired minimum and maximum values
        mip_stitch = (mip_stitch-desired_min)/(desired_max-desired_min);
        mip_stitch(mip_stitch>1) = 1;
        mip_stitch(mip_stitch<0) = 0;
        mip_stitch = uint16(mip_stitch*65535);

        imwrite(mip_stitch, [dir_data, '.tiff']);


    end


end