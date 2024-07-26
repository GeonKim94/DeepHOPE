list_mat = dir('*.mat');

for iter_mat = 1:length(list_mat)
    fname = list_mat(iter_mat).name;
    idx_colony = fname(5:7);
    idx_colony_new = ['1' idx_colony(2:end)];
    fname_new = fname;
    fname_new = replace(fname_new,idx_colony,idx_colony_new);

    movefile(fname,fname_new);

end