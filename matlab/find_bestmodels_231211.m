% cd /data02/gkim/IVF_chlee/outs/231110_mulsec17_time0003_b016_in_lr0.0010_lowlow_fishdeep3
list_model = dir('*.tar');

accs_val = [];
accs_train = [];
for iter_model = 1:length(list_model)

    fname_model = list_model(iter_model).name;
    idx_str1 = strfind(fname_model,'_va[');
    idx_str2 = strfind(fname_model,']_te[');
    accs_val = [accs_val;str2num(fname_model(idx_str1+4:idx_str2-1))];
    idx_str1 = strfind(fname_model,'_tr[');
    idx_str2 = strfind(fname_model,']_va[');
    accs_train = [accs_train;str2num(fname_model(idx_str1+4:idx_str2-1))];
end


idxs_best_v = find(accs_val >= max(accs_val - 0.025));%== max(accs_val));
idxs_best_b = find(accs_val+accs_train >= max(accs_val+accs_train - 0.025));

idxs_best = unique([idxs_best_v;idxs_best_b]);

strjoin({list_model(idxs_best).name},',')