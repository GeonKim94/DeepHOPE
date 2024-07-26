cd '/data02/gkim/stem_cell_jwshin/outs/230811_MIPH5_wider_12h_b002_in_lr0.001';
%%'/data02/gkim/stem_cell_jwshin/outs/230811_MIPH5_wider_12h_noGM_b002_in_lr0.001';
%'/data02/gkim/stem_cell_jwshin/outs/230502_MIPH5_wider_12h_b002_in_lr0.001/'
list_model = dir('*.tar');

accs_val = [];
for iter_model = 1:length(list_model)

    fname_model = list_model(iter_model).name;
    idx_str1 = strfind(fname_model,'acc[');
    idx_str2 = strfind(fname_model,']_test');
    accs_val = [accs_val;str2num(fname_model(idx_str1+4:idx_str2-1))];
end
idxs_best = find(accs_val > 0.95)%== max(accs_val));

strjoin({list_model(idxs_best).name},',')