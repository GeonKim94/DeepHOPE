close all
clear
clc


% 230502_MIPH5_wider_allh_b002_in_lr0.001 (no _2) for allh
% 230502_MIPH5_wider_12h_b002_in_lr0.001 (no _2) for 12h
% 230502_MIPH5_wider_24h_b002_in_lr0.001_2 for 24h
save_fig =true;
save_png =true;
str_lineex = {'onRA'}%{'all', 'noGM', 'noH9', 'noJAX'};
str_hour = {'all'}% {'12', '24', 'all'};
cmap_line = [1 1 0; 1 0 1; 0 1 1];
for iter_lineex = 1:length(str_lineex)
    for iter_hour = 1:length(str_hour)% 3 to be done later
        if contains(str_lineex{iter_lineex}, 'no')
            dir_out = sprintf('/data02/gkim/stem_cell_jwshin/outs/230811+230502_MIPH5_wider_v2_%sh_%s_b002_in_lr0.001',...
                str_hour{iter_hour}, str_lineex{iter_lineex});
            dir_out_gen = [dir_out '_' strrep(str_lineex{iter_lineex},'no','test')];
        elseif contains(str_lineex{iter_lineex}, 'on')
            dir_out = sprintf('/data02/gkim/stem_cell_jwshin/outs/230811+230502_MIPH5_wider_v2_%sh_%s_b002_in_lr0.001',...
                str_hour{iter_hour}, str_lineex{iter_lineex});
            dir_out_gen = 'Nothing';
        else
            dir_out = sprintf('/data02/gkim/stem_cell_jwshin/outs/230811+230502_MIPH5_wider_v2_%sh_b002_in_lr0.001',...
                str_hour{iter_hour});%;, str_lineex{iter_lineex});
            dir_out_gen = 'Nothing';
        end
        dir_out_genp = [dir_out sprintf('_test%s', 'PD')];
        dir_out_genb = [dir_out sprintf('_test%s', 'BJ')];
        cd(dir_out);
        list_model = dir('*epoch*]');
        if length(list_model) == 0
            continue
        end
        epochs = [];
        accs_tr = [];
        accs_va = [];
        accs_te = [];
        for iter_model = 1:length(list_model)
            dir_model = list_model(iter_model).name;
            idxs_lbra = strfind(dir_model,'[');
            epochs = [epochs; str2num(dir_model(idxs_lbra(1)+1:idxs_lbra(1)+5))];
            accs_tr = [accs_tr; str2num(dir_model(idxs_lbra(2)+1:idxs_lbra(2)+5))];
            accs_va = [accs_va; str2num(dir_model(idxs_lbra(3)+1:idxs_lbra(3)+5))];
            accs_te = [accs_te; str2num(dir_model(idxs_lbra(4)+1:idxs_lbra(4)+5))];
        end
        
        metrics = accs_tr+accs_va;
        idx_bestmodel = min(find(metrics == max(metrics)));
        dir_bestmodel = list_model(idx_bestmodel).name;
            idxs_lbra = strfind(dir_bestmodel,'[');
        epoch_bestmodel_str = (dir_bestmodel(idxs_lbra(1)+1:idxs_lbra(1)+5));
        
        cd(dir_bestmodel)
        load('result_test.mat');
        
        targets = single(targets);
        test_wholesize = length(paths);
        scores_test = reshape(scores, [length(scores)/(test_wholesize), test_wholesize])';
        feats = cell2mat(feats)';
        feats_test = reshape(feats, [length(feats)/(test_wholesize), test_wholesize])';
        paths_test = paths';
        targets_test = single(targets');
        preds_test = -ones(size(targets_test));
        for iter_data = 1:length(targets_test)
            preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
        end
        preds_test = preds_test-1;
        
        lines_test = zeros(size(targets_test));
        for iter_data = 1:length(targets_test)
            fname = paths_test{iter_data};
            idx_slash = strfind(fname,'/');
            fname = fname(idx_slash(end)+1:end);
            if contains(fname, 'GM') || contains(fname, 'gm') || contains(fname, '230719.')  || contains(fname, '230720.') 
                lines_test(iter_data) = 1;
            elseif contains(fname, 'H9') || contains(fname, 'h9') || contains(fname, '230714.') 
                lines_test(iter_data) = 2;
            elseif contains(fname, 'JAX') || contains(fname, 'Jax') || contains(fname, '230713.') 
                lines_test(iter_data) = 3;
            else
                error('No line found')
            end
        end
        
        targets_test_h = zeros(size(targets_test));
        for iter_data = 1:length(targets_test)
            fname = paths_test{iter_data};
            idx_slash = strfind(fname,'/');
            fname = fname(idx_slash(end)+1:end);
            if contains(fname, '24h') || contains(fname, '24H')
                targets_test_h(iter_data) = -1;
            elseif contains(fname, '12h') || contains(fname, '12H')
                targets_test_h(iter_data) = 0;
            elseif contains(fname, 'untreated') || contains(fname, 'Untreated') || contains(fname, 'UNTREATED')
                targets_test_h(iter_data) = 1;
            else
                error('No hour found')
            end
        end
        
        %%
        h1 = figure(1);
        set(h1, 'Position', [0 200 1600 240])
        set(h1, 'Color', [1 1 1])
        
        subplot(4,1,1);
        imagesc(lines_test', [1 3])
        set(gca,'Colormap', cmap_line), axis off
        
        subplot(4,1,2);
        imagesc([targets_test-0.5*(targets_test == 0) targets_test_h]', [-1 1])
        set(gca,'Colormap', jet), axis off
        
        subplot(4,1,3);
        imagesc(scores_test', [-5 5])
        set(gca,'Colormap', parula), axis off
        
        subplot(4,1,4);
        imagesc((preds_test-0.5*(preds_test == 0))', [-1 1])
        set(gca,'Colormap', jet), axis off
        
        
        %%
        h2 = figure(2);
        set(h2, 'Position', [0 200 900 300])
        set(h2, 'Color', [1 1 1])
        cmap = jet;
        for iter_h = 1:3
            idxs_data = find(targets_test_h == (iter_h-2));
            subplot(1,3,iter_h)
            bar(1, sum(preds_test(idxs_data)==0), 'FaceColor', cmap(round(end/4),:))
            hold on
            bar(2, sum(preds_test(idxs_data)==1), 'FaceColor', cmap(end,:))
            hold off
            xticks([-10 10])
            if length(idxs_data) > 0 
                ylim([0 length(idxs_data)])
            end
        end
        
        %%
        h3 = figure(3);
        set(h3, 'Color', [1 1 1])
        set(h3, 'Position', [0 200 400 400])
        
        rocObj = rocmetrics(targets_test',...
            reshape(scores_test(:,2) - scores_test(:,1), size(targets')),1);
        plot(rocObj)
        
        
        % save('test_summary.mat', 'paths_test', 'targets_test', 'scores_test', 'preds_test', 'feats_test');
        
        %%
        if save_fig
            saveas(h1, [dir_out '_' 'figbest_profile.fig'])
            saveas(h2, [dir_out '_' 'figbest_inference.fig'])
            saveas(h3, [dir_out '_' 'figbest_ROC.fig'])
        end
        if save_png
            saveas(h1, [dir_out '_' 'figbest_profile.png'])
            saveas(h2, [dir_out '_' 'figbest_inference.png'])
            saveas(h3, [dir_out '_' 'figbest_ROC.png'])
        end
        
       
        %%
        
        if ~exist(dir_out_genp)
        else
        cd(dir_out_genp)
        list_model = dir(['epoch[' epoch_bestmodel_str '*]']);
        dir_bestmodel_genp = list_model(1).name;
        cd(dir_bestmodel_genp)
        
        load('result_test.mat');
        
        targets = single(targets);
        test_wholesize = length(paths);
        scores_test = reshape(scores, [length(scores)/(test_wholesize), test_wholesize])';
        feats = cell2mat(feats)';
        feats_test = reshape(feats, [length(feats)/(test_wholesize), test_wholesize])';
        paths_test = paths';
        targets_test = single(targets');
        preds_test = -ones(size(targets_test));
        for iter_data = 1:length(targets_test)
            preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
        end
        preds_test = preds_test-1;
        
        lines_test = zeros(size(targets_test));
        for iter_data = 1:length(targets_test)
            fname = paths_test{iter_data};
            idx_slash = strfind(fname,'/');
            fname = fname(idx_slash(end)+1:end);
            if contains(fname, '.A20.') || contains(fname, 'A20_')
                lines_test(iter_data) = 1;
            elseif contains(fname, '.A19.') || contains(fname, 'A19_')
                lines_test(iter_data) = 2;
            elseif contains(fname, '.A15.') || contains(fname, 'A15_')
                continue%lines_test(iter_data) = 3;
        
            elseif contains(fname, '.E5.') || contains(fname, 'E5_')
                continue%lines_test(iter_data) = 4;
            elseif contains(fname, '.E3.') || contains(fname, 'E3_')
                continue%lines_test(iter_data) = 5;
        
            elseif contains(fname, '.A4.') || contains(fname, 'A4_')
                continue%lines_test(iter_data) = 6;
            elseif contains(fname, '.A2.') || contains(fname, 'A2_')
                lines_test(iter_data) = 3;%7;
            elseif contains(fname, '.A12.') || contains(fname, 'A12_')
                lines_test(iter_data) = 4;%8;
            else
                error('No line found')
            end
        end
        
        % targets_test_h = zeros(size(targets_test));
        % for iter_data = 1:length(targets_test)
        %     fname = paths_test{iter_data};
        %     idx_slash = strfind(fname,'/');
        %     fname = fname(idx_slash(end)+1:end);
        %     if contains(fname, '24h') || contains(fname, '24H')
        %         targets_test_h(iter_data) = -1;
        %     elseif contains(fname, '12h') || contains(fname, '12H')
        %         targets_test_h(iter_data) = 0;
        %     elseif contains(fname, 'untreated') || contains(fname, 'Untreated') || contains(fname, 'UNTREATED')
        %         targets_test_h(iter_data) = 1;
        %     else
        %         error('No hour found')
        %     end
        % end
        idx_del = find(lines_test == 0);
        lines_test(idx_del) = [];
        feats_test(idx_del,:) = [];
        paths_test(idx_del,:) = [];
        targets_test(idx_del,:) = [];
        scores_test(idx_del,:) = [];
        preds_test(idx_del,:) = [];
        
        sum(targets_test-preds_test == 0)/length(targets_test) % accuracy
        %%
        h7 = figure(7);
        set(h7, 'Position', [0 200 1600 240])
        set(h7, 'Color', [1 1 1])
        
        subplot(4,1,1);
        imagesc(lines_test', [1 12])
        set(gca,'Colormap', colorcube), axis off
        
        subplot(4,1,2);
        imagesc([targets_test-0.5*(targets_test == 0)]', [-1 1])
        set(gca,'Colormap', jet), axis off
        
        subplot(4,1,3);
        imagesc(scores_test', [-5 5])
        set(gca,'Colormap', parula), axis off
        
        subplot(4,1,4);
        imagesc((preds_test-0.5*(preds_test == 0))', [-1 1])
        set(gca,'Colormap', jet), axis off
        
        
        %%
        h8 = figure(8);
        set(h8, 'Position', [0 200 900 300])
        set(h8, 'Color', [1 1 1])
        
        for iter_h = 1:length([unique(lines_test)]')
            line_plot = unique(lines_test);
            line_plot = line_plot(iter_h);
            idxs_data = find(lines_test == line_plot);
            subplot(1,length([unique(lines_test)]'),iter_h)
            bar(1, sum(preds_test(idxs_data)==0), 'FaceColor', cmap(round(end/4),:))
            hold on
            bar(2, sum(preds_test(idxs_data)==1), 'FaceColor', cmap(end,:))
            hold off
            xticks([-10 10])
            if length(idxs_data) > 0 
                ylim([0 length(idxs_data)])
            end
        end
        
        %%
        h9 = figure(9);
        set(h9, 'Color', [1 1 1])
        set(h9, 'Position', [0 200 400 400])
        rocObj = rocmetrics(targets_test',...
            reshape(scores_test(:,2) - scores_test(:,1), size(targets_test')),1);
        plot(rocObj)
        
        % save('test_summary.mat', 'paths_test', 'targets_test', 'scores_test', 'preds_test', 'feats_test');
        
        %%
        if save_fig
            saveas(h7, [dir_out_genp '_' 'figbest_profile.fig'])
            saveas(h8, [dir_out_genp '_' 'figbest_inference.fig'])
            saveas(h9, [dir_out_genp '_' 'figbest_ROC.fig'])
        end
        if save_png
            saveas(h7, [dir_out_genp '_' 'figbest_profile.png'])
            saveas(h8, [dir_out_genp '_' 'figbest_inference.png'])
            saveas(h9, [dir_out_genp '_' 'figbest_ROC.png'])
        end

        end

        if ~exist(dir_out_genb)
        else
        cd(dir_out_genb)
        list_model = dir(['epoch[' epoch_bestmodel_str '*]']);
        dir_bestmodel_genb = list_model(1).name;
        cd(dir_bestmodel_genb)
        
        load('result_test.mat');
        
        targets = single(targets);
        test_wholesize = length(paths);
        scores_test = reshape(scores, [length(scores)/(test_wholesize), test_wholesize])';
        feats = cell2mat(feats)';
        feats_test = reshape(feats, [length(feats)/(test_wholesize), test_wholesize])';
        paths_test = paths';
        targets_test = single(targets');
        preds_test = -ones(size(targets_test));
        for iter_data = 1:length(targets_test)
            preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
        end
        preds_test = preds_test-1;
        
        lines_test = zeros(size(targets_test));
        for iter_data = 1:length(targets_test)
            fname = paths_test{iter_data};
            idx_slash = strfind(fname,'/');
            fname = fname(idx_slash(end)+1:end);

            if contains(fname, '.B1.') || contains(fname, 'B1_')
                lines_test(iter_data) = 1;
            elseif contains(fname, '.B7.') || contains(fname, 'B7_')
                lines_test(iter_data) = 2;
            elseif contains(fname, '.B12.') || contains(fname, 'B12_')
                lines_test(iter_data) = 3;
            elseif contains(fname, '.B13.') || contains(fname, 'B13_')
                lines_test(iter_data) = 4;
            elseif contains(fname, '.B17.') || contains(fname, 'B17_')
                lines_test(iter_data) = 5;
            elseif contains(fname, '.B18.') || contains(fname, 'B18_')
                lines_test(iter_data) = 6;
            elseif contains(fname, '.B21.') || contains(fname, 'B21_')
                lines_test(iter_data) = 7;
            elseif contains(fname, '.B23.') || contains(fname, 'B23_')
                lines_test(iter_data) = 8;
            else
                error('No line found')
            end
        end
        
        
        % targets_test_h = zeros(size(targets_test));
        % for iter_data = 1:length(targets_test)
        %     fname = paths_test{iter_data};
        %     idx_slash = strfind(fname,'/');
        %     fname = fname(idx_slash(end)+1:end);
        %     if contains(fname, '24h') || contains(fname, '24H')
        %         targets_test_h(iter_data) = -1;
        %     elseif contains(fname, '12h') || contains(fname, '12H')
        %         targets_test_h(iter_data) = 0;
        %     elseif contains(fname, 'untreated') || contains(fname, 'Untreated') || contains(fname, 'UNTREATED')
        %         targets_test_h(iter_data) = 1;
        %     else
        %         error('No hour found')
        %     end
        % end
        idx_del = find(lines_test == 0);
        lines_test(idx_del) = [];
        feats_test(idx_del,:) = [];
        paths_test(idx_del,:) = [];
        targets_test(idx_del,:) = [];
        scores_test(idx_del,:) = [];
        preds_test(idx_del,:) = [];
        
        sum(targets_test-preds_test == 0)/length(targets_test) % accuracy
        %%
        h17 = figure(17);
        set(h17, 'Position', [0 200 1600 240])
        set(h17, 'Color', [1 1 1])
        
        subplot(4,1,1);
        imagesc(lines_test', [1 12])
        set(gca,'Colormap', colorcube), axis off
        
        subplot(4,1,2);
        imagesc([targets_test-0.5*(targets_test == 0)]', [-1 1])
        set(gca,'Colormap', jet), axis off
        
        subplot(4,1,3);
        imagesc(scores_test', [-5 5])
        set(gca,'Colormap', parula), axis off
        
        subplot(4,1,4);
        imagesc((preds_test-0.5*(preds_test == 0))', [-1 1])
        set(gca,'Colormap', jet), axis off
        
        
        %%
        h18 = figure(18);
        set(h18, 'Position', [0 200 900 300])
        set(h18, 'Color', [1 1 1])
        
        for iter_h = 1:length([unique(lines_test)]')
            line_plot = unique(lines_test);
            line_plot = line_plot(iter_h);
            idxs_data = find(lines_test == line_plot);
            subplot(1,length([unique(lines_test)]'),iter_h)
            bar(1, sum(preds_test(idxs_data)==0), 'FaceColor', cmap(round(end/4),:))
            hold on
            bar(2, sum(preds_test(idxs_data)==1), 'FaceColor', cmap(end,:))
            hold off
            xticks([-10 10])
            if length(idxs_data) > 0 
                ylim([0 length(idxs_data)])
            end
        end
        
        %%
        h19 = figure(19);
        set(h19, 'Color', [1 1 1])
        set(h19, 'Position', [0 200 400 400])
        rocObj = rocmetrics(targets_test',...
            reshape(scores_test(:,2) - scores_test(:,1), size(targets_test')),1);
        plot(rocObj)
        
        % save('test_summary.mat', 'paths_test', 'targets_test', 'scores_test', 'preds_test', 'feats_test');
        
        %%
        if save_fig
            saveas(h17, [dir_out_genb '_' 'figbest_profile.fig'])
            saveas(h18, [dir_out_genb '_' 'figbest_inference.fig'])
            saveas(h19, [dir_out_genb '_' 'figbest_ROC.fig'])
        end
        if save_png
            saveas(h17, [dir_out_genb '_' 'figbest_profile.png'])
            saveas(h18, [dir_out_genb '_' 'figbest_inference.png'])
            saveas(h19, [dir_out_genb '_' 'figbest_ROC.png'])
        end

        end

         %%

%         if ~exist(dir_out_gen)
%             continue
%         end
%         cd(dir_out_gen)
%         list_model = dir(['epoch[' epoch_bestmodel_str '*]']);
%         dir_bestmodel_gen = list_model(1).name;
%         cd(dir_bestmodel_gen)
%         
%         load('result_test.mat');
%         
%         targets = single(targets);
%         test_wholesize = length(paths);
%         scores_test = reshape(scores, [length(scores)/(test_wholesize), test_wholesize])';
%         feats = cell2mat(feats)';
%         feats_test = reshape(feats, [length(feats)/(test_wholesize), test_wholesize])';
%         paths_test = paths';
%         targets_test = single(targets');
%         preds_test = -ones(size(targets_test));
%         for iter_data = 1:length(targets_test)
%             preds_test(iter_data) = find(scores_test(iter_data,:) == max(scores_test(iter_data,:)));
%         end
%         preds_test = preds_test-1;
%         
%         lines_test = zeros(size(targets_test));
%         for iter_data = 1:length(targets_test)
%             fname = paths_test{iter_data};
%             idx_slash = strfind(fname,'/');
%             fname = fname(idx_slash(end)+1:end);
%             if contains(fname, 'GM') || contains(fname, 'gm') || contains(fname, '230719.')  || contains(fname, '230720.') 
%                 lines_test(iter_data) = 1;
%             elseif contains(fname, 'H9') || contains(fname, 'h9') || contains(fname, '230714.') 
%                 lines_test(iter_data) = 2;
%             elseif contains(fname, 'JAX') || contains(fname, 'Jax') || contains(fname, '230713.') 
%                 lines_test(iter_data) = 3;
%             else
%                 error('No line found')
%             end
%         end
%         
%         targets_test_h = 2*ones(size(targets_test));
%         for iter_data = 1:length(targets_test)
%             fname = paths_test{iter_data};
%             idx_slash = strfind(fname,'/');
%             fname = fname(idx_slash(end)+1:end);
%             if contains(fname, '24h') || contains(fname, '24H')
%                 if strcmp(str_hour{iter_hour},'12')
%                     continue
%                 end
%                 targets_test_h(iter_data) = -1;
%             elseif contains(fname, '12h') || contains(fname, '12H')
%                 if strcmp(str_hour{iter_hour},'24')
%                     continue
%                 end
%                 targets_test_h(iter_data) = 0;
%             elseif contains(fname, 'untreated') || contains(fname, 'Untreated') || contains(fname, 'UNTREATED')
%                 targets_test_h(iter_data) = 1;
%             else
%                 error('No hour found')
%             end
%         end
%         idx_del = find(targets_test_h == 2);
%         lines_test(idx_del) = [];
%         feats_test(idx_del,:) = [];
%         paths_test(idx_del,:) = [];
%         targets_test(idx_del,:) = [];
%         targets_test_h(idx_del,:) = [];
%         scores_test(idx_del,:) = [];
%         preds_test(idx_del,:) = [];
%         
%         %%
%         h4 = figure(4);
%         set(h4, 'Position', [0 200 1600 240])
%         set(h4, 'Color', [1 1 1])
%         
%         subplot(4,1,1);
%         imagesc(lines_test', [1 3])
%         set(gca,'Colormap', cmap_line), axis off
%         
%         subplot(4,1,2);
%         imagesc([targets_test-0.5*(targets_test == 0) targets_test_h]', [-1 1])
%         set(gca,'Colormap', jet), axis off
%         
%         subplot(4,1,3);
%         imagesc(scores_test', [-5 5])
%         set(gca,'Colormap', parula), axis off
%         
%         subplot(4,1,4);
%         imagesc((preds_test-0.5*(preds_test == 0))', [-1 1])
%         set(gca,'Colormap', jet), axis off        
%         %%
%         h5 = figure(5);
%         set(h5, 'Position', [0 200 900 300])
%         set(h5, 'Color', [1 1 1])
%         
%         for iter_h = 1:3
%             idxs_data = find(targets_test_h == (iter_h-2));
%             subplot(1,3,iter_h)
%             bar(1, sum(preds_test(idxs_data)==0), 'FaceColor', cmap(round(end/4),:))
%             hold on
%             bar(2, sum(preds_test(idxs_data)==1), 'FaceColor', cmap(end,:))
%             hold off
%             xticks([-10 10])
%             if length(idxs_data) > 0 
%                 ylim([0 length(idxs_data)])
%             end
%         end
%         
%         %%
%         h6 = figure(6);
%         set(h6, 'Color', [1 1 1])
%         set(h6, 'Position', [0 200 400 400])
%         rocObj = rocmetrics(targets_test',...
%             reshape(scores_test(:,2) - scores_test(:,1), size(targets_test')),1);
%         plot(rocObj)
% 
%         % save('test_summary.mat', 'paths_test', 'targets_test', 'scores_test', 'preds_test', 'feats_test');
%         
%         %%
%         if save_fig
%             saveas(h4, [dir_out_gen '_' 'figbest_profile.fig'])
%             saveas(h5, [dir_out_gen '_' 'figbest_inference.fig'])
%             saveas(h6, [dir_out_gen '_' 'figbest_ROC.fig'])
%         end
%         if save_png
%             saveas(h4, [dir_out_gen '_' 'figbest_profile.png'])
%             saveas(h5, [dir_out_gen '_' 'figbest_inference.png'])
%             saveas(h6, [dir_out_gen '_' 'figbest_ROC.png'])
%         end
    end
end