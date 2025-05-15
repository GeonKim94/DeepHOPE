clear
close all
clc
addpath(fileparts(mfilename('fullpath')));

patch_size = 512;
crop_size = 512;

dir_infer = '/gkim/demo/infer_patch/PD-1'; % change to you inference path
dir_data = '/gkim/demo/data_patch/PD-1'; % toggle to show patch grid on HT MIP image or not
dir_coor = '/gkim/demo/coor_patch/PD-1'; % toggle to show patch grid on HT MIP image or not
dir_data_wide = '/gkim/demo/data_wide/PD-1'; % toggle to show patch grid on HT MIP image or not

cd(dir_infer)
list_mat = dir('*.mat'); 

see_patch_grid = true; % toggle to show patch grid on HT MIP image or not
see_patch_center = true; % toggle to show patch center on HT MIP image or not

h_ = figure(1);
h_.Position = [0 0 1920 1080];
h_.Color = [1 1 1];

list_stitch = {list_mat.name};
for iter_mat = 1:length(list_stitch)
    fname_wide = list_stitch{iter_mat};
    fname_wide = fname_wide(1:max(strfind(fname_wide,'_'))-1);
    list_stitch{iter_mat} = fname_wide;
end

list_stitch = unique(list_stitch);

for iter_stitch = 1:length(list_stitch)
    fname_wide = list_stitch{iter_stitch};
    
    list_patch = findFilesWithPattern(dir_infer, fname_wide);
    
    scores_patch = [];
    for iter_patch = 1:length(list_patch)
        path_patch = list_patch{iter_patch};
        
        load(path_patch);
        scores_patch = [scores_patch; score];
    
    end
    score_wide = mean(scores_patch,1);
    probs_patch = [];

    path_wide = findFirstFileWithPattern(dir_data_wide,fname_wide);
    ri = h5read(path_wide,'/ri');
    list_coor = findFilesWithPattern(dir_coor,fname_wide);
    
    dxs = [];
    dys = [];
    probs = [];
    grid_n = zeros(size(ri,[1,2]));
    grid_prob = zeros(size(ri,[1,2]));
    eps = 1/255;
    for iter_p = 1:length(list_coor)
        load(list_coor{iter_p}, 'dx','dy');
        dxs = [dxs; dx];
        dys = [dys; dy];
       
        
        prob = exp(scores_patch(iter_p,1))/(exp(scores_patch(iter_p,1))+exp(scores_patch(iter_p,2)));
        probs = [probs;prob];
        grid_prob(dx+1+floor((patch_size-crop_size)/2):dx+floor(patch_size/2+crop_size/2),...
            dy+1+floor((patch_size-crop_size)/2):dy+floor(patch_size/2+crop_size/2)) = ...
            grid_prob(dx+1+floor((patch_size-crop_size)/2):dx+floor(patch_size/2+crop_size/2),...
            dy+1+floor((patch_size-crop_size)/2):dy+floor(patch_size/2+crop_size/2)) + ...
            prob+eps;
    
        grid_n(dx+1+floor((patch_size-crop_size)/2):dx+floor(patch_size/2+crop_size/2),...
            dy+1+floor((patch_size-crop_size)/2):dy+floor(patch_size/2+crop_size/2)) = ...
            grid_n(dx+1+floor((patch_size-crop_size)/2):dx+floor(patch_size/2+crop_size/2),...
            dy+1+floor((patch_size-crop_size)/2):dy+floor(patch_size/2+crop_size/2)) + ...
            1;
    
        probs_patch = [probs_patch; prob];
    end
    
    grid_prob = grid_prob./grid_n;
    
    mask_grid = grid_prob >= eps;


    cmap = turbo;
    cmap(1,:) = [1 1 1];
    set(0, 'CurrentFigure', h_)
    subplot(1,2,1),imagesc(max(ri,[],3), [13370 13770]), axis image
    ax = gca;
    ax.Colormap = gray;
    title('RI MIP (live)')
    hold on

    if see_patch_center
        for iter_patch = 1:length(dxs)
            dx = dxs(iter_patch);dy = dys(iter_patch);
            plot([dy+1 dy+1],[dx+1 dx+crop_size], 'w-', 'LineWidth',1);
            plot([dy+crop_size dy+crop_size],[dx+1 dx+crop_size], 'w-', 'LineWidth',1);
            plot([dy+1 dy+crop_size],[dx+1 dx+1], 'w-', 'LineWidth',1);
            plot([dy+crop_size dy+1],[dx+crop_size dx+crop_size], 'w-', 'LineWidth',1);
        end
    end

    if see_patch_center
        for iter_patch = 1:length(dxs)
            dx = dxs(iter_patch);dy = dys(iter_patch);
            plot([dy+ceil(1/2+crop_size/2)],[dx+ceil(1/2+crop_size/2)],'r*');
        end
        hold off
    end
    
    subplot(1,2,2), imagesc(grid_prob,[0,1]),axis image
    ax = gca;
    ax.Colormap = cmap;
    title('P(undifferentiated)')
    drawnow


end