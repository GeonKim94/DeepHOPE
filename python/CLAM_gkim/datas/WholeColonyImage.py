import math
import os
import time
import xml.etree.ElementTree as ET
from xml.dom import minidom
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
#import openslide
from PIL import Image
import pdb
import h5py
import math
# from wsi_core.wsi_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, coord_generator, save_hdf5, sample_indices, screen_coords, isBlackPatch, isWhitePatch, to_percentiles

import itertools
from utils.utils_classes import isInContourV1, isInContourV2, isInContourV3_Easy, isInContourV3_Hard, Contour_Checking_fn
from utils.file_utils import load_pkl, save_pkl
from utils.colony_utils import savePatchIter_bag_hdf5, initialize_hdf5_bag, isBlackPatch, isWhitePatch, save_hdf5, to_percentiles, screen_coords
from scipy import ndimage
from skimage import morphology
from datas import colony

resx = 0.155432865023613
resy = 0.155432865023613
resz = 0.949573814868927
#Benchmarking WholeSlideImage.py of CLAM

def create_binary_ellipse(array_size, axis_lengths):
    # Create a grid of coordinates
    x = np.linspace(-1, 1, array_size[1])
    y = np.linspace(-1, 1, array_size[0])
    xx, yy = np.meshgrid(x, y)

    # Compute the equation of the ellipse
    ellipse = ((xx / axis_lengths[1])**2 + (yy / axis_lengths[0])**2) <= 1

    return ellipse.astype(int)

def imfill_holes_mask(mask): # THIS DOESN'T WORK! CHAT GPT SUCKS
    # Find the external contours
    contours, _ = cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a new mask for drawing external contours
    filled_mask = np.zeros_like(mask)

    # Draw external contours on the new mask
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)

    # Invert the new mask
    filled_mask = cv2.bitwise_not(filled_mask)

    # Fill the contours in the inverted mask
    filled_mask = cv2.bitwise_or(filled_mask, filled_mask, mask=mask)

    return filled_mask


def fill_mask_holes(mask): # has to be a binary mask
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask for filling
    filled_mask = np.zeros_like(mask)
    
    # Draw contours onto the blank mask
    cv2.drawContours(filled_mask, contours, -1, (1), thickness=cv2.FILLED)
    
    return filled_mask


def imfill_contours(shape_img, contours, hierarchy):
    # Create a black mask
    mask = np.zeros(shape_img, dtype=np.uint8)

    # Draw contours on the mask
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Find the outer contours (parent == -1)
    outer_contours = [i for i, h in enumerate(hierarchy[0]) if h[3] == -1]

    # Fill the holes for each outer contour
    for outer_contour_idx in outer_contours:
        cv2.drawContours(mask, contours, outer_contour_idx, 255, thickness=cv2.FILLED)

    # Find new contours and hierarchy after filling holes
    filled_contours, filled_hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    return filled_contours, filled_hierarchy

class WholeColonyImage(object): #
    def __init__(self,path):
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.colony = colony.colony(path)
        self.level_downsamples = self._assertLevelDownsamples()
        self.level_dim = self.colony.level_dimensions
    
        self.contours_colony = None
        self.contours_tumor = None
        self.hdf5_file = None
    
    def getColony(self):
        return self.colony

    def _assertLevelDownsamples(self): #useless for uniform sampling
        level_downsamples = []
        dim_0 = self.colony.level_dimensions[0]
        for downsample, dim in zip(self.colony.level_downsamples, self.colony.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))
        return level_downsamples    

    def initSegmentation(self, mask_file):
        # load segmentation results from pickle file
        asset_dict = load_pkl(mask_file)
        self.holes_colony = asset_dict['holes']
        self.contours_colony = asset_dict['colony']

    def saveSegmentation(self, mask_file):
        # save segmentation results using pickle
        asset_dict = {'holes': self.holes_colony, 'colony': self.contours_colony}
        save_pkl(mask_file, asset_dict)

    # def segmentTissue(self, seg_level=0, sthresh=20, sthresh_up = 255, close = 0, use_otsu=False, 
    #                         filter_params={'a_t':100}, ref_patch_size=512, exclude_ids=[], keep_ids=[]):

    def segmentColony_fromcoords(self, coords, patch_size = 384): # only for visualization of
        #pdb.set_trace()
        mask_new = np.zeros(self.colony.mask.shape).astype('int8')
        for idx, coord_temp in enumerate(coords):
            mask_new[coord_temp[0]:coord_temp[0]+patch_size, coord_temp[1]:coord_temp[1]+patch_size] = 1
        self.colony.mask = mask_new


    def segmentColony(self, seg_level = 0, sthresh=13420, sthresh_up = 15000, mthresh=1, close = round(2/resx), use_otsu=False, 
                        filter_params={'a_t':100}, exclude_ids=[], keep_ids=[]):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        
        def _filter_contours(contours, hierarchy, filter_params): # unused for SC colony HTs
            """
                Filter contours by: area.
                    # filter_params
                    a_t: tissue/colony area threashold
                    a_h: hole area threshold
                    max_n_holes: hole number threshold
            """
            filtered = []

            # find indices of foreground contours (parent == -1)
            hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
            all_holes = []
            
            # loop through foreground contour indices
            for cont_idx in hierarchy_1:
                # actual contour
                cont = contours[cont_idx]
                # indices of holes contained in this contour (children of parent contour)
                holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
                # take contour area (includes holes)
                a = cv2.contourArea(cont)
                # calculate the contour area of each hole
                hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
                # actual area of foreground contour region
                a = a - np.array(hole_areas).sum()
                if a == 0: continue
                if tuple((filter_params['a_t'],)) < tuple((a,)): 
                    filtered.append(cont_idx)
                    all_holes.append(holes)
            foreground_contours = [contours[cont_idx] for cont_idx in filtered]
            
            hole_contours = []

            for hole_ids in all_holes:
                unfiltered_holes = [contours[idx] for idx in hole_ids ]
                unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
                # take max_n_holes largest holes by area
                unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
                filtered_holes = []
                
                # filter these holes
                for hole in unfilered_holes:
                    if cv2.contourArea(hole) > filter_params['a_h']:
                        filtered_holes.append(hole)

                hole_contours.append(filtered_holes)

            return foreground_contours, hole_contours
        
        img = np.array(self.colony.read_region((0,0),0,self.colony.dimension))

        if len(img.shape) > 2:
            img = np.max(img,axis = 2)

        # when read_region didn't have swapaxes included
        #img = np.swapaxes(img,0,2)
        #img = np.max(img, axis = 0) 
        
        if mthresh > 1.:
            img = cv2.medianBlur(img, mthresh)  # Apply median blurring

        # Thresholding
        if use_otsu:
            _, img_thres = cv2.threshold(img, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_thres = cv2.threshold(img, sthresh, sthresh_up, cv2.THRESH_BINARY)
        img_thres = np.uint8(img_thres)


        #morphological closing

        if close > 0:
            kernel = np.uint8(create_binary_ellipse((close,close), (close, close)))
            #img_thres = cv2.morphologyEx(img_thres, cv2.MORPH_CLOSE, kernel)
            img_thres = cv2.dilate(img_thres, kernel, iterations = 1)

        # Filling (not in the original CLAM, just for colonies)
        img_thres = fill_mask_holes(img_thres)
    
        if close > 0:
            img_thres = cv2.erode(img_thres, kernel, iterations = 1)

        #scale = self.level_downsamples[seg_level]
        #scaled_ref_patch_area = int(ref_patch_size**2 / (scale[0] * scale[1]))

        self.colony.mask = img_thres

        #find and filter contours (this part may not be necessary for mask-based)
        contours, hierarchy = cv2.findContours(img_thres, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours (RETR_CCOMP only retreives two-level contour)
        
        #print(contours[0].shape)
        if hierarchy is not None:
            hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # filter_params = filter_params.copy()
        # filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area
        # filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area
        # if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
        # self.contours_colony = self.scaleContourDim(foreground_contours, scale)
        # self.holes_colony = self.scaleHolesDim(hole_contours, scale)
        self.contours_colony = contours
        self.holes_colony = None # for now

        max_area = -1
        max_area_contour = None

        # Iterate over all contours in the "colonies" list
        for contour_colony in self.contours_colony:
            # Calculate the area of the current contour
            area = cv2.contourArea(contour_colony)
            
            # Check if the current contour has a larger area than the maximum area found so far
            if area > max_area:
                max_area = area
                max_area_contour = contour_colony
        self.contours_colony = [max_area_contour]

        if len(keep_ids) > 0:
            contour_ids = set(keep_ids) - set(exclude_ids)
        else:
            contour_ids = set(np.arange(len(self.contours_colony))) - set(exclude_ids)

        self.contours_colony = [self.contours_colony[i] for i in contour_ids]
        if self.holes_colony is not None:
            self.holes_colony = [self.holes_colony[i] for i in contour_ids]

    def createPatches_bag_hdf5(self, save_path, patch_size=256, step_size=256, save_coord=True): # not used in fp

        print("Creating patches for: ", self.name, "...",)
        #for idx, cont in enumerate(contours):
        patch_gen = self._getPatchGenerator(save_path, patch_size, step_size)
        
        if self.hdf5_file is None:
            first_patch = next(patch_gen)
            # try:
            #     first_patch = next(patch_gen)
            # # empty contour, continue
            # except StopIteration:
            #     continue

            file_path = initialize_hdf5_bag(first_patch, save_coord=save_coord)
            self.hdf5_file = file_path

        for patch in patch_gen:
            savePatchIter_bag_hdf5(patch)

        return self.hdf5_file
    
    def _getPatchGenerator(self, save_path, patch_size=256, step_size=256): # not used in fp
        #start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])
        #print("Bounding Box:", start_x, start_y, w, h)
        #print("Contour Area:", cv2.contourArea(cont))
        
        start_x = 0.0
        start_y = 0.0
        stop_x = self.colony.dimension[0]-patch_size+1 # 2nd element in the dataset size (it's flipped)
        stop_y = self.colony.dimension[1]-patch_size+1 # 0th element in the dataset size (it's flipped)

        count = 0
        for y in range(start_y, stop_y, step_size):
            for x in range(start_x, stop_x, step_size):

                if sum(self.colony.mask[x:x+patch_size,y:y+patch_size])/(patch_size**2) < 0.5:
                    continue
                
                count+=1
                
                patch_PIL = self.colony.read_region((x,y),0,(patch_size,patch_size)) #patch_PIL = self.wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')

                patch_info = {'x':x, 'y':y,
                'patch_PIL':patch_PIL, 'name':self.name, 'save_path':save_path}

                yield patch_info
        
        print("patches extracted: {}".format(count))

    def process_contours(self, save_path, patch_level=0, patch_size=512, step_size=256, **kwargs):
        save_path_hdf5 = os.path.join(save_path, str(self.name) + '.h5')
        print("Creating patches for: ", self.name, "...",)
        n_contours = len(self.contours_colony)
        print("Total number of contours to process: ", n_contours)
        fp_chunk_size = math.ceil(n_contours * 0.05)
        init = True
        for idx, cont in enumerate(self.contours_colony):
            if self.holes_colony is not None:
                contour_holes = self.holes_colony[idx]
            else:
                contour_holes = None
            if (idx + 1) % fp_chunk_size == fp_chunk_size:
                print('Processing contour {}/{}'.format(idx, n_contours))
            
            asset_dict, attr_dict = self.process_contour(cont, contour_holes, patch_level, save_path, patch_size, step_size, **kwargs)
            if len(asset_dict) > 0:
                if init:
                    save_hdf5(save_path_hdf5, asset_dict, attr_dict, mode='w')
                    init = False
                else:
                    save_hdf5(save_path_hdf5, asset_dict, mode='a')

        return self.hdf5_file


    def process_contour(self, cont, contour_holes, patch_level, save_path, patch_size = 512, step_size = 256,
                        contour_fn='four_pt', use_padding=True, top_left=None, bot_right=None):
        start_x, start_y, w, h = cv2.boundingRect(cont) if cont is not None else (0, 0, self.level_dim[patch_level][0], self.level_dim[patch_level][1])

        patch_downsample = (int(self.level_downsamples[patch_level][0]), int(self.level_downsamples[patch_level][1]))
        ref_patch_size = (patch_size*patch_downsample[0], patch_size*patch_downsample[1])
        

        start_x = 0
        start_y = 0
        
        stop_x = self.colony.dimension[0]-ref_patch_size[0]+1 # 2nd element in the dataset size (it's flipped)
        stop_y = self.colony.dimension[1]-ref_patch_size[1]+1 # 0th element in the dataset size (it's flipped)
        
        step_size_x = step_size * patch_downsample[0]
        step_size_y = step_size * patch_downsample[1]
        x_range = np.arange(start_x, stop_x, step=step_size_x)
        y_range = np.arange(start_y, stop_y, step=step_size_y)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose()

        
        coords = []
        for y in range(start_y, stop_y, step_size_x):
            for x in range(start_x, stop_x, step_size_y): 
                if np.sum(self.colony.mask[x:x+patch_size,y:y+patch_size])/(patch_size**2) < 0.2: # 0.2 is arbitrarily set by gkim
                    continue
                coords.append([x,y])
        coords = np.array(coords)

        if len(coords)>0:
            asset_dict = {'coords' :          coords}
            
            attr = {'patch_size' :            patch_size, # To be considered...
                    'patch_level' :           patch_level,
                    'downsample':             self.level_downsamples[patch_level],
                    'downsampled_level_dim' : tuple(np.array(self.level_dim[patch_level])),
                    'level_dim':              self.level_dim[patch_level],
                    'name':                   self.name,
                    'save_path':              save_path}

            attr_dict = { 'coords' : attr}
            return asset_dict, attr_dict

        else:
            return {}, {}


    def visWCI(self, vis_level=0, color = (0,255,0), hole_color = (0,0,0), annot_color=(255,0,0), 
                    line_thickness=250, max_size=None, top_left=None, bot_right=None, custom_downsample=1, view_slide_only=False,
                    number_contours=False, seg_display=True, annot_display=True):
        
        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]]
        
        if top_left is not None and bot_right is not None:
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)
        else:
            top_left = (0,0)
            region_size = self.level_dim[vis_level]

        img = self.colony.read_region(top_left, vis_level, region_size)
        img = np.max(img, axis = 2)
        img = (img-13300.)/(14000.-13300.)
        img[img<0] = 0
        img[img>1] = 1
        img = (img*255).astype(np.uint8)
        img = np.array(Image.fromarray(img).convert("RGB"))

        #img = Image.fromarray(img)#.convert("RGB")
        #img = np.array(self.colony.read_region(top_left, vis_level, region_size).convert("RGB"))
        
        if not view_slide_only:
            offset = tuple(-(np.array(top_left) * scale).astype(int))
            line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
            if self.contours_colony is not None and seg_display:
                if not number_contours:
                    cv2.drawContours(img, self.scaleContourDim(self.contours_colony, scale), 
                                     -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)

                else: # add numbering to each contour
                    for idx, cont in enumerate(self.contours_colony):
                        contour = np.array(self.scaleContourDim(cont, scale))
                        M = cv2.moments(contour)
                        cX = int(M["m10"] / (M["m00"] + 1e-9))
                        cY = int(M["m01"] / (M["m00"] + 1e-9))
                        # draw the contour and put text next to center

                        cv2.drawContours(img,  [contour], -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
                        cv2.putText(img, "{}".format(idx), (cX, cY),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 10)

                if self.holes_colony is not None:
                    for holes in self.holes_colony:
                        cv2.drawContours(img, self.scaleContourDim(holes, scale), 
                                        -1, hole_color, line_thickness, lineType=cv2.LINE_8)
                
            if self.contours_tumor is not None and annot_display:
                cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), 
                                 -1, annot_color, line_thickness, lineType=cv2.LINE_8, offset=offset)
        
        img = Image.fromarray(img)
    
        w, h = img.size
        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img

    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    
    def visHeatmap(self, scores, coords, vis_level=-1, 
                   top_left=None, bot_right=None,
                   patch_size=(384, 384), 
                   blank_canvas=False, canvas_color=(220, 20, 50), alpha=0.4, 
                   blur=False, overlap=0.0, 
                   segment=True, use_holes=True,
                   convert_to_percentiles=False, 
                   binarize=False, thresh=0.5,
                   max_size=None,
                   custom_downsample = 1,
                   cmap='coolwarm'):

        """
        Args:
            scores (numpy array of float): Attention scores 
            coords (numpy array of int, n_patches x 2): Corresponding coordinates (relative to lvl 0)
            vis_level (int): WSI pyramid level to visualize
            patch_size (tuple of int): Patch dimensions (relative to lvl 0)
            blank_canvas (bool): Whether to use a blank canvas to draw the heatmap (vs. using the original slide)
            canvas_color (tuple of uint8): Canvas color
            alpha (float [0, 1]): blending coefficient for overlaying heatmap onto original slide
            blur (bool): apply gaussian blurring
            overlap (float [0 1]): percentage of overlap between neighboring patches (only affect radius of blurring)
            segment (bool): whether to use tissue segmentation contour (must have already called self.segmentTissue such that 
                            self.contours_tissue and self.holes_tissue are not None
            use_holes (bool): whether to also clip out detected tissue cavities (only in effect when segment == True)
            convert_to_percentiles (bool): whether to convert attention scores to percentiles
            binarize (bool): only display patches > threshold
            threshold (float): binarization threshold
            max_size (int): Maximum canvas size (clip if goes over)
            custom_downsample (int): additionally downscale the heatmap by specified factor
            cmap (str): name of matplotlib colormap to use
        """

        ### following lines set the tissue mask, regardless of the argument segment (added by gkim)
        contours_ori = self.contours_colony
        self.segmentColony()
        coords = areafilter_coords(self.colony, coords, patch_size = patch_size[0]) #added by gkim
        
        self.contours_colony = contours_ori
        self.colony.mask = np.ones(self.colony.mask.shape)
        #pdb.set_trace()
        #self.contours_colony = None
        # self.segmentColony_fromcoords(coords, patch_size = patch_size[0]) # I'm not sure but this doesn't work (gkim)
        
        if vis_level < 0:
            vis_level = self.colony.get_best_level_for_downsample(32)

        downsample = self.level_downsamples[vis_level]
        scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
                
        if len(scores.shape) == 2:
            scores = scores.flatten()

        if binarize:
            if thresh < 0:
                threshold = 1.0/len(scores)
                
            else:
                threshold =  thresh
        
        else:
            threshold = 0.0

        ##### calculate size of heatmap and filter coordinates/scores outside specified bbox region #####
        if top_left is not None and bot_right is not None:
            scores, coords = screen_coords(scores, coords, top_left, bot_right)
            coords = coords - top_left
            top_left = tuple(top_left)
            bot_right = tuple(bot_right)
            w, h = tuple((np.array(bot_right) * scale).astype(int) - (np.array(top_left) * scale).astype(int))
            region_size = (w, h)

        else:
            region_size = self.level_dim[vis_level]
            top_left = (0,0)
            bot_right = self.level_dim[0]
            w, h = region_size

        patch_size  = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)
        coords = np.ceil(coords * np.array(scale)).astype(int)
        
        print('\ncreating heatmap for: ')
        print('top_left: ', top_left, 'bot_right: ', bot_right)
        print('w: {}, h: {}'.format(w, h))
        print('scaled patch size: ', patch_size)

        ###### normalize filtered scores ######
        if convert_to_percentiles:
            scores = to_percentiles(scores) 

        scores /= 100
        
        ######## calculate the heatmap of raw attention scores (before colormap) 
        # by accumulating scores over overlapped regions ######
        
        # heatmap overlay: tracks attention score over each pixel of heatmap
        # overlay counter: tracks how many times attention score is accumulated over each pixel of heatmap
        overlay = np.full(region_size, 0).astype(float) # removed flip by gkim from np.flip(region_size)
        counter = np.full(region_size, 0).astype(np.uint16) # removed flip by gkim from np.flip(region_size)
        count = 0
        for idx in range(len(coords)):
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:
                if binarize:
                    score=1.0
                    count+=1
            else:
                score=0.0
            # accumulate attention
            #overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score #gkim fix -> index 0 is x and 1 is y
            overlay[coord[0]:coord[0]+patch_size[0],coord[1]:coord[1]+patch_size[1]] += score
            # accumulate counter
            #counter[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += 1 #gkim fix -> index 0 is x and 1 is y
            counter[coord[0]:coord[0]+patch_size[0],coord[1]:coord[1]+patch_size[1]] += 1

        if binarize:
            print('\nbinarized tiles based on cutoff of {}'.format(threshold))
            print('identified {}/{} patches as positive'.format(count, len(coords)))
        
        # fetch attended region and average accumulated attention
        zero_mask = counter == 0

        if binarize:
            overlay[~zero_mask] = np.around(overlay[~zero_mask] / counter[~zero_mask])
        else:
            overlay[~zero_mask] = overlay[~zero_mask] / counter[~zero_mask]
        del counter 
        if blur:
            overlay = cv2.GaussianBlur(overlay,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if segment:
            tissue_mask = self.get_seg_mask(np.flip(region_size), scale, use_holes=use_holes, offset=tuple(top_left)) # changed by gkim to match overlay/counter
            # tissue_mask = self.colony.mask# added by gkim for consistency
        
        if not blank_canvas:
            # downsample original image and use as canvas
            # img = np.array(self.colony.read_region(top_left, vis_level, region_size).convert("RGB")) # replaced by GKim
            img = self.colony.read_region(top_left, vis_level, region_size)
            img = np.max(img, axis = 2)
            img = (img-13300.)/(14000.-13300.)
            img[img<0] = 0
            img[img>1] = 1
            img = (img*255).astype(np.uint8)
            img = np.array(Image.fromarray(img).convert("RGB"))

        else:
            # use blank canvas
            img = np.array(Image.new(size=region_size, mode="RGB", color=(255,255,255))) 

        #return Image.fromarray(img) #raw image

        print('\ncomputing heatmap image')
        print('total of {} patches'.format(len(coords)))
        twenty_percent_chunk = max(1, int(len(coords) * 0.2))

        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        
        for idx in range(len(coords)):
            if (idx + 1) % twenty_percent_chunk == 0:
                print('progress: {} coords/{} coords'.format(idx, len(coords)))
            
            score = scores[idx]
            coord = coords[idx]
            if score >= threshold:

                # attention block
                # raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] #gkim fix -> index 0 is x and 1 is y
                raw_block = overlay[coord[0]:coord[0]+patch_size[0],coord[1]:coord[1]+patch_size[1]]

                # image block (either blank canvas or orig image)
                # img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy() #gkim fix -> index 0 is x and 1 is y
                img_block = img[coord[0]:coord[0]+patch_size[0],coord[1]:coord[1]+patch_size[1]].copy()

                # color block (cmap applied to attention block)
                color_block = (cmap(raw_block) * 255)[:,:,:3].astype(np.uint8)

                if segment:
                    
                    # tissue mask block
                    # mask_block = tissue_mask[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] #gkim fix -> index 0 is x and 1 is y                    
                    #gkim fix -> to account for block size smaller than patch
                    mask_block = tissue_mask[coord[0]:coord[0]+min(patch_size[0],img_block.shape[0]),coord[1]:coord[1]+min(patch_size[1],img_block.shape[1])]

                    #print([img_block.shape, color_block.shape]) # gkim for checking
                    # copy over only tissue masked portion of color block
                    try:
                        img_block[mask_block] = color_block[mask_block]
                    except:
                        pdb.set_trace()
                else:
                    # copy over entire color block
                    img_block = color_block

                # rewrite image block
                #gkim fix -> index 0 is x and 1 is y   
                img[coord[0]:coord[0]+min(patch_size[0],img_block.shape[0]), coord[1]:coord[1]+min(patch_size[1],img_block.shape[1])] = img_block.copy()
        
        #return Image.fromarray(img) #overlay
        print('Done')
        del overlay

        if blur:
            img = cv2.GaussianBlur(img,tuple((patch_size * (1-overlap)).astype(int) * 2 +1),0)  

        if alpha < 1.0:
            img = self.block_blending(img, vis_level, top_left, bot_right, alpha=alpha, blank_canvas=blank_canvas, block_size=768)#1024)
        
        img = Image.fromarray(img)
        w, h = img.size

        if custom_downsample > 1:
            img = img.resize((int(w/custom_downsample), int(h/custom_downsample)))

        if max_size is not None and (w > max_size or h > max_size):
            resizeFactor = max_size/w if w > h else max_size/h
            img = img.resize((int(w*resizeFactor), int(h*resizeFactor)))
       
        return img

    
    def block_blending(self, img, vis_level, top_left, bot_right, alpha=0.5, blank_canvas=False, block_size=1024):
        #### gkim fixed: ridiculous how they juggle between C and fortran convention to expresss coordinate
        #### they prally wanted to slow down other groups using this despite earning credit for "open" research by sharing on github smh
        print('\ncomputing blend')
        downsample = self.level_downsamples[vis_level]
        w = img.shape[0]#[1]#gkim fixed
        h = img.shape[1]#[0]#gkim fixed
        block_size_x = min(block_size, w)
        block_size_y = min(block_size, h)
        print('using block size: {} x {}'.format(block_size_x, block_size_y))

        shift = top_left # amount shifted w.r.t. (0,0)
        for x_start in range(top_left[0], bot_right[0], block_size_x * int(downsample[0])):
            for y_start in range(top_left[1], bot_right[1], block_size_y * int(downsample[1])):
                #print(x_start, y_start)

                # 1. convert wsi coordinates to image coordinates via shift and scale
                x_start_img = int((x_start - shift[0]) / int(downsample[0]))
                y_start_img = int((y_start - shift[1]) / int(downsample[1]))
                
                # 2. compute end points of blend tile, careful not to go over the edge of the image
                y_end_img = min(h, y_start_img+block_size_y)
                x_end_img = min(w, x_start_img+block_size_x)

                if y_end_img == y_start_img or x_end_img == x_start_img:
                    continue
                print('start_coord: {} end_coord: {}'.format((x_start_img, y_start_img), (x_end_img, y_end_img)))
                
                # 3. fetch blend block and size
                blend_block = img[x_start_img:x_end_img, y_start_img:y_end_img] #[y_start_img:y_end_img, x_start_img:x_end_img] #gkim fixed
                blend_block_size = (x_end_img-x_start_img,y_end_img-y_start_img) # (y_end_img-y_start_img, x_end_img-x_start_img) #gkim fixed
                
                #print(blank_canvas)
                if not blank_canvas:
                    # 4. read actual wsi block as canvas block
                    pt = (x_start, y_start)
                    # canvas = np.array(self.colony.read_region(pt, vis_level, blend_block_size).convert("RGB")) # replaced by GKim
                    #print(blend_block_size)

                    canvas = self.colony.read_region(pt, vis_level, blend_block_size)
                    ####### This line makes canvas size weird! #######

                    #print(canvas.shape)
                    canvas = np.max(canvas, axis = 2)
                    #print(canvas.shape)
                    canvas = (canvas-13300.)/(14000.-13300.)
                    canvas[canvas<0] = 0
                    canvas[canvas>1] = 1
                    canvas = (canvas*255).astype(np.uint8)
                    canvas = np.array(Image.fromarray(canvas).convert("RGB"))

                else:
                    # 4. OR create blank canvas block
                    canvas = np.array(Image.new(size=blend_block_size, mode="RGB", color=(255,255,255)))

                # 5. blend color block and canvas block
                img[x_start_img:x_end_img,y_start_img:y_end_img,:] = cv2.addWeighted(blend_block, alpha, canvas, 1 - alpha, 0, canvas)
                #x-y switched and last dim added by gkim
        return img

    def get_seg_mask(self, region_size, scale, use_holes=False, offset=(0,0)):
        print('\ncomputing foreground tissue mask')
        tissue_mask = np.full(np.flip(region_size), 0).astype(np.uint8)
        contours_colony = self.scaleContourDim(self.contours_colony, scale)
        offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))

        # contours_holes = self.scaleHolesDim(self.holes_colony, scale) # removed by GKim bc we don't exclude holes now
        # contours_colony, contours_holes = zip(*sorted(zip(contours_colony, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True)) # removed by GKim bc we don't exclude holes now
        for idx in range(len(contours_colony)):
            cv2.drawContours(image=tissue_mask, contours=contours_colony, contourIdx=idx, color=(1), offset=offset, thickness=-1)

            # if use_holes: # removed by GKim bc we don't exclude holes now
            #     cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0), offset=offset, thickness=-1) # removed by GKim bc we don't exclude holes now
            # contours_holes = self._scaleContourDim(self.holes_colony, scale, holes=True, area_thresh=area_thresh)
                
        tissue_mask = tissue_mask.astype(bool)
        print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
        return tissue_mask


    @staticmethod
    def scaleContourDim(contours, scale):
        return [np.array(cont * scale, dtype='int32') for cont in contours]

    @staticmethod
    def scaleHolesDim(contours, scale):
        return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]
            
def areafilter_coords(colony, coords, patch_size = 384, thres_area_ratio = 0.2):
    coords_new = []
    for idx, coord_temp in enumerate(coords):
        x = coord_temp[0]
        y = coord_temp[1]
        if np.sum(colony.mask[x:x+int(patch_size),y:y+int(patch_size)])/(patch_size**2) < 0.2:
            continue
        coords_new.append([x,y])
    return coords_new