import h5py
import numpy as np
import cv2

class colony(object): # object to mimic OpenSlide objects in OpenSllide
    def __init__(self,path):
        self.path = path
        self.level_count = 1 #number of levels
        with h5py.File(self.path, 'r') as file:
            self.dimension = (file['/ri'].shape[2], file['/ri'].shape[1])#file['/ri'].shape[[2,1]]
        self.level_dimensions = [self.dimension]#should be a list of dimensions (getting smaller)
        self.level_downsamples = [1] #idk why this should be explicit but well
        self.associated_images = None
        self.color_profile = None
        self.mask = None
    
    def read_region(self, location = (0,0), level = 0, patch_size = (384, 384)):
        whole_img = np.array(h5py.File(self.path, 'r')['/ri'])
        if len(whole_img.shape) > 2: # 3D img
            whole_img = np.swapaxes(whole_img,0,2)
            whole_img = np.resize(whole_img, self.level_dimensions[level] + (whole_img.shape[2],))
        else: # 2D img
            whole_img = np.swapaxes(whole_img,0,1)
            whole_img = np.resize(whole_img, self.level_dimensions[level])

        slices = tuple(slice(loc, loc + siz) for loc, siz in zip(location, patch_size))
        region_img = whole_img[slices]
        return region_img
    
    def get_best_level_for_downsample(self, target_downsample):
        best_level_index = 0
        best_level_diff = abs(1.0 - target_downsample)
        
        for i, dimensions in enumerate(self.level_dimensions):
            level_downsample = self.level_dimensions[0][0] / dimensions[0]
            level_diff = abs(target_downsample - level_downsample)
            
            if level_diff < best_level_diff:
                best_level_index = i
                best_level_diff = level_diff
        
        return best_level_index