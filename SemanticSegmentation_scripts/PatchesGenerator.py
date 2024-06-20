import os
import cv2
from PIL import Image
import numpy as np
from patchify import patchify
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import matplotlib.pyplot as plt


'''
Instead of rescaling, i use the crop function. To avoid that same types  objects get different sizes in the process.
I crop from the top left corner, so losing some info  of the images
'''

class PatchesGenerator:
    ''' this takes the patch size such that each new patch have the following shape (patch_size, patch_size)
    and PATH is the path for the dataset directory'''
   
    
    def __init__(self, patch_size, PATH) -> None:
        self.PATH = PATH
        self.patch_size= patch_size
        self.image_dataset = []
        self.mask_dataset = []  
    
    def crop_image_creating(self):
        for path, subdir, files in os.walk(self.PATH):
            # filter out the masks
            name_dir = path.split(os.path.sep)[-1]
            if name_dir == "images":
                images = os.listdir(path)
                for i,name in enumerate(images):
                    if name.endswith(".jpg"):
                        img = cv2.imread(path+"/"+name,1) # read in BRG
                        
                        ##############
                        SIZE_X = (img.shape[1]//self.patch_size)*self.patch_size #Nearest size divisible by our patch size
                        SIZE_Y = (img.shape[0]//self.patch_size)*self.patch_size #Nearest size divisible by our patch size
                        
                        img = Image.fromarray(img)
                        img = img.crop((0,0,SIZE_X,SIZE_Y))
                        img = np.array(img)
                        ##############
                        # from here the images have dimension divisible by the patch size
                        patches_img = patchify(img, (self.patch_size, self.patch_size, 3), step=self.patch_size)  #Step=256 for 256 patches means no overlap

                        scaler = MinMaxScaler()

                        for i in range(patches_img.shape[0]):
                            for j in range(patches_img.shape[1]):
                        
                                single_patch_img = patches_img[i,j,:,:]
                                
                                #Use minmaxscaler instead of just dividing by 255. 
                                single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
                                
                                #single_patch_img = (single_patch_img.astype('float32')) / 255. 
                                single_patch_img = single_patch_img[0] #Drop the extra unecessary dimension that patchify adds.                               
                                self.image_dataset.append(single_patch_img)
        self.image_dataset = np.array(self.image_dataset)

        
    def crop_mask_creating(self):
        for path, subdirs, files in os.walk(self.PATH):
            dirname = path.split(os.path.sep)[-1]
            if dirname == 'masks':   #Find all 'masks' directories
                masks = os.listdir(path)  
                for i, mask_name in enumerate(masks):  
                    if mask_name.endswith(".png"):   
    
                        mask = cv2.imread(path+"/"+mask_name, 1)  # read the img
                        mask = cv2.cvtColor(mask,cv2.COLOR_BGR2RGB)
                        SIZE_X = (mask.shape[1]//self.patch_size)*self.patch_size #Nearest size divisible by our patch size
                        SIZE_Y = (mask.shape[0]//self.patch_size)*self.patch_size #Nearest size divisible by our patch size
                        
                        mask = Image.fromarray(mask)
                        mask = mask.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
                        mask = np.array(mask)             
            
                        #Extract patches from each image
                        patches_mask = patchify(mask, (self.patch_size, self.patch_size, 3), step=self.patch_size)  #Step=256 for 256 patches means no overlap
                
                        for i in range(patches_mask.shape[0]):
                            for j in range(patches_mask.shape[1]):
                                
                                single_patch_mask = patches_mask[i,j,:,:]
                                #single_patch_img = (single_patch_img.astype('float32')) / 255. #No need to scale masks, but you can do it if you want
                                single_patch_mask = single_patch_mask[0] #Drop the extra unecessary dimension that patchify adds.                               
                                self.mask_dataset.append(single_patch_mask) 
                                
        self.mask_dataset =  np.array(self.mask_dataset)
        
        
        
        
        
        
    def check(self):
        image_number = random.randint(0, len(self.image_dataset))
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(np.reshape(self.image_dataset[image_number], (self.patch_size, self.patch_size, 3)))
        plt.subplot(122)
        plt.imshow(np.reshape(self.mask_dataset[image_number], (self.patch_size, self.patch_size, 3)))
        plt.show()
        
