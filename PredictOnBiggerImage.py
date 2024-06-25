import cv2
from PIL import Image
from patchify import patchify, unpatchify
import numpy as np
from sklearn.preprocessing import MinMaxScaler

patch_size = 256
n_classes = 6

def predict_onBiggerImage(patch_size, n_classes, PATH_img, PATH_msk, model):

    scaler = MinMaxScaler()
    img = cv2.imread(PATH_img, 1)
    original_mask = cv2.imread(PATH_msk, 1)
    original_mask = cv2.cvtColor(original_mask,cv2.COLOR_BGR2RGB)

    SIZE_X = (img.shape[1]//patch_size)*patch_size #Nearest size divisible by our patch size
    SIZE_Y = (img.shape[0]//patch_size)*patch_size #Nearest size divisible by our patch size
    large_img = Image.fromarray(img)
    large_img = large_img.crop((0 ,0, SIZE_X, SIZE_Y))  #Crop from top left corner
    #image = image.resize((SIZE_X, SIZE_Y))  #Try not to resize for semantic segmentation
    large_img = np.array(large_img)     


    patches_img = patchify(large_img, (patch_size, patch_size, 3), step=patch_size)  #Step=256 for 256 patches means no overlap
    patches_img = patches_img[:,:,0,:,:,:]

    patched_prediction = []
    for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            
            single_patch_img = patches_img[i,j,:,:,:]
            
            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            single_patch_img = np.expand_dims(single_patch_img, axis=0)
            pred = model.predict(single_patch_img)
            pred = np.argmax(pred, axis=3)
            pred = pred[0, :,:]
                                    
            patched_prediction.append(pred)

    patched_prediction = np.array(patched_prediction)
    patched_prediction = np.reshape(patched_prediction, [patches_img.shape[0], patches_img.shape[1], 
                                                patches_img.shape[2], patches_img.shape[3]])

    unpatched_prediction = unpatchify(patched_prediction, (large_img.shape[0], large_img.shape[1]))

    return unpatched_prediction
