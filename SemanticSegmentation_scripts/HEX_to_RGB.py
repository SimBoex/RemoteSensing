import numpy as np

class HEX_to_RGB:
    def __init__(self) -> None:
        pass
    
    def convert(self,value):
        assert type(value) == str
        value = value.lstrip('#')
        transf_value = np.array(tuple(int(value[i:i+2], 16) for i in (0, 2, 4))) 
        return transf_value
    


class RGB_to_2DLabels:
    ''' This takes a list of rgb labels and returns the corrisponding converted mask
    '''
    def __init__(self, labels) -> None:
        self.labels = labels
    
    def convert(self,original_mask):
        conv_mask = np.zeros(original_mask.shape,dtype= np.uint8)
        # new label depends on the order of the list of rgb labels
        new_label = 0
        for el in self.labels:
            # taking the innermost sub-array that is the pixel dim with -1
            conv_mask[np.all(original_mask == el, axis = -1)]  = new_label
            new_label+=1
        # now let's remove the other channels except the first
        conv_mask = conv_mask[:,:,0]
        return conv_mask
        
    




