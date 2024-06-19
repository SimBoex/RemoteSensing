import numpy as np

class HEX_to_RGB:
    def __init__(self) -> None:
        pass
    
    def convert(self,value):
        assert type(value) == str
        value = value.lstrip('#')
        transf_value = np.array(tuple(int(value[i:i+2], 16) for i in (0, 2, 4))) # 60, 16, 152
        return transf_value
    


class RGB_to_2DLabels:
    

        