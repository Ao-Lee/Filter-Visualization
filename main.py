from keras.applications.resnet50 import ResNet50
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt

def GetFirstLayerWeight():
    net = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    layer = net.layers[1]
    # print(layer.name)
    # print(layer.input_shape)
    # print(layer.output_shape)
    w_tensor,b_tensor = layer.weights
    # print(w_tensor.shape)
    w = K.eval(w_tensor)
    return w

def ScaleWeights(w):
    w -= np.min(w)
    factor = 255/np.max(w)
    w = w*factor
    w = w.astype(np.uint8)
    return w
    
def ShowImg(img, title=''):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.title(title)
    plt.show()
    plt.close()
    
if __name__=='__main__':
    W = GetFirstLayerWeight()
    W = ScaleWeights(W)
    
    num_filter = 64
    assert num_filter == W.shape[-1]
    plt.figure(figsize=(10,10)) 

    for idx in range(num_filter):
        w = W[..., idx]
        ax = plt.subplot(8, 8, idx+1)
        ax.set_xticks(())
        ax.set_yticks(())
        ax.imshow(w)
    
    
    

