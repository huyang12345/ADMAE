import matplotlib.pyplot as plt
import torch
import numpy as np

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def patchify(imgs):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = 16
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = 16
    h = w = int(x.shape[1] ** .5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs
def show_img(img,title=None,index=0):
    if img.dim()==3:
        image = unpatchify(img)
    elif img.dim()==4:
        image = img
    else:
        print('please input size is 1*196*768 or 1*3*224*224')
        return
    image = torch.einsum('nchw->nhwc',image).detach().cpu()
    plt.imshow(torch.clip((image[index] * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    if title:
        plt.title(title)
    plt.axis('off')
    return