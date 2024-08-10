import numpy as np
import torchvision.transforms as T
from imgaug import augmenters as iaa


    
class ImgAugTransform:
    """
    Wrapper to allow imgaug to work with Pytorch transformation pipeline
    """

    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Sometimes(0.8, iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5)
            ])),
            iaa.Sometimes(0.5, iaa.Sequential([
                iaa.Crop(percent=(0.1, 0.2))
            ])),
            iaa.LinearContrast((0.75, 1.5)),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.8,
                          iaa.Affine(
                              scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                              translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                              rotate=(-25, 25),
                              shear=(-8, 8)
                          )),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img

im_size = 224

val_transform = T.Compose([T.Resize((im_size, im_size)),
                           T.ToTensor(),
                           T.Normalize((0.6984, 0.5219, 0.4197), (0.1396, 0.1318, 0.1236)) 
                           ])

train_transform = T.Compose([T.Resize((im_size, im_size)),
                            ImgAugTransform(),
                             T.ToTensor(),
                             # value calculated by calculating the mean and std of the dataset
                             T.Normalize((0.6984, 0.5219, 0.4197), (0.1396, 0.1318, 0.1236)) 
                             ])