import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import numpy as np
from PIL import Image
import random
import PIL
import torchvision
import numbers

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,img,meta):
        for t in self.transforms:
            img,meta = t(img,meta)
        return img,meta

class Normalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def __call__(self, image, label):
        # if  not isinstance(image,Image.Image):
        #     image=Image.fromarray(image)
        image=F.normalize(image,self.mean,self.std)
        return image,label

class UnNormalize(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std

    def __call__(self, image):
        for t, m, s in zip (image, self.mean, self.std):
            t.mul_ (s).add_ (m)

        return image

class UnNormalize_clip(object):
    def __init__(self,mean,std):
        self.mean=mean
        self.std=std
        self.trans=UnNormalize(mean,std)

    def __call__(self, clip):
        for image in clip:
            for t, m, s in zip (image, self.mean, self.std):
                t.mul_ (s).add_ (m)
        return clip

class ToTensor(object):
    def __call__(self, img,label):
        if  not isinstance(img,Image.Image):
            print(len(img))
            img=Image.fromarray(np.array(img),mode="RGB")

        return F.to_tensor(img),torch.tensor(label)

class RandomColorjitter (object):
    def __init__(self,brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2):
        self.trans=transforms.ColorJitter (brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, img,target):
        p = np.random.uniform (0, 1, 1)
        if p>0.5:
            if (len(np.array(img).shape) == 2):
            # add that third color dim
                img = img.reshape(img.shape[0], img.shape[1], 3)
            if  not isinstance(img,Image.Image) or not isinstance(img,torch.Tensor):
                img=Image.fromarray(np.array(img),mode="RGB")

            img=self.trans(img)
        
        return img, target

class RandomCrop (object):
    def __init__(self,size=256,pad=0, pad_if_needed=False, fill=0):
        self.size=size
        self.trans=transforms.RandomCrop(size,padding=pad, pad_if_needed=pad_if_needed, fill=fill)

    def __call__(self, img,target):
        if  not isinstance(img,Image.Image):
            img=nptopil(img)
        img=self.trans(img)
        
        return img, target
class Resize (object):
    def __init__(self,size=256):
        self.size=size
        self.trans=transforms.Resize(size)

    def __call__(self, img,target):
        if  not isinstance(img,Image.Image):
            img=nptopil(img)
        img=self.trans(img)
        
        return img, target

class CenterCrop (object):
    def __init__(self,size=256):
        self.size=size
        self.trans=transforms.CenterCrop(size)

    def __call__(self, img, target):
        if  not isinstance(img,Image.Image):
            img=nptopil(img)
        
        img=self.trans(img)

        return img, target

def _is_tensor_clip(clip):
    return torch.is_tensor(clip) and clip.ndimension() == 4


class ToTensor_clip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, clip,label):
        # swap color axis because
        # numpy image: batch_size x H x W x C
        # torch image: batch_size x C X H X W
        if not isinstance(clip[0],Image.Image):
            clip=[nptopil(img) for img in clip]
        if  isinstance(clip[0],Image.Image):
            clip=[F.to_tensor(img) for img in clip]
            
        return torch.stack(clip,dim=0),torch.from_numpy(label)


class Normalize_clip(object):
    """Normalize a clip with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, clip,label):
        """
        Args:
            clip (Tensor): Tensor clip of size (T, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor clip.
        """
        return F.normalize(clip, self.mean, self.std),label


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ColorJitter_clip(object):
    """Randomly change the brightness, contrast and saturation and hue of the clip
    Args:
    brightness (float): How much to jitter brightness. brightness_factor
    is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
    contrast (float): How much to jitter contrast. contrast_factor
    is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
    saturation (float): How much to jitter saturation. saturation_factor
    is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
    hue(float): How much to jitter hue. hue_factor is chosen uniformly from
    [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.25, hue=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def get_params(self, brightness, contrast, saturation, hue):
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
        else:
            brightness_factor = None

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
        else:
            contrast_factor = None

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
        else:
            saturation_factor = None

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
        else:
            hue_factor = None
        return brightness_factor, contrast_factor, saturation_factor, hue_factor

    def __call__(self, clip,label):
        """
        Args:
        clip (list): list of PIL.Image
        Returns:
        list PIL.Image : list of transformed PIL.Image
        """
        if isinstance(clip[0], np.ndarray):
            clip=[nptopil(img) for img in clip]

        brightness, contrast, saturation, hue = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue)

        # Create img transform function sequence
        img_transforms = []
        if brightness is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_brightness(img, brightness))
        if saturation is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_saturation(img, saturation))
        if hue is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_hue(img, hue))
        if contrast is not None:
            img_transforms.append(lambda img: torchvision.transforms.functional.adjust_contrast(img, contrast))
        random.shuffle(img_transforms)

        # Apply to all images
        jittered_clip = []
        for img in clip:
            for func in img_transforms:
                jittered_img = func(img)
            jittered_clip.append(jittered_img)

        return jittered_clip,label

def nptopil(arr):
    img = Image.fromarray(arr.astype('uint8')).convert('RGB')
    return img

class RandomCrop_clip(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size,pad=10):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size
        self.pad=pad
        self.trans=transforms.RandomCrop(size,padding=pad)

    def __call__(self, clip,label):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """

        if  not isinstance(clip[0],Image.Image):
            clip=[nptopil(img) for img in clip]
        
        transed=[self.trans(img) for img in clip]

        return transed,label


class CenterCrop_clip(object):
    """Extract random crop at the same location for a list of images
    Args:
    size (sequence or int): Desired output size for the
    crop in format (h, w)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            size = (size, size)

        self.size = size
        self.trans=transforms.CenterCrop(size)

    def __call__(self, clip,label):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray
        Returns:
        PIL.Image or numpy.ndarray: Cropped list of images
        """

        if  not isinstance(clip[0],Image.Image):
            clip=[nptopil(img) for img in clip]
        
        transed=[self.trans(img) for img in clip]

        return transed,label
        

if __name__ == '__main__':
    input=np.zeros((256,256,3))
    label=np.zeros((1,24))
    # transform=Compose(
    #     [
    #     RandomCrop(256),
    #     ColorJitter_clip(),
    #     ToTensor_clip(),
    #     Normalize_clip(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    #     )
    transform=Compose(
        [
            CenterCrop(225),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input,label=transform(img=input,meta=label)
    print(input.shape)
