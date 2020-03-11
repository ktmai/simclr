"""Helper code for data augmentations used in training
"""

import random
from scipy.ndimage import gaussian_filter
from torchvision import transforms


def get_color_distortion(s=1.0):
    """
    Colour distortion code based in jittering and dropping
    Implementation from Appendix A of the SimCLR paper
    
    Args:
        s = strength parameter of the colour distortion
        
    Returns: 
        color_distort = transformed image
    """
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter,rnd_gray])
    return color_distort


class GetGaussianBlur(object):
    """
    Gaussian blur transformation used for training model on ImageNet.
    Image is blurred 50% of the time and sigma is a value between 0.1 and 0.2
    
    Args:
        sample: Sample data used for transformatins
        
    Returns:
        Transformed sample
    """
    def __call__(self, sample):
        # 50% of time, don't do anything
        if random.randint(0, 1) == 0:
            return sample
        sigma = random.uniform(0.1, 0.2)
        return gaussian_filter(sample, sigma=sigma)
