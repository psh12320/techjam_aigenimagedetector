import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image, ImageFilter
import io
import random
import numpy as np

class JPEGCompression:
    def __init__(self, quality):
        self.quality = quality

    def __call__(self, img):
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer)

class GaussianBlur:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

class GaussianNoise:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        img_array = np.array(img).astype(np.float32) / 255.0
        noise = np.random.normal(0, self.sigma, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 1)
        return Image.fromarray((noisy * 255).astype(np.uint8))

class ResizeDownUp:
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        w, h = img.size
        small = img.resize((int(w * self.scale), int(h * self.scale)), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

def get_competition_transforms(transform_type, **kwargs):
    """Get specific competition transform"""
    base_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if transform_type == 'jpeg':
        return T.Compose([
            JPEGCompression(quality=kwargs.get('quality', 70)),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_type == 'blur':
        return T.Compose([
            GaussianBlur(sigma=kwargs.get('sigma', 1.0)),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_type == 'noise':
        return T.Compose([
            T.Resize((224, 224)),
            GaussianNoise(sigma=kwargs.get('sigma', 0.05)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_type == 'resize':
        return T.Compose([
            ResizeDownUp(scale=kwargs.get('scale', 0.5)),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_type == 'color_jitter':
        return T.Compose([
            T.Resize((224, 224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif transform_type == 'crop':
        return T.Compose([
            T.Resize((224, 224)),
            T.CenterCrop(int(224 * 0.8)),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return base_transform

class RobustAugmentation:
    """Augmentation strategy for training with competition transforms"""
    def __init__(self, p_augment=0.3, transform_types=None):
        self.p_augment = p_augment
        self.transform_types = transform_types or ['jpeg', 'blur', 'noise', 'color_jitter', 'crop']

        self.transform_params = {
            'jpeg': [90, 70, 50, 30],
            'blur': [0.5, 1.0, 2.0],
            'noise': [0.02, 0.05, 0.10],
            'resize': [0.5, 0.25],
            'color_jitter': [0.2],
            'crop': [0.8]
        }

        self.base_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        if random.random() > self.p_augment:
            return self.base_transform(img)

        # Apply 1-3 random transforms
        n_transforms = random.randint(1, 3)
        selected = random.sample(self.transform_types, min(n_transforms, len(self.transform_types)))

        for transform_type in selected:
            param = random.choice(self.transform_params[transform_type])

            if transform_type == 'jpeg':
                img = JPEGCompression(quality=param)(img)
            elif transform_type == 'blur':
                img = GaussianBlur(sigma=param)(img)
            elif transform_type == 'noise':
                img = T.Resize((224, 224))(img)
                img = GaussianNoise(sigma=param)(img)
            elif transform_type == 'resize':
                img = ResizeDownUp(scale=param)(img)
            elif transform_type == 'color_jitter':
                img = T.Resize((224, 224))(img)
                img = T.ColorJitter(brightness=param, contrast=param, saturation=param)(img)
            elif transform_type == 'crop':
                img = T.Resize((224, 224))(img)
                img = T.CenterCrop(int(224 * param))(img)

        return self.base_transform(img)
