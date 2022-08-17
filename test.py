import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from saliency_eval import get_pretrained_saliency_fn
import torchvision.transforms as transforms
from torchvision.utils import save_image

input_transform = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    
])
input_normalize = transforms.Normalize(
    [0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])

sal_fn = get_pretrained_saliency_fn(cuda=False)

img = Image.open('ostrich-009.jpg')
img = input_transform(img)
img = input_normalize(img)
img = img.unsqueeze(0)
# get the saliency map (see get_pretrained_saliency_fn doc for details)
sal_map = sal_fn(img, 9).detach().squeeze()
# sal_img = sal_map.squeeze().detach().numpy()
save_image(sal_map.clone(), 'explanations.jpg')
plt.imshow(sal_map.clone())
plt.show()