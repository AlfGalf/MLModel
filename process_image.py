
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from fastai import *
from fastai.vision.all import *
import numpy as np

from m_utils import itoa

mod_path = "/Users/alfierichards/Documents/GitRepos/IPPythonWebServer/MLModel"

# learn = synth_learner(path=mod_path)
# learn = learn.load("model.pkl")

learner = load_learner(mod_path + '/learn_export.pkl')

def label_func(f):
    return f[0]

data_transforms = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor()
])

def image_loader(loader, image):
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def predict_image(image):
    print(image)

    out = learner.predict(image)
    print(out)
    return 'A'

