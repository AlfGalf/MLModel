
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from fastai import *
from fastai.vision.all import *
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from m_utils import *

mod_path = 'C:\\Users\\Joseph\\Desktop\\Uni\\Y2\\IP\\IPPythonServer\\MLModel'

# learn = synth_learner(path=mod_path)
# learn = learn.load("model.pkl")

learner = load_learner(mod_path + '\\export.pkl')

#data_transforms = transforms.Compose([
#    transforms.Resize(size),
##    transforms.ToTensor()
#])

def image_loader(loader, image):
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def predict_image(image):

    out = learner.predict(image)
    print(out)
    return out[0]

