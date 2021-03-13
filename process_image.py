
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from fastai import *
from fastai.vision.all import *
import numpy as np
from m_utils import *
from PIL import Image

mod_path = '/Users/alfierichards/Documents/GitRepos/MLModel/model.pth'

learner = load_learner(mod_path)

def to_byte_arr(image):
    byteIO = io.BytesIO()
    img.save(byteIO, format='PNG')
    byteArr = byteIO.getvalue()
    return byteArr

def predict_image(image):

    out = learner.predict(to_byte_arr(image))
    return out[0]

