
import torch
import numpy as np

model = torch.load("model")
model.eval()

size = 150

def predict_image(image):
    print(image)
    out = model(image)
    print(out)
    return out
