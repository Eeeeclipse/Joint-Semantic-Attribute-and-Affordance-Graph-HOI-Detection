import torch
import clip
from PIL import Image
import numpy as np
import os
import json
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_path = './hico_20160224_det'

with open(os.path.join(dataset_path, "action_list.json"), 'r') as file:
    action_list = json.load(file)
with open(os.path.join(dataset_path, "object_list.json"), 'r') as file:
    object_list = json.load(file)


action_token = clip.tokenize(action_list).to(device)
object_token = clip.tokenize(object_list).to(device)


# def generate_attr_aff(image_array):
#     image = Image.fromarray(image_array)
def generate_attr_aff(image):
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        aff_pred, _ = model(image, action_token)
        affordance = aff_pred.softmax(dim=-1).cpu().numpy()
        attr_pred, _ = model(image, object_token)
        attribution = attr_pred.softmax(dim=-1).cpu().numpy()
    return [affordance.tolist(), attribution.tolist()]

# def generate_human_feature(image_array):
#     image = Image.fromarray(image_array)
def generate_human_feature(image):
    image = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        human_pred, _ = model(image, action_token)
        human_feature = human_pred.softmax(dim=-1).cpu().numpy()

    return human_feature.tolist()
