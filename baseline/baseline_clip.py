import torch
import clip
from PIL import Image
import numpy as np
import os
import json
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
dataset_path = './hico_20160224_det'
folder = 'images'
test_folder = 'test2015'
test_dir = os.path.join(dataset_path, folder)
test_dir = os.path.join(test_dir, test_folder)

with open(os.path.join(dataset_path, "safe_test.json"), 'r') as file:
    safe_test = json.load(file)

with open(os.path.join(dataset_path, "action_pair.json"), 'r') as file:
    action_pair = json.load(file)

with open(os.path.join(dataset_path, "test.json"), 'r') as file:
    test_anno = json.load(file)

count = 0
action_token = clip.tokenize(action_pair).to(device)

for test in safe_test:
    print(test)
    test_img = Image.open(os.path.join(test_dir, test))
    interactions = test_anno[test]
    ground_truth = []
    for interaction in interactions.keys():
        ground_truth.append(int(interaction) - 1)
    test_img = preprocess(test_img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, _ = model(test_img, action_token)
        feature = pred.softmax(dim=-1).cpu()
    prediction = feature.argmax(dim=-1).tolist()[0]
    if (prediction in ground_truth) :
        count += 1

print('-' * 80)
print('Clip baseline')
print('Accuracy:')
print(count / len(safe_test))
print('-' * 80)
