from PIL import Image
import numpy as np
import os
import json
from clip_generate_attr_aff import generate_attr_aff, generate_human_feature
from tqdm import tqdm
dataset_path = './hico_20160224_det'
with open(os.path.join(dataset_path, "safe_train.json"), 'r') as file:
    safe_train = json.load(file)
with open(os.path.join(dataset_path, "train.json"), 'r') as file:
    train_anno = json.load(file)
with open(os.path.join(dataset_path, "action_list.json"), 'r') as file:
    action_list = json.load(file)
with open(os.path.join(dataset_path, "object_list.json"), 'r') as file:
    object_list = json.load(file)
graph = []
folder = 'images'
train_folder = 'train2015'
train_dir = os.path.join(dataset_path, folder)
train_dir = os.path.join(train_dir, train_folder)
for train in tqdm(safe_train):
    train_img = Image.open(os.path.join(train_dir, train))
    interactions = train_anno[train]
    for key in interactions.keys():
        interaction = interactions[key]
        interact_dict = dict()
        interact_dict['id'] = train
        act = interaction['act']
        obj = interaction['obj']
        act_label = action_list.index(act)
        obj_label = object_list.index(obj)
        edge_label = [act_label, obj_label]
        obj_feature_dict = dict()
        human_feature_dict = dict()
        edge_feature_dict = dict()
        for hid, bboxhuman in enumerate(interaction['bboxhuman']):
            b_human = [bboxhuman[0], bboxhuman[2], bboxhuman[1], bboxhuman[3]]
            if hid not in human_feature_dict.keys():
                train_human = train_img.crop(b_human)
                human_feature_dict[hid] = generate_human_feature(train_human)
            for oid, bboxobject in enumerate(interaction['bboxobject']):
                b_object = [bboxobject[0], bboxobject[2], bboxobject[1], bboxobject[3]]
                if oid not in obj_feature_dict.keys():
                    train_object = train_img.crop(b_object)
                    obj_feature_dict[oid] = generate_attr_aff(train_object)
                edge_feature_dict[str(hid)+';'+str(oid)] = ((bboxhuman[0] + bboxhuman[1] - (bboxobject[0] + bboxobject[1])) / 2, 
                                (bboxhuman[2] + bboxhuman[3] - (bboxobject[2] + bboxobject[3])) / 2)  
        interact_dict['obj_feature'] = obj_feature_dict
        interact_dict['human_feature'] = human_feature_dict
        interact_dict['edge_label'] = edge_label
        interact_dict['edge_feature'] = edge_feature_dict
        graph.append(interact_dict)
print(len(safe_train))
print(len(graph))
with open("./graph_train_multi.json", "w") as file: 
    json.dump(graph, file)