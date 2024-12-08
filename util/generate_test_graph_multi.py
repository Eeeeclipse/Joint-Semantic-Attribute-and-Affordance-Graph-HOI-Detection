from PIL import Image
import os
import json
from clip_generate_attr_aff import generate_attr_aff, generate_human_feature
from tqdm import tqdm

dataset_path = './hico_20160224_det'
with open(os.path.join(dataset_path, "safe_test.json"), 'r') as file:
    safe_test = json.load(file)
with open(os.path.join(dataset_path, "test.json"), 'r') as file:
    test_anno = json.load(file)
with open(os.path.join(dataset_path, "action_list.json"), 'r') as file:
    action_list = json.load(file)
with open(os.path.join(dataset_path, "object_list.json"), 'r') as file:
    object_list = json.load(file)

graph = []
folder = 'images'
test_folder = 'test2015'
test_dir = os.path.join(dataset_path, folder)
test_dir = os.path.join(test_dir, test_folder)

for test in tqdm(safe_test):
    test_img = Image.open(os.path.join(test_dir, test))
    interactions = test_anno[test]
    interact_dict = dict()
    interact_dict['id'] = test
    key = list(interactions.keys())[0]
    interaction = interactions[key]
    bboxhuman = interaction['bboxhuman'][0]
    bboxobject = interaction['bboxobject'][0]
    obj_feature_dict = dict()
    human_feature_dict = dict()
    edge_feature_dict = dict()
    for hid, bboxhuman in enumerate(interaction['bboxhuman']):
        b_human = [bboxhuman[0], bboxhuman[2], bboxhuman[1], bboxhuman[3]]
        if hid not in human_feature_dict.keys():
            train_human = test_img.crop(b_human)
            human_feature_dict[hid] = generate_human_feature(train_human)
        for oid, bboxobject in enumerate(interaction['bboxobject']):
            b_object = [bboxobject[0], bboxobject[2], bboxobject[1], bboxobject[3]]
            if oid not in obj_feature_dict.keys():
                train_object = test_img.crop(b_object)
                obj_feature_dict[oid] = generate_attr_aff(train_object)
            edge_feature_dict[str(hid)+';'+str(oid)] = ((bboxhuman[0] + bboxhuman[1] - (bboxobject[0] + bboxobject[1])) / 2, 
                            (bboxhuman[2] + bboxhuman[3] - (bboxobject[2] + bboxobject[3])) / 2)
    interact_dict['obj_feature'] = obj_feature_dict
    interact_dict['human_feature'] = human_feature_dict
    interact_dict['edge_feature'] = edge_feature_dict
    all_label = []
    for k in interactions.keys():  
        interaction = interactions[k]
        act = interaction['act']
        obj = interaction['obj']
        act_label = action_list.index(act)
        obj_label = object_list.index(obj)
        edge_label = [act_label, obj_label]
        all_label.append(edge_label)
    interact_dict['labels'] = all_label
    graph.append(interact_dict) 
with open("./graph_test_multi.json", "w") as file: 
    json.dump(graph, file)