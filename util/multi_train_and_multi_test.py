import json 
import os
dataset_path = './hico_20160224_det'

# Build multi_train (at lease one person and one object)
with open(os.path.join(dataset_path, "train.json"), 'r') as file:
    train_dict = json.load(file)
multi_train = []
for img in train_dict.keys():
    actions = train_dict[img]
    append = True
    for a in actions.keys():
        bboxhuman = actions[a]['bboxhuman']
        bboxobject = actions[a]['bboxobject']
        if len(bboxhuman) == 0 or len(bboxobject) == 0:
            append = False
            break
        if len(bboxhuman) == 1 and len(bboxobject) == 1:
            append = False
            break
    if append:
        multi_train.append(img)
with open(os.path.join(dataset_path, "multi_train.json"), "w") as e_f: 
    json.dump(multi_train, e_f)

# Build multi_test (at lease one person and one object)
with open(os.path.join(dataset_path, "test.json"), 'r') as file:
    test_dict = json.load(file)
multi_test = []
for img in test_dict.keys():
    actions = test_dict[img]
    append = True
    for a in actions.keys():
        bboxhuman = actions[a]['bboxhuman']
        bboxobject = actions[a]['bboxobject']
        if len(bboxhuman) == 0 or len(bboxobject) == 0:
            append = False
            break
        if len(bboxhuman) == 1 and len(bboxobject) == 1:
            append = False
            break
    if append:
        multi_test.append(img)
with open(os.path.join(dataset_path, "multi_test.json"), "w") as e_f: 
    json.dump(multi_test, e_f)
