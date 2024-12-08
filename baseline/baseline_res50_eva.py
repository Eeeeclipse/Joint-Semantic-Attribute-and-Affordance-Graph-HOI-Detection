import torch
import torchvision
from baseline_res50_data import load_datasets
from tqdm import tqdm

device = 'cuda'
weight = "./resnet50_hico_det.pth"
batch_size = 24
model = torchvision.models.resnet50(pretrained=False)
model.fc = torch.nn.Linear(in_features=2048, out_features=600, bias=True)
r50_weight = torch.load(weight)
model.load_state_dict(r50_weight)
model.to(device)
model.eval()

count = 0
total = 0
test_loader = load_datasets('train', 24)
for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    outputs = model(inputs)
    pred = torch.argmax(outputs, dim=1).cpu()
    gt = torch.argmax(labels, dim=1)
    count += torch.sum(pred == gt)
    total += len(inputs)

    
print('-' * 80)
print('Naive baseline ResNet50(unpretrained)')
print('Accuracy:')
print(count / total)
print('-' * 80)
