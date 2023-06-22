import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])
dataset_test = ImageFolder('../dataset/test', transforms)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=True)
model = torchvision.models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model_state=torch.load("model.pth")
model.load_state_dict(model_state)
preds, labels = [], []
with torch.no_grad():
    model.eval()
    for data in test_loader:
        imgs, targets = data
        outputs = model(imgs)
        outputs=F.softmax(outputs,dim=1)
        test_pred = outputs[:,-1]
        preds.extend(test_pred)
        labels.extend(targets)
labels=np.array(labels)
preds=np.array(preds)
print(preds)
fpr, tpr, thresholds= roc_curve(labels, preds)
print(fpr)
print(tpr)
print(thresholds)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig("roc.jpg")
plt.show()
