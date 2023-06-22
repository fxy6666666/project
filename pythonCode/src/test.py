import torch.cuda
import torch
import torch.utils.data.distributed
import torchvision.models
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import ImageFolder
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
model.cuda()
#定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()
# 测试过程
with torch.no_grad():
        model.eval()
        test_sum_loss = 0
        test_total_correct = 0
        test_total_num = len(dataset_test)
        for data in test_loader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            test_sum_loss += loss.data.item()
            test_pred=outputs.argmax(1)
            correct = (test_pred == targets).sum()
            test_total_correct += correct
        test_acc = test_total_correct / test_total_num
        print("测试集预测正确的图片数量是{}".format(test_total_correct.data.item()))
        print("测试集的准确率: {}".format(test_acc.data.item()))
        print('-----测试结束-------')