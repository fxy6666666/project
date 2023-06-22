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


# 设置超参数
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LR = 0.0000005
# 数据预处理
transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

# 读取训练集和验证集
dataset_train = ImageFolder('../dataset/train', transforms)
dataset_val = ImageFolder('../dataset/val', transforms)


# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

# 选择resnet18模型并且将最后一层全连接层的输出维度改为2
# model = torchvision.models.resnet18(weights=None)
# in_features = model.fc.in_features
# model.fc = nn.Linear(in_features, 2)
# model.to(DEVICE)
#加载训练最优的模型参数
model = torchvision.models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 2)
model_state=torch.load("model.pth")
print(model_state)
model.load_state_dict(model_state)
model.to(DEVICE)
#定义损失函数和选择优化器
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

# 训练模型
writer=SummaryWriter("../logs")
#记录验证集最好的ACC
val_best_acc = 0
for epoch in range(EPOCHS):
    # 训练过程
    model.train()
    train_sum_loss = 0
    train_total_correct=0
    train_total_num = len(dataset_train)
    for data in train_loader:
        imgs,targets=data
        imgs=imgs.to(DEVICE)
        targets=targets.to(DEVICE)
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_sum_loss += loss.data.item()
        train_pred = outputs.argmax(1)
        correct=(train_pred == targets).sum()
        train_total_correct += correct
    epoch_train_acc=train_total_correct/train_total_num
    print(train_total_correct)
    print('-----一轮训练结束-------')

    # 验证过程
    with torch.no_grad():
        model.eval()
        val_sum_loss = 0
        val_total_correct = 0
        val_total_num = len(dataset_val)
        for data in val_loader:
            imgs, targets = data
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            val_sum_loss += loss.data.item()
            val_pred=outputs.argmax(1)
            correct = (val_pred == targets).sum()
            val_total_correct += correct
        epoch_val_acc = val_total_correct / val_total_num
        print(val_total_correct)
        print('-----一轮验证结束-------')
        #保存在验证集上表现最好的参数
        if epoch_val_acc > val_best_acc:
            val_best_acc = epoch_val_acc
            torch.save(model.state_dict(), 'model.pth')
    # 绘制测试集和验证集每轮的loss
    writer.add_scalars('loss', {"train_loss": train_sum_loss,
                                "val_loss": val_sum_loss}, epoch + 151)
    # 绘制测试集和验证集每轮的acc
    writer.add_scalars('acc', {"train_acc": epoch_train_acc,
                               "val_ acc": epoch_val_acc}, epoch + 151)

writer.close()








