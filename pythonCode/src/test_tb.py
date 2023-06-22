import torch.utils.data
import cv2
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
# 数据预处理
img = cv2.imread("D:/test.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img = Image.fromarray(img).convert('RGB')
# img = Image.fromarray(img)默认就是RGB模式 可以不用转换


transforms = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])
img=transforms(img)

img=torch.reshape(img,(1,3,224,224))

img=img.cuda()
model = torchvision.models.resnet18(weights=None)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 2)
model_state=torch.load("model.pth")
model.load_state_dict(model_state)
model=model.cuda()
model.eval()
with torch.no_grad():
    output=model(img)
    print(output)
    output=F.softmax(output,dim=1)
    pred=output.argmax(1)
    print(output)
    print(pred)
