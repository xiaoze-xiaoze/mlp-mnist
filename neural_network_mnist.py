import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np

#下载训练集
training_data = datasets.MNIST(
    root="data",             # 数据集存放的路径
    train=True,              # 是否为训练集
    download=True,           # 是否下载
    transform=ToTensor(),    # 数据转换
)

# 下载测试集
test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True, 
    transform=ToTensor(),
)

# 显示样本
def show_samples(dataset, model, n_samples=10):
    fig, axes = plt.subplots(1, n_samples, figsize=(15,3))
    indices = np.random.choice(len(dataset), n_samples)
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        with torch.no_grad():
            pred = model(img.unsqueeze(0).to(device))
            pred_label = pred.argmax().item()
        
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"True: {label}\nPred: {pred_label}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

train_dataloader = DataLoader(training_data, batch_size=64)    # 训练集
test_dataloader = DataLoader(test_data, batch_size=64)         # 测试集

for x,y in train_dataloader:
    print(f"shape of x[N,C,H,W]:{x.shape}")    # 图像形状
    print(f"shape of y:{y.shape,y.dtype}")     # 标签形状和数据类型
    break

# 定义设备
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using {device} device")

# 构建神经网络
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
    
model = NeuralNetwork().to(device)
print(model)

# 记录训练和测试的loss
train_loss_history = []
test_loss_history = []

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    epoch_loss = 0
    
    batch_size_num = 1
    for x,y in dataloader:                  
        x,y = x.to(device), y.to(device)
        pred = model.forward(x)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_value = loss.item()
        epoch_loss += loss_value
        if batch_size_num % 100 == 0:
            print(f"loss:{loss_value:>7f}    [number:{batch_size_num}]")
        batch_size_num += 1
    
    # 记录每个epoch的平均loss
    avg_loss = epoch_loss / len(dataloader)
    train_loss_history.append(avg_loss)
    return avg_loss

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            x,y = x.to(device),y.to(device)
            pred = model.forward(x)
            test_loss += loss_fn(pred,y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    test_loss_history.append(test_loss)
    print(f"\nTest Error:\n")
    print(f"Accuracy: {(100*correct)}%")
    print(f"Avg loss: {test_loss}")
    show_samples(test_data, model, n_samples=10)
    return test_loss, correct

loss_fn = nn.CrossEntropyLoss()    # 定义损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0045)    # 定义优化器

if __name__ == "__main__":
    print(len(training_data))    # 训练集大小
    print(len(test_data))        # 测试集大小
    
    epochs = 25
    for i in range(epochs):
        print(f"\nEpoch {i+1}")
        train(train_dataloader, model, loss_fn, optimizer)

    test(test_dataloader, model, loss_fn)

    # 绘制训练loss曲线
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_history, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()