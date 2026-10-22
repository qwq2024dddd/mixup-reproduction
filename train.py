"""train.py — mixup reproduction on CIFAR-10 with PreAct ResNet-18.

Extracted verbatim from the Colab notebook that produced the reported
results (comments kept as in the original coursework code).
alpha = 0  -> ERM baseline;  alpha = 1.0 -> mixup (paper's CIFAR value).

Examples:
    python train.py --alpha 0   --save_dir runs/erm_s0
    python train.py --alpha 1.0 --save_dir runs/mixup_s0
"""
# 第 2 格：导入 + 随机种子 + 模型定义（PreAct ResNet-18，CIFAR 版本）
import csv, json, os, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out)

def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])

print('模型定义好了 ✅')

# 第 3 格：数据 + mixup + 训练/测试 + 一键实验函数
def get_dataloaders(data_dir, batch_size, num_workers=2):
    # 论文引用的标准 CIFAR 增强：随机裁剪（padding 4）+ 随机水平翻转 + 通道归一化
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=256, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def mixup_data(x, y, alpha):
    # 论文 Figure 1a 的做法：一个 batch 内随机打乱后两两混合
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)

def train_one_epoch(model, loader, criterion, optimizer, device, alpha):
    model.train()
    total_loss, n = 0.0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if alpha > 0:  # 实验组：mixup
            inputs, t_a, t_b, lam = mixup_data(inputs, targets, alpha)
            loss = mixup_criterion(criterion, model(inputs), t_a, t_b, lam)
        else:          # 对照组：普通 ERM 训练
            loss = criterion(model(inputs), targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        n += inputs.size(0)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        n += inputs.size(0)
    return total_loss / n, correct / n

def run_experiment(alpha, epochs, milestones, seed, save_dir, warmup_epochs=5):
    # 一组完整实验：alpha=0 是对照组（ERM），alpha=1.0 是实验组（mixup）
    # 带 lr warmup：lr=0.1 冷启动会在前几十步发散（实测 ERM 第 42 步 nan），
    # 前 warmup_epochs 个 epoch 把 lr 从小线性升到 0.1 后即稳定。两组同等处理，对照公平。
    seed_everything(seed)
    os.makedirs(save_dir, exist_ok=True)
    config = dict(alpha=alpha, epochs=epochs, milestones=milestones, seed=seed,
                  batch_size=128, lr=0.1, wd=1e-4, warmup_epochs=warmup_epochs)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(config, f, indent=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备：{device} | alpha={alpha} | epochs={epochs} | 结果存到 {save_dir}')
    trainloader, testloader = get_dataloaders('./data', 128)
    model = PreActResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    log_path = os.path.join(save_dir, 'log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'test_loss', 'test_acc', 'lr'])
    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # 本 epoch 学习率：warmup 阶段线性上升，之后在 milestones 处除以 10（论文规则）
        lr_now = 0.1
        if epoch <= warmup_epochs:
            lr_now = 0.1 * epoch / warmup_epochs
        else:
            for m in milestones:
                if epoch > m:
                    lr_now *= 0.1
        for g in optimizer.param_groups:
            g['lr'] = lr_now
        t0 = time.time()
        tr_loss = train_one_epoch(model, trainloader, criterion, optimizer, device, alpha)
        te_loss, te_acc = evaluate(model, testloader, criterion, device)
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f'{tr_loss:.4f}', f'{te_loss:.4f}', f'{te_acc:.4f}', f'{lr_now:.6f}'])
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
        print(f'[{epoch}/{epochs}] train_loss={tr_loss:.4f} test_loss={te_loss:.4f} '
              f'test_acc={te_acc * 100:.2f}% lr={lr_now}（{time.time() - t0:.0f}秒）')
    torch.save(model.state_dict(), os.path.join(save_dir, 'last.pth'))
    print(f'✅ 跑完了！最好 test_acc={best_acc * 100:.2f}%，日志在 {log_path}')
    return best_acc

print('训练函数准备好了 ✅')

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--milestones', type=int, nargs='+', default=[50, 75])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--save_dir', type=str, default='runs/erm_s0')
    ap.add_argument('--warmup_epochs', type=int, default=5)
    a = ap.parse_args()
    run_experiment(a.alpha, a.epochs, a.milestones, a.seed, a.save_dir, a.warmup_epochs)
