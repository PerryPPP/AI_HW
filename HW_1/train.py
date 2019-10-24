import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import visdom
from ResNet import ResNet18
from torch.utils.data import DataLoader
import numpy as np

# GPU/CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# params
EPOCH = 200
pre_epoch = 0
BATCH_SIZE = 128
LR = 0.001


# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')
parser.add_argument('--net', default='./model/Resnet18.pth', help="path to net (to continue training)")
args = parser.parse_args()


# prepare the data
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
    )
trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
    )

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=False,
    transform=transform_test
    )
testloader = DataLoader(
    testset,
    batch_size=100,
    shuffle=False,
    num_workers=2
    )
# label for Cifar-10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# build the model : ResNet
net = ResNet18().to(device)

# criterion & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)

# 训练
if __name__ == "__main__":
    viz = visdom.Visdom(env='train-CIFAR10')
    viz.image(torchvision.utils.make_grid(next(iter(trainloader))[0], nrow=8), win='train-image')
    best_acc = 85
    print("Start Training the model : Resnet-18  Dataset: Cifar-10")  # 定义遍历数据集的次数
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            global_step = 0
            batch_step = 0
            for epoch in range(pre_epoch, EPOCH):
                iter_count = 0
                global_step += 1
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                tr_correct = 0.0
                tr_total = 0.0
                for i, data in enumerate(trainloader, 0):
                    batch_step += 1
                    iter_count += 1
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    tr_total += labels.size(0)
                    tr_correct += predicted.eq(labels.data).cpu().sum()
                    tr_acc = 100. * tr_correct / tr_total
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), tr_acc))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), tr_acc))
                    f2.write('\n')
                    f2.flush()
                    viz.line(Y=np.array([sum_loss / iter_count]), X=np.array([batch_step]), update='replace' if batch_step == 1 else 'append', win="loss_win")

                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    ts_correct = 0
                    ts_total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        ts_total += labels.size(0)
                        ts_correct += (predicted == labels).sum()
                    ts_acc = 100. * ts_correct /ts_total
                    print('测试分类准确率为：%.3f%%' % (ts_acc))
                    # 将每次测试结果实时写入acc.txt文件中
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d.pth' % (args.outf, epoch + 1))
                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, ts_acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if ts_acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, ts_acc))
                        f3.close()
                        best_acc = ts_acc

                    viz.line(Y=np.column_stack((np.array([tr_acc.item()]), np.array([ts_acc.item()]))),
                             X=np.column_stack((np.array([global_step]), np.array([global_step]))),
                             win="acc_win", update='replace' if epoch == 0 else 'append',
                             opts=dict(legned=['Train_acc', 'Val_acc']))


