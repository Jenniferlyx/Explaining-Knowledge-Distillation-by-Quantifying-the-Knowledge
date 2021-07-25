import os
import sys
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
import random
import math


def _initialize_weights(self, seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


class vgg11_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg11_pretrained, self).__init__()
        self.net = torchvision.models.vgg11(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, x):
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x


class vgg11_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg11_without_pretrained, self).__init__()
        self.net = torchvision.models.vgg11(pretrained=False)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.features, seed)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, input):
        out = self.net(input)
        return out


class vgg16_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg16_pretrained, self).__init__()
        self.net = torchvision.models.vgg16(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, x):
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x


class vgg16_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg16_without_pretrained, self).__init__()
        self.net = torchvision.models.vgg16(pretrained=False)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net, seed)

    def forward(self, input):
        out = self.net(input)
        return out


class vgg19_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg19_pretrained, self).__init__()
        self.net = torchvision.models.vgg19(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, x):
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x


class vgg19_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg19_without_pretrained, self).__init__()
        self.net = torchvision.models.vgg19(pretrained=False)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.features, seed)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, input):
        out = self.net(input)
        return out


class vgg16_without_pretrained_bn(nn.Module):
    def __init__(self, out_planes, seed):
        super(vgg16_without_pretrained_bn, self).__init__()
        self.net = torchvision.models.vgg16_bn(pretrained=False)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.features, seed)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, input):
        out = self.net(input)
        return out




class alexnet_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(alexnet_pretrained, self).__init__()
        self.net = torchvision.models.alexnet(pretrained=True)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, x):
        x = self.net.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.classifier(x)
        return x


class alexnet_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(alexnet_without_pretrained, self).__init__()
        self.net = torchvision.models.alexnet(pretrained=False)
        self.net.classifier[6] = nn.Linear(4096, out_planes)
        _initialize_weights(self.net.features, seed)
        _initialize_weights(self.net.classifier, seed)

    def forward(self, input):
        out = self.net(input)
        return out




class ResNet18_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet18_pretrained, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 512
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),  # 512  4096
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_planes, bias=True)
        )
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet18_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet18_without_pretrained, self).__init__()
        self.net = torchvision.models.resnet18(pretrained=False)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 512
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 4096),  # 512  4096
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_planes, bias=True)
        )
        _initialize_weights(self.net, seed)
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet50_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet50_pretrained, self).__init__()
        self.net = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 2048
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # 2048  4096
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, out_planes, bias=True)
        )
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet50_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet50_without_pretrained, self).__init__()
        self.net = torchvision.models.resnet50(pretrained=False)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 2048
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # 2048  2048
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, out_planes, bias=True)
        )
        _initialize_weights(self.net, seed)
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet101_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet101_pretrained, self).__init__()
        self.net = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 2048
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # 2048  2048
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, out_planes, bias=True)
        )
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet101_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet101_without_pretrained, self).__init__()
        self.net = torchvision.models.resnet101(pretrained=False)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 2048
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # 2048  2048
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, out_planes, bias=True)
        )
        _initialize_weights(self.net, seed)
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet152_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet152_pretrained, self).__init__()
        self.net = torchvision.models.resnet152(pretrained=True)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 2048
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # 2048  2048
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, out_planes, bias=True)
        )
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class ResNet152_without_pretrained(nn.Module):
    def __init__(self, out_planes, seed):
        super(ResNet152_without_pretrained, self).__init__()
        self.net = torchvision.models.resnet152(pretrained=False)
        num_ftrs = self.net.fc.in_features
        self.features = nn.Sequential(
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),  # 2048
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_ftrs, num_ftrs, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_ftrs),
            nn.ReLU(inplace=True),
        )
        self.net.fc = nn.Sequential(
            nn.Linear(num_ftrs, 2048),  # 2048  2048
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048,2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, out_planes, bias=True)
        )
        _initialize_weights(self.net, seed)
        _initialize_weights(self.features, seed)
        _initialize_weights(self.net.fc, seed)

    def forward(self, x):
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.features(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return x


class student_network(nn.Module):
    def __init__(self, out_planes, seed):
        super(student_network, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),#(64,56,56)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), #(128,28,28)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256,14,14)  # (512,7,7)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_planes),
        )
        _initialize_weights(self.classifier, seed)
        _initialize_weights(self.features, seed)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out


class student_network_bn(nn.Module):
    def __init__(self, out_planes, seed):
        super(student_network_bn, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),#(64,56,56)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), #(128,28,28)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256,14,14)  # (512,7,7)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, out_planes),
        )
        _initialize_weights(self.classifier, seed)
        _initialize_weights(self.features, seed)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out


class student_network_no_drop(nn.Module):
    def __init__(self, out_planes, seed):
        super(student_network_no_drop, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),#(64,56,56)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2), #(128,28,28)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (256,14,14)  # (512,7,7)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, out_planes),
        )
        _initialize_weights(self.classifier, seed)
        _initialize_weights(self.features, seed)

    def forward(self, input):
        out = self.features(input)
        out = out.view(out.shape[0], -1)
        out = self.classifier(out)
        return out


class NoiseLayer(nn.Module):
    def __init__(self, sigma_init, image_size):
        super(NoiseLayer, self).__init__()
        self.sigma = torch.nn.Parameter(sigma_init)
        self.sigma_size = sigma_init.size()
        self.image_size = image_size

    def forward(self, input, unit_noise):
        # input is image, restrict the sigma minimum
        self.sigma.data.clamp_(0.001, 1)
        output = input + F.interpolate(self.sigma.expand((unit_noise.size()[0], self.sigma_size[1], self.sigma_size[2], self.sigma_size[3])), size=self.image_size, mode='nearest')*unit_noise
        sigma_vector = torch.log(self.sigma)
        penalty = torch.mean(sigma_vector) + 0.5 * torch.log(torch.tensor(2 * math.pi)) + torch.tensor(0.5)

        return output, penalty
