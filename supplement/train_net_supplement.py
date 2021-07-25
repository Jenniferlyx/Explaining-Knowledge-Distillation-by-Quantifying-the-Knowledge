import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'    # change GPU here
import sys
sys.path.extend(['./KD'])
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import shutil
import datetime as dt
import time
import argparse
from torch.utils.data import DataLoader
from model.models import *
from function.dataset import *
from function.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="vgg19")
parser.add_argument("--mode", type=str, default="_without_augment", help="please add '_' before the mode")
parser.add_argument("--date", type=str, default="_1020_supplement", help="please add '_' before the date")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument('--scheduler', default=False, type=bool)
parser.add_argument('--logspace', default=True, type=bool)
parser.add_argument("--lr_down", type=int, default=1, help="learning rate")
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=150, help="number of training iterations")
parser.add_argument('--dataset', default='ILSVRC', type=str)
parser.add_argument('--classes', default=12, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--weight_decay", type=float, default=5*1e-4, help="weight decay")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--gpu', default=0, type=int, help='Set it to 0. Change it above before importing torch.')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N')
parser.add_argument('--print-freq', '-p', default=30, type=int, metavar='N')
parser.add_argument('--save_per_epoch', default=False, type=bool)
parser.add_argument('--resume', default='', type=str, metavar='PATH')
args = parser.parse_args()

PATH = './KD/trained_model/' + str(args.dataset) + '_' + str(args.model) + str(args.mode) + str(args.date) + '/'
if not os.path.exists(PATH):
    os.mkdir(PATH)
print(vars(args))
print("Time: {}".format(dt.datetime.now()))
print("Python: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Pytorch: {}".format(torch.__version__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_network(PATH, args):
    if args.seed is not None:
        set_random(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    if args.dataset == 'CUB':
        traindir, valdir = './Dataset/CUB/train', './Dataset/CUB/val'
    elif args.dataset == 'VOC':
        traindir, valdir = './Dataset/VOCdevkit/train', './Dataset/VOCdevkit/val'
    elif args.dataset == 'ILSVRC':
        traindir, valdir = './Dataset/ILSVRC2013_mammal/train', './Dataset/ILSVRC2013_mammal/val'
    else:
        sys.exit("dataset error")

    if args.mode == '_without_augment':
        train_loader = torch.utils.data.DataLoader(SigmaDataSet(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
    elif args.mode == '_augment':
        train_loader = torch.utils.data.DataLoader(SupplementDataSet(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
    val_loader = torch.utils.data.DataLoader(SigmaDataSet(valdir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    if args.model == 'vgg11':
        model = vgg11_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'vgg16':
        model = vgg16_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'vgg19':
        model = vgg19_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'alexnet':
        model = alexnet_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'resnet18':
        model = ResNet18_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'resnet50':
        model = ResNet50_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'resnet101':
        model = ResNet101_without_pretrained(args.classes, seed=args.seed)
    elif args.model == 'resnet152':
        model = ResNet152_without_pretrained(args.classes, seed=args.seed)
    else:
        sys.exit('Model Name Error')

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    if args.logspace:
        logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.lr_down, args.epochs)
        print("lr logspace:", logspace_lr)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=50, verbose=True)

    train_logger_path = './KD/logs/{}_{}{}{}/train'.format(args.dataset, args.model, args.mode, args.date)
    val_logger_path = './KD/logs/{}_{}{}{}/val'.format(args.dataset, args.model, args.mode, args.date)
    if os.path.exists(train_logger_path):
        shutil.rmtree(train_logger_path)
    if os.path.exists(val_logger_path):
        shutil.rmtree(val_logger_path)
    logger_train = Logger(train_logger_path)
    logger_val = Logger(val_logger_path)

    best_acc1, acc1 = 0, 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    weight_diff = torch.zeros(4, args.epochs+1)       # conv fc2
    w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3 = get_weight(model)
    w_init_conv = torch.norm(w_origin_conv)
    w_init_fc1 = torch.norm(w_origin_fc1)
    w_init_fc2 = torch.norm(w_origin_fc2)
    w_init_fc3 = torch.norm(w_origin_fc3)
    torch.cuda.empty_cache()

    for epoch in range(args.start_epoch, args.epochs):
        # remember best acc@1 and save checkpoint
        best_acc1 = max(acc1, best_acc1)
        if args.save_per_epoch:
            save_dir = PATH + '{}.pth.tar'.format(epoch)
            save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, save_dir)

        if args.logspace:
            for param_group in optimizer.param_groups:
                param_group['lr'] = logspace_lr[epoch]
        lr = get_learning_rate(optimizer)
       

        weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3 = train(train_loader, model, criterion, optimizer, epoch, logger_train)
        weight_diff[0, epoch + 1], weight_diff[1, epoch + 1] = weight_distance_conv / w_init_conv, weight_distance_fc1 / w_init_fc1
        weight_diff[2, epoch + 1], weight_diff[3, epoch + 1] = weight_distance_fc2 / w_init_fc2, weight_distance_fc3 / w_init_fc3
       

        acc1, val_loss = validate(val_loader, model, criterion, epoch, logger_val)
        if args.scheduler:
            scheduler.step(acc1)

        weight_diff_save_path = PATH + 'weight_diff.npy'
        np.save(weight_diff_save_path, weight_diff.numpy())


    save_dir = PATH + '{}.pth.tar'.format(epoch+1)
    best_acc1 = max(acc1, best_acc1)
    save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'best_acc1': best_acc1, 'optimizer': optimizer.state_dict()}, save_dir)


def train(train_loader, model, criterion, optimizer, epoch, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3 = 0, 0, 0, 0
    end = time.time()
    for i, (id, img_name, input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3 = get_weight(model)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        w_curr_conv, w_curr_fc1, w_curr_fc2, w_curr_fc3 = get_weight(model)
        weight_distance_conv += torch.norm((w_curr_conv - w_origin_conv))
        weight_distance_fc1 += torch.norm((w_curr_fc1 - w_origin_fc1))
        weight_distance_fc2 += torch.norm((w_curr_fc2 - w_origin_fc2))
        weight_distance_fc3 += torch.norm((w_curr_fc3 - w_origin_fc3))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t''Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t''Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1,
                top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)
    return weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3


@torch.no_grad()        # deactivate autograd engine to reduce memory consumption and speed up computations
def validate(val_loader, model, criterion, epoch, logger):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (id, img_name, input, target) in enumerate(val_loader):
        input = input.to(device=device, non_blocking=True)
        target = target.to(device=device, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))
    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'Loss': losses.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)
    return top1.avg, losses.avg


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


# set for tensorboard
def set_tensorboard(log_dict, epoch, logger):

    info = log_dict
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.epoch_step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_error(pre, lbl):
    acc = torch.sum(pre == lbl).item() / len(pre)
    return 1 - acc


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def Pick_Sample(false_times, np_dir):
    length = len(false_times)
    false_seq = np.argsort(np.array(false_times))
    easy_id = false_seq[: int(length * 0.1)]
    hard_id = false_seq[length - int(length * 0.1):]
    np.save(np_dir + 'easy_id.npy', easy_id)
    np.save(np_dir + 'hard_id.npy', hard_id)


def get_learning_rate(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
          lr += [param_group['lr']]
    return lr


def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


def set_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def get_weight(model):

    if args.model[:3] == 'vgg':
        w_origin_fc1 = model.net.classifier[0].weight.clone().detach()
        w_origin_fc2 = model.net.classifier[3].weight.clone().detach()
        w_origin_fc3 = model.net.classifier[6].weight.clone().detach()
        if args.model == 'vgg11':
            w_origin_conv = model.net.features[18].weight.clone().detach()
        elif args.model == 'vgg16':
            w_origin_conv = model.net.features[28].weight.clone().detach()
        elif args.model == 'vgg19':
            w_origin_conv = model.net.features[34].weight.clone().detach()
        else:
            sys.exit('Model Name Error')
    elif args.model == 'alexnet':
        w_origin_conv = model.net.features[10].weight.clone().detach()
        w_origin_fc1 = model.net.classifier[1].weight.clone().detach()
        w_origin_fc2 = model.net.classifier[4].weight.clone().detach()
        w_origin_fc3 = model.net.classifier[6].weight.clone().detach()
    elif args.model[:6] == 'resnet':
        w_origin_conv = model.features[3].weight.clone().detach()
        w_origin_fc1 = model.net.fc[0].weight.clone().detach()
        w_origin_fc2 = model.net.fc[3].weight.clone().detach()
        w_origin_fc3 = model.net.fc[6].weight.clone().detach()
    else:
        sys.exit('Model Name Error')

    return w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3


train_network(PATH, args)






