"""
suggested lr:
vgg:                                                    resnet:
conv:
fc1: logsapce(-4, -5)/-4, epochs=150                    logsapce(1, -1), epochs=150
fc2: logsapce(-3, -4), epochs=150                       logsapce(-1, -2), epochs=150
fc3: logsapce(-3, -4)/-3, epochs=150                    -1/-2, epochs=150

CUB: ./Dataset/CUB/                          200
VOC: ./Dataset/VOCdevkit/                    20
ILSVRC: ./Dataset/ILSVRC2013_mammal/         12
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # change GPU here
import sys
sys.path.extend(['./KD'])
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import datetime as dt
import shutil
import argparse
from model.models import *
from torch.utils.data import DataLoader
from function.dataset import *
from function.logger import Logger

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="fc1") ## fc1 fc2 fc3 ##
parser.add_argument("--model", type=str, default="vgg16") ## vgg11 vgg16 vgg19 resnet50 resnet101 resnet152 ##
parser.add_argument("--distilation_train", type=bool, default=True)
parser.add_argument("--classifier_train", type=bool, default=True)
parser.add_argument("--date", type=str, default="_")
parser.add_argument("--lr", type=float, default=1, help="learning rate")
parser.add_argument("--lr_down", type=int, default=1, help="learning rate")
parser.add_argument('--logspace', default=False, type=bool)
parser.add_argument('--teacher_checkpoint', default='./', type=str, metavar='PATH')
parser.add_argument('--normalization', default=False, type=bool, help='whether to normalize the output of teacher')# no use
parser.add_argument('--alpha', default=1, type=float)   # adjust mse loss ##
parser.add_argument("--batch_size", type=int, default=64, help="batch size")
parser.add_argument("--epochs", type=int, default=150, help="number of distillation training iterations")
parser.add_argument("--classify_epochs", type=int, default=100, help="number of classifier training iterations")
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument('--dataset', default='CUB', type=str)
parser.add_argument('--classes', default=200, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument("--weight_decay", type=float, default=1e-4, help="weight decay")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
parser.add_argument('--gpu', default=0, type=int, help='Set it to 0. Change it above before importing torch.')
parser.add_argument('--resume', default='', type=str, metavar='PATH')
parser.add_argument('--print-freq', '-p', default=20, type=int, metavar='N')
parser.add_argument('--save_per_epoch', default=True, type=bool)
parser.add_argument('--save_best_acc', default=False, type=bool)
args = parser.parse_args()

## get the path for project ##
path = './KD/'
print(vars(args))
print("Time: {}".format(dt.datetime.now()))
print("Python: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Pytorch: {}".format(torch.__version__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def distillation(path,args):
    if args.seed is not None:
        set_random(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    ## get dataset path ##
    if args.dataset == 'CUB':
        # dataset_root = path +'dataset/{}'.format(args.dataset)
        traindir, valdir = './Dataset/CUB/train', './Dataset/CUB/val'
    elif args.dataset == 'VOC':
        traindir, valdir = './Dataset/VOCdevkit/train', './Dataset/VOCdevkit/val'
    elif args.dataset == 'ILSVRC':
        traindir, valdir = './Dataset/ILSVRC2013_mammal/train', './Dataset/ILSVRC2013_mammal/val'
    else:
        sys.exit("dataset error")

    ## prepare dataloader ##
    if args.dataset == 'CUB':
        train_loader = torch.utils.data.DataLoader(SigmaDataSet(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
    elif args.dataset == 'VOC' or args.dataset == 'ILSVRC':
        train_loader = torch.utils.data.DataLoader(AugmentDataSet(traindir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)
    else:
        sys.exit("Dataset Error")
    val_loader = torch.utils.data.DataLoader(SigmaDataSet(valdir, model=args.model), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    if args.distilation_train:
        ## load teacher checkpoint ##
        if args.model == 'vgg11':
            teacher_model = vgg11_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = vgg11_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        elif args.model == 'vgg16':
            teacher_model = vgg16_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = vgg16_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        elif args.model == 'vgg19':
            teacher_model = vgg19_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = vgg19_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        elif args.model == 'alexnet':
            teacher_model = alexnet_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = alexnet_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        elif args.model == 'resnet50':
            teacher_model = ResNet50_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = ResNet50_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        elif args.model == 'resnet101':
            teacher_model = ResNet101_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = ResNet101_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        elif args.model == 'resnet152':
            teacher_model = ResNet152_pretrained(out_planes=args.classes, seed=args.seed).to(device)
            model = ResNet152_without_pretrained(out_planes=args.classes, seed=args.seed).to(device)
        else:
            sys.exit('Teacher Model Name Error')
        # fetch teacher outputs using teacher_model under eval() mode ##
        load_checkpoint(args.teacher_checkpoint, teacher_model)
        teacher_model.eval()

        print("get teacher train")
        teacher_train = fetch_teacher_outputs(teacher_model=teacher_model, dataloader=train_loader, model=args.model, mode=args.mode, classes=args.classes)
        print("\nget teacher val")
        teacher_val = fetch_teacher_outputs(teacher_model=teacher_model, dataloader=val_loader, model=args.model, mode=args.mode, classes=args.classes)


        ## choose optimizer ##
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        if args.logspace:
            logspace_lr = np.logspace(np.log10(args.lr), np.log10(args.lr) - args.lr_down, args.epochs)
            print("lr logspace:", logspace_lr)

        ## resume to load for further training##
        if args.resume:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

        train_mse_logger_path = path + 'logs/{}_{}_distil_{}_{}/train_mse'.format(args.dataset, args.model, args.mode, args.date)
        val_mse_logger_path = path + 'logs/{}_{}_distil_{}_{}/val_mse'.format(args.dataset, args.model, args.mode, args.date)
        if os.path.exists(train_mse_logger_path):
            shutil.rmtree(train_mse_logger_path)
        if os.path.exists(val_mse_logger_path):
            shutil.rmtree(val_mse_logger_path)
        logger_train_mse = Logger(train_mse_logger_path)
        logger_val_mse = Logger(val_mse_logger_path)
        save_path = path + 'trained_model/' + '{}_{}_distil_{}_{}/'.format(args.dataset, args.model, args.mode, args.date)
        check_dir(save_path)

        val_acc = 0
        weight_diff = torch.zeros(4, args.epochs+1)       # conv fc1 fc2 fc3
        w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3 = get_weight(model)
        w_init_conv = torch.norm(w_origin_conv)
        w_init_fc1 = torch.norm(w_origin_fc1)
        w_init_fc2 = torch.norm(w_origin_fc2)
        w_init_fc3 = torch.norm(w_origin_fc3)
        torch.cuda.empty_cache()

        print('----------- distillation is ready -----------')
        for epoch in range(args.start_epoch, args.epochs):
            if epoch <= 10 or epoch % 5 == 0:
                save_dir = save_path + '{}.pth.tar'.format(epoch)
                save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(), 'best_acc': val_acc, 'optimizer': optimizer.state_dict()}, save_dir)

            if args.logspace:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = logspace_lr[epoch]
            

            weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3 = train_kd(model, optimizer, loss_fn_kd, train_loader, epoch, logger_train_mse, teacher_model, mode=args.mode)
            weight_diff[0, epoch + 1], weight_diff[1, epoch + 1] = weight_distance_conv / w_init_conv, weight_distance_fc1 / w_init_fc1
            weight_diff[2, epoch + 1], weight_diff[3, epoch + 1] = weight_distance_fc2 / w_init_fc2, weight_distance_fc3 / w_init_fc3
           

            val_loss, val_acc = evaluate_kd(model, val_loader, epoch, logger_val_mse, teacher_model, mode=args.mode)
            torch.cuda.empty_cache()

            weight_diff_save_path = save_path + 'weight_diff.npy'.format(args.dataset, args.mode, args.epochs)
            np.save(weight_diff_save_path, weight_diff.numpy())
        

        if args.save_per_epoch:
            save_dir = save_path + '{}.pth.tar'.format(epoch+1)
        else:
            save_dir = save_path + '{}_{}_{}.pth.tar'.format(args.dataset, args.mode, args.epochs)
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'best_acc': val_acc, 'optimizer': optimizer.state_dict()}, save_dir)

    

    ## fine-tune model to classify ##
    print('------------fine-tune model to classify is ready!------------\n')
    if args.classifier_train and args.mode != 'fc3':
        if args.model[:3] == 'vgg':
            if args.mode == 'conv':
                for param in model.net.features.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.classifier.parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc1':
                for param in model.net.features.parameters():
                    param.requires_grad = False
                for param in model.net.classifier[:1].parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.classifier[1:].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc2':
                for param in model.net.features.parameters():
                    param.requires_grad = False
                for param in model.net.classifier[:-1].parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.classifier[-1].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                sys.exit('Layer Name Error')
        elif args.model == 'alexnet':
            if args.mode == 'conv':
                for param in model.net.features.parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.classifier.parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc1':
                for param in model.net.features.parameters():
                    param.requires_grad = False
                for param in model.net.classifier[:2].parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.classifier[2:].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc2':
                for param in model.net.features.parameters():
                    param.requires_grad = False
                for param in model.net.classifier[:-1].parameters():
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.classifier[-1].parameters(), 1e-3, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                sys.exit('Layer Name Error')
        elif args.model[:6] == 'resnet':
            if args.mode == 'conv':
                large_lr_layers = list(map(id, model.net.fc.parameters()))
                freeze_layers = list(filter(lambda p: id(p) not in large_lr_layers, model.parameters()))
                for param in freeze_layers:
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.fc.parameters(), 1e-2, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc1':
                large_lr_layers = list(map(id, model.net.fc[1:].parameters()))
                freeze_layers = list(filter(lambda p: id(p) not in large_lr_layers, model.parameters()))
                for param in freeze_layers:
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.fc[1:].parameters(), 1e-2, momentum=args.momentum, weight_decay=args.weight_decay)
            elif args.mode == 'fc2':
                large_lr_layers = list(map(id, model.net.fc[-1].parameters()))
                freeze_layers = list(filter(lambda p: id(p) not in large_lr_layers, model.parameters()))
                for param in freeze_layers:
                    param.requires_grad = False
                optimizer = torch.optim.SGD(model.net.fc[-1].parameters(), 1e-2, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                sys.exit('Layer Name Error')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=10, verbose=False) # you can change other lr settings

        train_ce_logger_path = path + 'logs/{}_{}_distil_{}_{}/train_ce'.format(args.dataset, args.model, args.mode, args.date)
        val_ce_logger_path = path + 'logs/{}_{}_distil_{}_{}/val_ce'.format(args.dataset, args.model, args.mode, args.date)
        if os.path.exists(train_ce_logger_path):
            shutil.rmtree(train_ce_logger_path)
        if os.path.exists(val_ce_logger_path):
            shutil.rmtree(val_ce_logger_path)
        logger_train_ce = Logger(train_ce_logger_path)
        logger_val_ce = Logger(val_ce_logger_path)

        for epoch in range(args.epochs, args.epochs + args.classify_epochs):

            print(optimizer.param_groups[0]['lr'])
            train(model, optimizer, train_loader, epoch, logger_train_ce)

            val_loss, val_acc = evaluate(model, val_loader, epoch, logger_val_ce)
            scheduler.step(val_loss)


def train_kd(model, optimizer, loss_fn_kd, dataloader, epoch, logger, teacher_model, mode):
    # set model to training mode
    model.train()
    teacher_model.eval()
    MSE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3 = 0, 0, 0, 0

    for i, (img_idx, img_name, train_batch, labels_batch) in enumerate(dataloader):

        train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
        with torch.no_grad():
            output_batch = model(train_batch)
            output_teacher = get_feature(teacher_model, mode, train_batch)

        output_layer = get_feature(model, mode, train_batch)
        mse = loss_fn_kd(output_layer, output_teacher)
        w_origin_conv, w_origin_fc1, w_origin_fc2, w_origin_fc3 = get_weight(model)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        mse.backward()

        # performs updates using calculated gradients
        optimizer.step()

        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))
        # update the average loss
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))
        MSE.update(mse.item(), labels_batch.size(0))

        w_curr_conv, w_curr_fc1, w_curr_fc2, w_curr_fc3 = get_weight(model)
        weight_distance_conv += torch.norm((w_curr_conv - w_origin_conv))
        weight_distance_fc1 += torch.norm((w_curr_fc1 - w_origin_fc1))
        weight_distance_fc2 += torch.norm((w_curr_fc2 - w_origin_fc2))
        weight_distance_fc3 += torch.norm((w_curr_fc3 - w_origin_fc3))
        torch.cuda.empty_cache()

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t''MSE {MSE.val:.4f} ({MSE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(dataloader), MSE=MSE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'MSE': MSE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)
    return weight_distance_conv, weight_distance_fc1, weight_distance_fc2, weight_distance_fc3


@torch.no_grad()        # deactivate autograd engine to reduce memory consumption and speed up computations
def evaluate_kd(model, dataloader, epoch, logger, teacher_model, mode):

    model.eval()
    teacher_model.eval()
    # summary for current eval loop
    MSE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (img_idx, img_name, data_batch, labels_batch) in enumerate(dataloader):

        data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
        output_batch = model(data_batch)

        # compute model output, fetch teacher output, and compute KD loss
        output_layer = get_feature(model, mode, data_batch)
        output_teacher = get_feature(teacher_model, mode, data_batch)
        mse = loss_fn_kd(output_layer, output_teacher)

        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))
        MSE.update(mse.item(), labels_batch.size(0))
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t''MSE {MSE.val:.4f} ({MSE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(dataloader), MSE=MSE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'MSE': MSE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)

    return MSE.avg, top1.avg.item()


def train(model, optimizer, dataloader, epoch, logger):
    # set model to training mode
    model.train()
    CE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (img_idx, img_name, train_batch, labels_batch) in enumerate(dataloader):

        train_batch, labels_batch = train_batch.to(device), labels_batch.to(device)
        output_batch = model(train_batch)

        ## compute loss ##
        ce = nn.CrossEntropyLoss()(output_batch, labels_batch)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        ce.backward()

        # performs updates using calculated gradients
        optimizer.step()
        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))

        # update the average loss
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))
        CE.update(ce.item(), labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Epoch: [{0}][{1}/{2}]\t''CE {CE.val:.4f} ({CE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, i, len(dataloader), CE=CE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'CE': CE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)


@torch.no_grad()        # deactivate autograd engine to reduce memory consumption and speed up computations
def evaluate(model, dataloader, epoch, logger):

    model.eval()
    # summary for current eval loop
    CE = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    for i, (img_idx, img_name, data_batch, labels_batch) in enumerate(dataloader):

        data_batch, labels_batch = data_batch.to(device), labels_batch.to(device)
        output_batch = model(data_batch)

        ce = nn.CrossEntropyLoss()(output_batch, labels_batch)

        acc1, acc5 = accuracy(output_batch, labels_batch, topk=(1, 5))
        CE.update(ce.item(), labels_batch.size(0))
        top1.update(acc1[0], labels_batch.size(0))
        top5.update(acc5[0], labels_batch.size(0))

        if i % args.print_freq == 0:
            print(
                'Test: [{0}/{1}]\t''CE {CE.val:.4f} ({CE.avg:.4f})\t''Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t''Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(dataloader), CE=CE, top1=top1, top5=top5))
    print('Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    log_dict = {'CE': CE.avg, 'top1_prec': top1.avg.item(), 'top5_prec': top5.avg.item()}
    set_tensorboard(log_dict, epoch, logger)

    return CE.avg, top1.avg.item()


def loss_fn_kd(output, output_teacher):

    if args.normalization:
        alpha = args.alpha
        mse_loss = nn.MSELoss()(output, alpha * output_teacher.float())
    else:
        mse_loss = nn.MSELoss()(output, output_teacher.float())
    return mse_loss


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer=None):

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint, map_location='cuda:0')
    else:
        # this helps avoid errors when loading single-GPU-trained weights onto CPU-model
        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])


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


def set_tensorboard(log_dict, epoch, logger):
    # set for tensorboard
    info = log_dict
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)
    return


def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def set_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


@torch.no_grad()        # deactivate autograd engine to reduce memory consumption and speed up computations
def fetch_teacher_outputs(teacher_model, dataloader, model, mode, classes):
    # set teacher_model to evaluation mode
    teacher_model.eval()

    if model[:3] == 'vgg':
        if mode == 'conv':
            teacher_conv = np.zeros((dataloader.dataset.size, 512, 14, 14))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                if model[:5] == 'vgg11':
                    output_teacher_conv = teacher_model.net.features[:19](data_batch)
                elif model[:5] == 'vgg16':
                    output_teacher_conv = teacher_model.net.features[:29](data_batch)
                elif model[:5] == 'vgg19':
                    output_teacher_conv = teacher_model.net.features[:35](data_batch)
                teacher_conv[idx] = output_teacher_conv.detach().cpu().numpy()
            return teacher_conv

        elif mode == 'fc1':
            teacher_fc = np.zeros((dataloader.dataset.size, 4096))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_fc = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_fc.detach().cpu().numpy()
            return teacher_fc

        elif mode == 'fc2':
            teacher_fc = np.zeros((dataloader.dataset.size, 4096))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_fc = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_fc.detach().cpu().numpy()
            return teacher_fc

        elif mode == 'fc3':
            teacher_fc = np.zeros((dataloader.dataset.size, classes))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_fc = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_fc.detach().cpu().numpy()
            return teacher_fc
        else:
            sys.exit('Layer Name Error')

    elif model[:6] == 'resnet':
        if mode == 'conv':
            teacher_conv = np.zeros((dataloader.dataset.size, 2048, 7, 7))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_batch = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_conv[idx] = output_teacher_batch.detach().cpu().numpy()
            return teacher_conv

        elif mode == 'fc1':
            teacher_fc = np.zeros((dataloader.dataset.size, 2048))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_batch = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_batch.detach().cpu().numpy()
            return teacher_fc

        elif mode == 'fc2':
            teacher_fc = np.zeros((dataloader.dataset.size, 2048))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_batch = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_batch.detach().cpu().numpy()
            return teacher_fc

        elif mode == 'fc3':
            teacher_fc = np.zeros((dataloader.dataset.size, classes))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_batch = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_batch.detach().cpu().numpy()
            return teacher_fc

    elif model == 'alexnet':
        if mode == 'conv':
            teacher_conv = np.zeros((dataloader.dataset.size, 256, 13, 13))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_conv = teacher_model.net.features[:11](data_batch)
                teacher_conv[idx] = output_teacher_conv.detach().cpu().numpy()
            return teacher_conv

        elif mode == 'fc1':
            teacher_fc = np.zeros((dataloader.dataset.size, 4096))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_fc = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_fc.detach().cpu().numpy()
            return teacher_fc

        elif mode == 'fc2':
            teacher_fc = np.zeros((dataloader.dataset.size, 4096))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_fc = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_fc.detach().cpu().numpy()
            return teacher_fc

        elif mode == 'fc3':
            teacher_fc = np.zeros((dataloader.dataset.size, classes))
            for i, (idx, img_name, data_batch, labels_batch) in enumerate(dataloader):
                print(i, idx)
                data_batch = data_batch.to(device)
                output_teacher_fc = get_feature(model=teacher_model, mode=mode, data_batch=data_batch)
                teacher_fc[idx] = output_teacher_fc.detach().cpu().numpy()
            return teacher_fc
        else:
            sys.exit('Layer Name Error')
    else:
        sys.exit('Model Name Error')


def get_feature(model, mode, data_batch):
    if mode == 'conv':
        if args.model == 'vgg11':
            output_layer = model.net.features[:19](data_batch)
        elif args.model == 'vgg16':
            output_layer = model.net.features[:29](data_batch)
        elif args.model == 'vgg19':
            output_layer = model.net.features[:35](data_batch)
        elif args.model == 'alexnet':
            output_layer = model.net.features[:11](data_batch)
        elif args.model[:6] == 'resnet':
            pre = model.net.layer4(model.net.layer3(model.net.layer2(model.net.layer1(
                model.net.maxpool(model.net.relu(model.net.bn1(model.net.conv1(data_batch))))))))
            output_layer = model.features[:5](pre)

    elif mode == 'fc1':
        if args.model[:3] == 'vgg':
            pre = model.net.features(data_batch)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.classifier[:1](pre)
        elif args.model == 'alexnet':
            pre = model.net.features(data_batch)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.classifier[:2](pre)
        elif args.model[:6] == 'resnet':
            pre = model.net.layer4(model.net.layer3(model.net.layer2(model.net.layer1(
                model.net.maxpool(model.net.relu(model.net.bn1(model.net.conv1(data_batch))))))))
            pre = model.features(pre)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.fc[:1](pre)

    elif mode == 'fc2':
        if args.model[:3] == 'vgg':
            pre = model.net.features(data_batch)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.classifier[:4](pre)
        elif args.model == 'alexnet':
            pre = model.net.features(data_batch)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.classifier[:5](pre)
        elif args.model[:6] == 'resnet':
            pre = model.net.layer4(model.net.layer3(model.net.layer2(model.net.layer1(
                model.net.maxpool(model.net.relu(model.net.bn1(model.net.conv1(data_batch))))))))
            pre = model.features(pre)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.fc[:4](pre)

    elif mode == 'fc3':
        if args.model[:3] == 'vgg':
            pre = model.net.features(data_batch)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.classifier(pre)
        elif args.model == 'alexnet':
            pre = model.net.features(data_batch)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.classifier(pre)
        elif args.model[:6] == 'resnet':
            pre = model.net.layer4(model.net.layer3(model.net.layer2(model.net.layer1(
                model.net.maxpool(model.net.relu(model.net.bn1(model.net.conv1(data_batch))))))))
            pre = model.features(pre)
            pre = model.net.avgpool(pre)
            pre = torch.flatten(pre, 1)
            output_layer = model.net.fc(pre)
    else:
        sys.exit('Layer Name Error')

    return output_layer


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


distillation(path, args)
