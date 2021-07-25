"""
suggested lambda:
vgg/alexnet:                             resnet:
conv:lambda = 7                         lambda = 6
fc1: lambda = 7                         lambda = 6
fc2: lambda = 7                         lambda = 6
fc3: lambda = 7                         lambda = 6
batch size as large as possible
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'    # change GPU here
import sys
sys.path.extend(['./KD'])
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import math
import torch.nn.functional as F
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import datetime as dt
from model.models import *
from function.dataset import *


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="alexnet")
parser.add_argument("--mode", type=str, default="label_net") ## teacher or label net or distil net  ##
parser.add_argument("--date", type=str, default="0415")
parser.add_argument("--capacity_layer", type=str, default='conv', help=" conv, fc1, fc2, fc3, all, distil_all")
parser.add_argument('--checkpoint_root', default='./KD/trained_model/VOC_alexnet_without_pretrain_1013/', type=str)
parser.add_argument('--checkpoint_step', default=3, type=int) ## the step for checkpoint ##
parser.add_argument('--dataset', default='VOC', type=str)
parser.add_argument('--classes', default=20, type=int)
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--batch_size", type=int, default=96, help="batch size, as large as possible")
parser.add_argument('--epoch', type=int, default=80, help='number of training iterations for sigma')
parser.add_argument('--epoch_break', type=int, default=20, help='number of training iterations for sigma')
parser.add_argument('--threshold_entropy', type=int, default=-2.8, help='threshold of entropy loss for selecting sigma')
parser.add_argument('--threshold_feature', type=int, default=1.5, help='threshold of feature loss for selecting sigma')
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--gpu', default=0, type=int, help='Set it to 0. Change it above before importing torch.')
parser.add_argument('--print_freq', '-p', default=5, type=int, metavar='N')
parser.add_argument('--lambda_init', type=float, default=6, help='lambda can be changed.')
parser.add_argument('--sigma_init_decay', type=float, default=0.01, help='the initialization of sigma is setting same value of all sigma')
parser.add_argument("--image_size", type=int, default=224)
parser.add_argument('--sigma_size', type=int, default=16, help='if the size of image is 224 and the size of sigma is 16, then 1 sigma have 14*14 pixels.')
parser.add_argument('--sigma_upsample_size', type=int, default=18, help='')
parser.add_argument('--fore_expand', default=False, type=bool, help='whether to expand the foreground')
parser.add_argument('--lambda_change_ratio', type=float, default=0.5, help='')
parser.add_argument('--pic_num', type=int, default=500, help='number of imgs to train sigma')
parser.add_argument('--pic_seed', type=int, default=0, help='set to same seed')
parser.add_argument('--pick_sample', default=True, type=bool)
parser.add_argument('--effect_num', type=int, default=50, help='')
parser.add_argument('--Top10', default=False, type=bool)
parser.add_argument('--sigma_iteration', default=1, type=int)
args = parser.parse_args()

## get the path for project ##
path = './KD/'
print(vars(args))
args.seed = 0               # seed set to same number
print('seed:', args.seed)
print("Time: {}".format(dt.datetime.now()))
print("Python: {}".format(sys.version))
print("Numpy: {}".format(np.__version__))
print("Pytorch: {}".format(torch.__version__))


def train_sigma(path, args, capacity_layer, checkpoint_root):
    if args.seed is not None:
        set_random(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    ## get dataset path ##
    if args.dataset == 'CUB':
        traindir, segdir = './Dataset/CUB/train', './Dataset/CUB/seg/train'
    elif args.dataset == 'VOC':
        traindir, segdir = './Dataset/VOCdevkit/seg/train/origin', './Dataset/VOCdevkit/seg/train/seg'
    elif args.dataset == 'ILSVRC':
        traindir = './Dataset/ILSVRC2013_mammal/train'
    else:
        sys.exit("dataset error")
    ## define loss function ##
    criterion = torch.nn.MSELoss()
    relu_method = nn.ReLU()
    if args.model == 'alexnet':
        args.image_size = 227
        transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
    else:
        args.image_size = 224
        transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    ## define model ##
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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if os.path.exists(checkpoint_root + '400.pth.tar'):
        checkpoint_list = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 130, 150, 170, 190, 240, 280, 320, 360, 400]
    else:
        checkpoint_list = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 130, 150]
    model_epoch = len(checkpoint_list)

    #train a sigma matrix for every checkpoint of every image of every layer
    lambda_init = args.lambda_init
    result_root = os.path.join(path, 'sigma_result')
    result_path = result_root + '/' + str(args.dataset) + '_' + str(args.model) + '_' + str(args.mode) + '_' + str(capacity_layer) + '_' + str(args.date) + '/'
    checkdir(result_root)
    checkdir(result_path)

    Select_DataSet = SelectDataSet(root=traindir, model=args.model, sample_num=args.pic_num, seed=args.pic_seed)
    name_path = result_path + '/full_name_list.txt'
    fw = open(name_path, 'w')
    for name in Select_DataSet.name_list:
        fw.write(name)
        fw.write("\n")
    fw.close()
    dataloader = torch.utils.data.DataLoader(Select_DataSet, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=_init_fn)

    effect_count, non_effect_count = 0, 0
    ## pic --> checkpoint ##
    ## each img has one folder ##
    for batch_id, (id, img_name, image, label) in enumerate(dataloader):
        if effect_count >= args.effect_num:
            break

        image = image.to(device)
        ## define path ##
        result_folder = os.path.join(result_path, 'img_' + str(id.data.cpu().numpy()[0]) + '_non_effect')
        plot_path = os.path.join(result_folder,'plot')
        npy_save_path = os.path.join(result_folder,'npy_files') ## to save feature_loss,penalty_loss,ect.##
        checkdir(result_folder), checkdir(plot_path), checkdir(npy_save_path)

        ## save origin img and prepare npy_files to save the sigma nearest to the threshold ##
        get_origin_img(traindir, img_name, args.image_size, result_folder)
        feature_loss_matrix, penalty_loss_matrix, entropy_matrix = np.zeros((model_epoch, args.sigma_iteration, args.epoch)), np.zeros((model_epoch, args.sigma_iteration, args.epoch)), np.zeros((model_epoch, args.sigma_iteration, args.epoch))
        sigma_matrix_entropy = np.zeros((model_epoch, args.sigma_iteration, args.sigma_size, args.sigma_size))
        sigma_f_matrix = np.zeros((model_epoch, args.sigma_iteration))

        # get seg, down-sample it to a bigger size than sigma size and get 16*16 size
        if args.dataset == 'CUB' or args.dataset == 'VOC':
            if args.dataset == 'CUB':
                seg_img = transform(Image.open(segdir + '/' + img_name[0].replace('jpg', 'png')).convert('L')).squeeze().to(device)
            if args.dataset == 'VOC':
                seg_orig = np.array(Image.open(segdir + '/' + img_name[0].replace('jpg', 'png')).convert('L'))
                seg_img = np.zeros(shape=(int(seg_orig.shape[0]*1.2), int(seg_orig.shape[1]*1.2)))
                a = int((seg_img.shape[0] - seg_orig.shape[0]) / 2)
                b = int((seg_img.shape[1] - seg_orig.shape[1]) / 2)
                seg_img[a: seg_orig.shape[0]+a, b: seg_orig.shape[1]+b] = seg_orig
                seg_img = transform(Image.fromarray(seg_img)).squeeze().to(device)
            if args.fore_expand:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=args.sigma_upsample_size, mode='nearest').squeeze()
                seg_pos1 = int((args.sigma_upsample_size - args.sigma_size) * 0.5)  # 1
                seg_pos2 = int((args.sigma_upsample_size + args.sigma_size) * 0.5)  # 17
                seg_img = seg_img[seg_pos1:seg_pos2, seg_pos1:seg_pos2]
            else:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=args.sigma_size, mode='nearest').squeeze()
        elif args.dataset == 'ILSVRC':
            seg_img = torch.zeros((args.sigma_size, args.sigma_size))
            seg_img[int(args.sigma_size/6):int(args.sigma_size*5/6)+1, int(args.sigma_size/6):int(args.sigma_size*5/6)+1] = 1
        else:
            sys.exit("dataset error")
        foreground_pos = (seg_img > 0).nonzero().to(device)
        background_pos = (seg_img == 0).nonzero().to(device)

        effect_flag = 1
        # train for every checkpoint
        for idx in reversed(range(model_epoch)):
            checkpoint_idx = checkpoint_list[idx]
            print('\n', id, img_name)
            print("effect count", effect_count, "\tnon_effect count", non_effect_count, '\n')
            plot_checkpoint_path = os.path.join(plot_path, 'checkpoint_' + str(checkpoint_idx))
            checkdir(plot_checkpoint_path)

            ## load models ##
            checkpoint = checkpoint_root + str(checkpoint_idx) + '.pth.tar'
            print(checkpoint)
            load_checkpoint(checkpoint, model)
            model.eval()

            with torch.no_grad():
                # calculate the sigma_f for this image
                sigma_init = torch.ones(size=(1, 1, args.sigma_size, args.sigma_size), device=device) * args.sigma_init_decay
                noise_layer = NoiseLayer(sigma_init, args.image_size).to(device)
                unit_noise = torch.randn(size=(args.batch_size, 3, args.image_size, args.image_size), device=device)
                noise_image, penalty = noise_layer(image, unit_noise)
                noise_feature = get_feature(noise_image, model, layer=capacity_layer)
                origin_feature = get_feature(image, model, layer=capacity_layer)

                noise_feature = relu_method(noise_feature)
                origin_feature = relu_method(origin_feature)
                origin_feature = origin_feature.expand(noise_feature.size())

                sigma_f = criterion(noise_feature, origin_feature)
                sigma_f_matrix[idx] = sigma_f.item()
                print("sigma_f = ", sigma_f, '\n')

                if sigma_f.item() == 0:     # sigma_f==0 当做无效样本
                    non_effect_count += 1
                    name_path = result_path + '/non_effect_name_list.txt'
                    fw = open(name_path, 'a')
                    fw.write(img_name[0])
                    fw.write("\n")
                    fw.close()
                    break

            for iter in range(args.sigma_iteration):
                if args.model[:3] == 'vgg':
                    lambda_param = 7
                    threshold_entropy = -2.8
                elif args.model[:6] == 'resnet':
                    lambda_param = 6
                    threshold_entropy = -2.8
                elif args.model == 'alexnet':
                    lambda_param = 7
                    threshold_entropy = -2.8
                else:
                    sys.exit("error")

                # train the sigma matrix
                sigma_init = torch.ones(size=(1, 1, args.sigma_size, args.sigma_size), device=device) * args.sigma_init_decay
                noise_layer = NoiseLayer(sigma_init, args.image_size).to(device)
                optimizer = torch.optim.SGD(noise_layer.parameters(), lr=args.lr)
                noise_layer.train()
                sigma_data_list = []
                train_feature_loss, train_penalty_loss = AverageMeter(), AverageMeter()
                torch.cuda.empty_cache()

                for epoch in range(args.epoch):
                    # lambda_param = lambda_init * math.e ** (args.lambda_change_ratio * epoch / args.epoch)
                    train_feature_loss.reset()
                    train_penalty_loss.reset()

                    unit_noise = torch.randn(size=(args.batch_size, 3, args.image_size, args.image_size), device=device)
                    noise_image, penalty = noise_layer(image, unit_noise)
                    if torch.isnan(penalty):
                        sys.exit("NaN Error")
                    params_data = noise_layer.sigma.detach()
                    sigma_data_list.append(params_data.cpu().numpy())

                    noise_feature = get_feature(noise_image, model, layer=capacity_layer)
                    noise_feature = relu_method(noise_feature)

                    ## cal loss ##
                    feature_loss = criterion(noise_feature, origin_feature)/sigma_f
                    penalty_loss = -penalty*lambda_param
                    loss = feature_loss + penalty_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if (epoch+1) % (args.print_freq) == 0 or epoch == 0:
                        print("lambda", lambda_param, "threshold", threshold_entropy)
                        print("Train: [" + str(epoch) + "/" + str(args.epoch) + "]" + "\n"+'feature_loss: '+str(float(feature_loss))+"\n"
                              +'entropy: '+str(float(penalty))+"\n")

                    train_feature_loss.update(feature_loss.data.cpu())
                    train_penalty_loss.update(penalty_loss.data.cpu())
                    feature_loss_matrix[idx,iter,epoch], penalty_loss_matrix[idx,iter,epoch], entropy_matrix[idx,iter,epoch] = train_feature_loss.avg, train_penalty_loss.avg, penalty

                    if epoch == args.epoch-args.epoch_break-1:
                        if np.max(entropy_matrix[idx, iter, :args.epoch-args.epoch_break]) > threshold_entropy:
                            break
                        else:
                            lambda_param = lambda_param * 1.5

                ## save the epoch nearest to entropy loss threshold ##
                best_idx_entropy = np.argmin(np.abs(entropy_matrix[idx, iter] - threshold_entropy))
                best_sigma_entropy = sigma_data_list[best_idx_entropy][0][0]
                print("best idx for entropy", best_idx_entropy), print("best sigma for entropy", best_sigma_entropy)
                params_data = torch.log(torch.from_numpy(best_sigma_entropy[np.newaxis,np.newaxis,:,:])) + 0.5 * torch.log(torch.tensor(2 * math.pi)) + torch.tensor(0.5)
                visual_data = F.interpolate(params_data, size=args.image_size, mode='nearest')[0][0].data.cpu()
                print('min:', np.min(np.array(visual_data))), print('max:', np.max(np.array(visual_data)), '\n')
                visual_path = os.path.join(plot_checkpoint_path, 'sigma_best_entropy_' + str(best_idx_entropy) + '_init_' + str(threshold_entropy) + '.jpg')
                plot_feature_new(visual_data, visual_path)

                if args.pick_sample:
                    ## select effect samples when checkpoint_idx >= 40 ##
                    if checkpoint_idx >= 40:
                        effect_flag = select_effect_smaples(best_sigma_entropy, foreground_pos, background_pos)
                        if not effect_flag:
                            non_effect_count += 1
                            name_path = result_path + '/non_effect_name_list.txt'
                            fw = open(name_path, 'a')
                            fw.write(img_name[0])
                            fw.write("\n")
                            fw.close()
                            break
                ## save the best sigma ##
                sigma_matrix_entropy[idx, iter] = best_sigma_entropy

            if not effect_flag:
                break
            # if effective sample, save data
            if idx == 0:
                np.save(npy_save_path+'/'+'feature_loss.npy', feature_loss_matrix)
                np.save(npy_save_path +'/'+ 'penalty_loss.npy', penalty_loss_matrix)
                np.save(npy_save_path + '/' + 'entropy.npy', entropy_matrix)
                np.save(npy_save_path + '/' + 'best_sigma_entropy.npy', sigma_matrix_entropy)
                np.save(npy_save_path + '/' + 'sigma_f_matrix.npy', sigma_f_matrix)
                name_path = result_path + '/effect_name_list.txt'
                fw = open(name_path, 'a')
                fw.write(img_name[0])
                fw.write("\n")
                fw.close()
                os.rename(result_folder, result_folder.replace('non_effect', 'effect'))
                effect_count += 1


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


def checkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def plot_feature_new(feature, visual_path):
    feature = np.array(feature)

    fig, ax = plt.subplots()
    im = ax.imshow(feature, cmap=plt.get_cmap('hot'), interpolation='nearest',
                   vmin=-4.5, vmax=-2.3)
    fig.colorbar(im)
    plt.savefig(visual_path, dpi=70)
    plt.close()


def get_origin_img(dir, image_name, image_size, save_root):
    img = Image.open(dir+'/'+image_name[0])
    img = img.resize((image_size,image_size))
    img.save(save_root+'/'+image_name[0].split('/')[1])


def select_effect_smaples(best_sigma, foreground_pos, background_pos):

    best_sigma = torch.from_numpy(best_sigma).to('cuda:0')
    best_entropy = torch.log(best_sigma) + 0.5 * torch.log(torch.tensor(2 * math.pi)) + torch.tensor(0.5)
    foreground_entropy = best_entropy[foreground_pos[:, 0], foreground_pos[:, 1]].flatten()
    background_entropy = best_entropy[background_pos[:, 0], background_pos[:, 1]].flatten()

    if args.Top10:
        foreground_entropy = foreground_entropy[np.argsort(foreground_entropy)[: int(len(foreground_entropy) * 0.1)]]
        print("top 10% foreground entropy", foreground_entropy.size(), foreground_entropy)
        background_entropy = background_entropy[np.argsort(background_entropy)[: int(len(background_entropy) * 0.1)]]
        print("top 10% background entropy", background_entropy.size(), background_entropy)

    foreground_mean = torch.mean(foreground_entropy)
    background_mean = torch.mean(background_entropy)
    entropy_diff = background_mean - foreground_mean
    print("foreground mean", foreground_mean, "\tbackground mean", background_mean)
    print('entropy_diff', entropy_diff, '\n')
    if entropy_diff < 0 or torch.isnan(entropy_diff):
        effect_flag = False
    else:
        effect_flag = True

    return effect_flag


def _init_fn(worker_id):
    np.random.seed(args.seed + worker_id)


def set_random(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.


def get_feature(data_batch, model, layer):
    if layer == 'conv':
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

    elif layer == 'fc1':
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

    elif layer == 'fc2':
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

    elif layer == 'fc3':
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


if args.capacity_layer == 'all':
    for layer in ['fc1', 'fc2']:
        print('capacity layer: ', layer)
        train_sigma(path, args, capacity_layer=layer, checkpoint_root=args.checkpoint_root)
elif args.capacity_layer == 'distil_all':
    distil_fc1 = './KD/trained_model/distil_resnet101_fc1_CUB_106_1-1/'
    distil_fc2 = './KD/trained_model/distil_resnet101_fc2_CUB_106_1-1/'
    distil_fc3 = './KD/trained_model/distil_resnet101_fc3_CUB_106_-2/'
    for layer, check_root in [('fc1', distil_fc1), ('fc2', distil_fc2), ('fc3', distil_fc3)]:
        print('capacity layer: ', layer)
        train_sigma(path, args, capacity_layer=layer, checkpoint_root=check_root)
else:
    print('capacity layer: ', args.capacity_layer)
    train_sigma(path, args, capacity_layer=args.capacity_layer, checkpoint_root=args.checkpoint_root)