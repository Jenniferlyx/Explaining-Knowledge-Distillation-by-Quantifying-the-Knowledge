## This code is used to find knowledge point, analyze, and plot figures ##
'''
 The main functions are :
 1. find sample: used to find same samples for student network and distillation network;

 2. find total background: used to find all same imgs' background entropy mean;

 3. find threshold: used to find the threshlod for total_background_mean - foreground(per pixel) > threshold;

 4. find peak: used to find knowledgepoints and their influences(compare function) for background/foreground;

 5. find convergence: used to find the convergence rate for each checkpoint;

 6. find percent: used to find knowledge points in the last checkpoints / all checkpoints;


'''

import numpy as np
import os
import sys
from PIL import Image
import torch
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from PIL import ImageDraw
import shutil
import xlwt
from torchvision import transforms
from model.models import *

threshold_mode = 'entropy'      # feature or entropy

FIND_SAME_SAMPLE = True         # find student network and distillation network same samples##
FIND_TOTAL_BACKGROUND = True    # find the mean value of all images' background entropy ##
FIND_THRESHOLD = False           # find threshold for background - foreground##
KNOWLEDGE_POINT = 'both'        # 'foreground' only finds knowledge points for foreground; 'both' finds both for foreground and background##
FIND_PEAK = False              # find every peak's influence ##
FIND_CONVERGENCE = True
FIND_PERCENT = False

image_size = 224
checkpoint_step = 3
PEAK_SIZE = 1
peak = False                    # whether to select highlight points
fore_expand = False              # whether to expand the foreground
sigma_size = 16
downsample_size = 18            # in order to enhance foreground influence ##

knowledge_num_max = 100
convergence_y_max = 100


def find_sample(label_file_path, distill_file_path):
    distillation_files = open(distill_file_path).readlines()
    student_files = open(label_file_path).readlines()
    same_sample = list(set(distillation_files).intersection(set(student_files)))
    print(len(same_sample), same_sample)
    print('---------------FIND SAME SAMPLE IS READY!---------------\n')
    return same_sample


def find_total_background(img_names, counter, entropy_thresh):
    full_name_list = open(sigma_root + 'full_name_list.txt').readlines()

    checkpoint_num = len(checkpoints)
    total_background_mean = [[] for _ in range(checkpoint_num)]
    nan_names, effect_names = [], []
    count = 0
    minimum = 1
    for img_name in img_names:
        img_idx = full_name_list.index(img_name)
        path_idx = 'img_' + str(img_idx) + '_effect'
        print('image index is:', img_idx)

        if threshold_mode == 'entropy':
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_entropy.npy'
        else:
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_feature.npy'
        best_sigma_list = np.load(sigma_path)

        minimum = min(minimum, np.min(best_sigma_list))
        print(np.min(best_sigma_list))
        if np.min(best_sigma_list) <= 0:
            nan_names.append(img_name)
            print(img_name, 'has nan')
            continue
        else:
            effect_names.append(img_name)

            if dataset == 'CUB' or dataset == 'VOC':
                if dataset == 'CUB':
                    seg_img = transform(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L')).squeeze()
                if dataset == 'VOC':
                    seg_orig = np.array(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L'))
                    seg_img = np.zeros(shape=(int(seg_orig.shape[0] * 1.2), int(seg_orig.shape[1] * 1.2)))
                    a = int((seg_img.shape[0] - seg_orig.shape[0]) / 2)
                    b = int((seg_img.shape[1] - seg_orig.shape[1]) / 2)
                    seg_img[a: seg_orig.shape[0] + a, b: seg_orig.shape[1] + b] = seg_orig
                    seg_img = transform(Image.fromarray(seg_img)).squeeze()
                if fore_expand:
                    seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=downsample_size, mode='nearest').squeeze()[1:sigma_size + 1, 1:sigma_size + 1]
                else:
                    seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=sigma_size, mode='nearest').squeeze()
            elif dataset == 'ILSVRC':
                seg_img = torch.zeros((sigma_size, sigma_size))
                seg_img[int(sigma_size / 6):int(sigma_size * 5 / 6) + 1, int(sigma_size / 6):int(sigma_size * 5 / 6) + 1] = 1

            background_pos = (seg_img == 0).nonzero()

            count += 1
            for checkpoint_idx in range(checkpoint_num):
                best_sigma = torch.from_numpy(best_sigma_list[checkpoint_idx, counter])# (16,16)
                background_sigma = best_sigma[background_pos[:, 0], background_pos[:, 1]].flatten()
                background_entropy_mean = torch.mean(entropy(background_sigma))

                if torch.isnan(background_entropy_mean):
                    print("nan", best_sigma)

                total_background_mean[checkpoint_idx].append(background_entropy_mean.item())


    total_background_mean = np.mean(total_background_mean, axis=1)
    print('effect name list:', len(effect_names), effect_names)
    print('total_background_mean', count, total_background_mean)
    print("sigma min", minimum)
    fw.write('---------------------------' + '\n')
    fw.write("background mean: " + str(total_background_mean) + '\n')

    print('---------------FIND TOTAL BACKGROUND MEAN IS READY!---------------\n')
    return total_background_mean, checkpoint_num, effect_names


def find_threshold(img_names, total_background_mean, checkpoint_num, counter, entropy_thresh):

    FOREGROUND_PLOT = True

    full_name_list = open(sigma_root + 'full_name_list.txt').readlines()
    pic_num = len(img_names)
    color_num = checkpoint_num * pic_num

    plt.figure(figsize=(20, 10))
    plt.xlabel('forground pixels')
    plt.ylabel('entropy difference')
    color = iter(plt.cm.rainbow(np.linspace(0, 1, color_num)))

    save_root = save_path + 'find_threshold/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    if threshold_mode == 'entropy':
        if FOREGROUND_PLOT:
            entropy_diff_img = save_root + str(entropy_thresh[counter]) + 'entropy_diff_foreground_entropy.jpg'
            title = model + '_Foreground_Difference_Entropy_Threshold'
        else:
            entropy_diff_img = save_root + str(entropy_thresh[counter]) + 'entropy_diff_whole_entropy.jpg'
            title = model + '_Whole_Pic_Difference_Entropy_Threshold'
    else:
        if FOREGROUND_PLOT:
            entropy_diff_img = save_root + str(entropy_thresh[counter]) + 'entropy_diff_foreground_feature.jpg'
            title = model + '_Foreground_Difference_Feature_Threshold'
        else:
            entropy_diff_img = save_root + str(entropy_thresh[counter]) + 'entropy_diff_whole_feature.jpg'
            title = model + '_Whole_Pic_Difference_Feature_Threshold'
    plt.title(title)

    entropy_diff_list = []
    back_entropy_list = []
    for img_name in img_names:
        img_idx = full_name_list.index(img_name)
        print('image index is:', img_idx)

        path_idx = 'img_' + str(img_idx) + '_effect'

        if threshold_mode == 'entropy':
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_entropy.npy'
        else:
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_feature.npy'
        best_sigma_list = np.load(sigma_path)

        if dataset == 'CUB' or dataset == 'VOC':
            if dataset == 'CUB':
                seg_img = transform(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L')).squeeze()
            if dataset == 'VOC':
                seg_orig = np.array(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L'))
                seg_img = np.zeros(shape=(int(seg_orig.shape[0] * 1.2), int(seg_orig.shape[1] * 1.2)))
                a = int((seg_img.shape[0] - seg_orig.shape[0]) / 2)
                b = int((seg_img.shape[1] - seg_orig.shape[1]) / 2)
                seg_img[a: seg_orig.shape[0] + a, b: seg_orig.shape[1] + b] = seg_orig
                seg_img = transform(Image.fromarray(seg_img)).squeeze()
            if fore_expand:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=downsample_size, mode='nearest').squeeze()[1:sigma_size + 1, 1:sigma_size + 1]
            else:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=sigma_size, mode='nearest').squeeze()
        elif dataset == 'ILSVRC':
            seg_img = torch.zeros((sigma_size, sigma_size))
            seg_img[int(sigma_size / 6):int(sigma_size * 5 / 6) + 1, int(sigma_size / 6):int(sigma_size * 5 / 6) + 1] = 1

        foreground_pos = (seg_img > 0).nonzero()
        background_pos = (seg_img == 0).nonzero()
        for checkpoint_idx in range(best_sigma_list.shape[0]):

            best_sigma = torch.from_numpy(best_sigma_list[checkpoint_idx, counter])
            background_sigma = best_sigma[background_pos[:, 0], background_pos[:, 1]].flatten()
            back_entropy_list += entropy(background_sigma).numpy().tolist()

            whole_pic_entorpy = entropy(best_sigma)
            if FOREGROUND_PLOT:
                entropy_diff_foreground = (total_background_mean[checkpoint_idx] - whole_pic_entorpy[foreground_pos[:, 0], foreground_pos[:, 1]]).data.cpu().numpy()
                plt.plot(list(range(entropy_diff_foreground.shape[0])), sorted(entropy_diff_foreground),color=next(color))
                entropy_diff_list += entropy_diff_foreground.tolist()
            else:
                entropy_diff = (total_background_mean[checkpoint_idx] - whole_pic_entorpy).flatten().data.cpu().numpy()
                plt.plot(list(range(entropy_diff.shape[0])), sorted(entropy_diff), color=next(color))
                entropy_diff_list += entropy_diff.tolist()

    background_mean = np.mean(back_entropy_list)
    entropy_diff_mean = np.mean(entropy_diff_list)
    entropy_diff_list.sort(reverse=True)
    entropy_diff_top10 = entropy_diff_list[int(len(entropy_diff_list) * 0.1)]
    entropy_diff_top5 = entropy_diff_list[int(len(entropy_diff_list) * 0.05)]
    entropy_diff_top1 = entropy_diff_list[int(len(entropy_diff_list) * 0.01)]
    print("background_mean", background_mean)
    print("entropy_diff_mean", entropy_diff_mean)
    print("entropy_diff_top10", entropy_diff_top10)
    print("entropy_diff_top5", entropy_diff_top5)
    print("entropy_diff_top1", entropy_diff_top1)

    fw.write('---------------------------' + '\n')
    fw.write("background_mean: "+str(background_mean)+'\n')
    fw.write("entropy_diff_mean: "+ str(entropy_diff_mean)+ '\n')
    fw.write("entropy_diff_top10: " + str(entropy_diff_top10) + '\n')
    fw.write("entropy_diff_top5: "+str(entropy_diff_top5)+ '\n')
    fw.write("entropy_diff_top1: " + str(entropy_diff_top1)+ '\n')

    plt.plot(list(range(220)), [entropy_diff_top1] * 220, color='blue', marker='o', linewidth=6, label='entropy_diff_top1')
    plt.plot(list(range(220)), [entropy_diff_top5] * 220, color='red', marker='o', linewidth=6, label='entropy_diff_top5')
    plt.plot(list(range(220)), [entropy_diff_top10] * 220, color='green', marker='o', linewidth=6, label='entropy_diff_top10')
    plt.plot(list(range(220)), [entropy_diff_mean] * 220, color='black', marker='o', linewidth=6, label='entropy_diff_mean')
    plt.legend()
    plt.savefig(entropy_diff_img)
    print("plot complete!\n")


def find_peak(img_names,sigma_size,threshold,peak_size,downsample_size,find_mode,total_background_mean, counter, entropy_thresh):

    save_root = save_path + 'peak_influence_thresh_mode_{}_{}_{}/'.format(threshold_mode, entropy_thresh[counter], THRESHOLD)
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    result_root = save_root + '/result/'
    if not os.path.exists(result_root):
        os.mkdir(result_root)

    full_name_list = open(sigma_root+'full_name_list.txt').readlines()

    knowledge_foreground_num, knowledge_background_num, knowledge_total_num = [], [], []     # area num
    knowledge_foreground_ratio, knowledge_background_ratio, knowledge_total_ratio = [], [], []     # area num
    for img_name in img_names:
        img_idx = full_name_list.index(img_name)
        path_idx = 'img_'+str(img_idx)+'_effect'

        print('image index is:', img_idx)
        highlight_img_root = save_root + 'img_'+str(img_idx)
        if not os.path.exists(highlight_img_root):
            os.mkdir(highlight_img_root)

        ## copy origin imgs ##
        new_img_name = img_name.split('/')[-1].strip('\n')
        shutil.copyfile(sigma_root + path_idx + '/' + new_img_name, highlight_img_root + '/' + new_img_name)

        if dataset == 'CUB' or dataset == 'VOC':
            if dataset == 'CUB':
                seg_img = transform(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L')).squeeze()
            if dataset == 'VOC':
                seg_orig = np.array(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L'))
                seg_img = np.zeros(shape=(int(seg_orig.shape[0] * 1.2), int(seg_orig.shape[1] * 1.2)))
                a = int((seg_img.shape[0] - seg_orig.shape[0]) / 2)
                b = int((seg_img.shape[1] - seg_orig.shape[1]) / 2)
                seg_img[a: seg_orig.shape[0] + a, b: seg_orig.shape[1] + b] = seg_orig
                seg_img = transform(Image.fromarray(seg_img)).squeeze()
            shutil.copyfile(segdir + '/' + img_name.replace('jpg', 'png').strip('\n'), highlight_img_root + '/' + new_img_name.strip('.jpg') + '_seg.png')
            if fore_expand:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=downsample_size, mode='nearest').squeeze()[1:sigma_size + 1, 1:sigma_size + 1]
            else:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=sigma_size, mode='nearest').squeeze()
        elif dataset == 'ILSVRC':
            seg_img = torch.zeros((sigma_size, sigma_size))
            seg_img[int(sigma_size / 6):int(sigma_size * 5 / 6) + 1, int(sigma_size / 6):int(sigma_size * 5 / 6) + 1] = 1

        foreground_pos = (seg_img > 0).nonzero()
        background_pos = (seg_img == 0).nonzero()

        if threshold_mode == 'entropy':
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_entropy.npy'
        else:
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_feature.npy'
        best_sigma_list = np.load(sigma_path)
        foreground_num_temp, background_num_temp, total_num_tmp = [], [], []
        foreground_ratio_temp, background_ratio_temp, total_ratio_temp = [], [], []

        for checkpoint_idx in range(best_sigma_list.shape[0]):

            print(checkpoints[checkpoint_idx])
            highlight_pos, highlight_entropy = [], []   # foreground highlight position and entropy
            foreground_num, background_num = 0, 0

            best_sigma = best_sigma_list[checkpoint_idx, counter] #(16,16)
            best_entropy = entropy(torch.from_numpy(best_sigma))
            whole_pic_entropy = total_background_mean[checkpoint_idx] - best_entropy

            highlight_pixel = (whole_pic_entropy > threshold).nonzero()
            below_pixel = (whole_pic_entropy <= threshold).nonzero()

            whole_pic_entropy_copy = whole_pic_entropy.clone()
            ## background entropy is set zero ##

            if find_mode == 'foreground':
                for pos in highlight_pixel.numpy().tolist():
                    if pos in foreground_pos.numpy().tolist():
                        highlight_pos.append(pos)
                        highlight_entropy.append(whole_pic_entropy[pos[0], pos[1]].numpy())

            elif find_mode == 'both':
                for pos in highlight_pixel.numpy().tolist():
                    highlight_pos.append(pos)
                    highlight_entropy.append(whole_pic_entropy[pos[0], pos[1]].numpy())

            elif find_mode == 'background':
                for pos in highlight_pixel.numpy().tolist():
                    if pos in background_pos.numpy().tolist():
                        highlight_pos.append(pos)
                        highlight_entropy.append(whole_pic_entropy[pos[0], pos[1]].numpy())


            if peak:
                highlight_entropy_sort = sorted(highlight_entropy, reverse=True)
                pixel_sort = np.argsort(highlight_entropy)[::-1]

                select_entropy = []
                select_pos = []
                for pixel_id in range(len(highlight_entropy_sort)):
                    pixel_x, pixel_y = highlight_pos[pixel_sort[pixel_id]][0], highlight_pos[pixel_sort[pixel_id]][1]
                    if whole_pic_entropy_copy[pixel_x, pixel_y] != 0:
                        slice = whole_pic_entropy_copy[np.max([0, pixel_x - peak_size]):np.min([pixel_x + peak_size + 1, sigma_size]), np.max([0, pixel_y - peak_size]):np.min([pixel_y + peak_size + 1, sigma_size])]
                        center_index_x = pixel_x - np.max([0, pixel_x - peak_size])
                        center_index_y = pixel_y - np.max([0, pixel_y - peak_size])
                        if torch.argmax(slice) == center_index_x * slice.shape[1] + center_index_y:
                            select_entropy.append(highlight_entropy_sort[pixel_id]), select_pos.append(highlight_pos[pixel_sort[pixel_id]])
                            # set zero to make peak sparse ##
                            whole_pic_entropy_copy[np.max([0, pixel_x - peak_size]):np.min([pixel_x + peak_size + 1, sigma_size]), np.max([0, pixel_y - peak_size]):np.min([pixel_y + peak_size + 1, sigma_size])] = 0
                            whole_pic_entropy_copy[pixel_x, pixel_y] = torch.Tensor(highlight_entropy_sort[pixel_id])

                select_num = len(select_pos)
                allpoint_size = whole_pic_entropy.flatten().shape[0]
                point_pos = np.zeros((allpoint_size, 4))

                for m in range(1, select_num + 1):
                    flag = np.ones((allpoint_size, 4))
                    position_x = select_pos[m - 1][0]
                    position_y = select_pos[m - 1][1]
                    position = position_x * sigma_size + position_y
                    point_pos[position][0] = m
                    point_pos[position][1] = 0  # iternum
                    point_pos[position][2] = select_pos[m - 1][0]  # x
                    point_pos[position][3] = select_pos[m - 1][1]  # y
                    iternum = 0
                    point_pos = compare(point_pos[position][2], point_pos[position][3], flag, m, iternum, sigma_size, point_pos, whole_pic_entropy)

                feature = point_pos[:, 0].reshape(sigma_size, sigma_size)
                feature[below_pixel[:, 0], below_pixel[:, 1]] = np.nan
                plt.imshow(feature)
                for pos in select_pos:
                    plt.scatter(pos[1], pos[0], color='red')
                figure_name = highlight_img_root + '/peak_back_' + str(checkpoints[checkpoint_idx]) + '.png'
                plt.savefig(figure_name, dpi=70)
                plt.close()

                feature[background_pos[:, 0], background_pos[:, 1]] = -1    # background
                plt.imshow(feature)

                for pos in select_pos:
                    plt.scatter(pos[1], pos[0], color='red')
                    if pos in foreground_pos.numpy().tolist():
                        foreground_num += 1
                    elif pos in background_pos.numpy().tolist():
                        background_num += 1
                figure_name = highlight_img_root + '/peak_'+str(checkpoints[checkpoint_idx]) + '.png'
                plt.savefig(figure_name, dpi=70)
                plt.close()

            else:
                feature = best_entropy
                feature[background_pos[:, 0], background_pos[:, 1]] = np.nan    # background
                fig, ax = plt.subplots()
                im = ax.imshow(feature, cmap=plt.get_cmap('hot'), interpolation='nearest', vmin=-4.5, vmax=-2.3)
                fig.colorbar(im)

                for pos in highlight_pos:
                    plt.scatter(pos[1], pos[0], color='blue')
                    if pos in foreground_pos.numpy().tolist():
                        foreground_num += 1
                    elif pos in background_pos.numpy().tolist():
                        background_num += 1
                figure_name = highlight_img_root + '/high_' + str(checkpoints[checkpoint_idx]) + '.png'
                plt.savefig(figure_name, dpi=70)
                plt.close()
            print("fore", foreground_num, "back", background_num)


            origin_img_path = sigma_root + path_idx + '/plot/checkpoint_' + str(checkpoints[checkpoint_idx])
            imgs = os.listdir(origin_img_path)
            for im in imgs:
                if threshold_mode == 'entropy':
                    if 'entropy' in im and str(entropy_thresh[counter]) in im:
                        sigma_best_img = im
                else:
                    if 'feature' in im and str(entropy_thresh[counter]) in im:
                        sigma_best_img = im
            origin_img = Image.open(os.path.join(origin_img_path, sigma_best_img))
            highlight_img_name = highlight_img_root + '/checkpoint_' + 'origin_' + str(checkpoints[checkpoint_idx]) + '_' + sigma_best_img
            origin_img.save(highlight_img_name)

            # calculate and save the knowledge
            foreground_num_temp.append(foreground_num)
            background_num_temp.append(background_num)
            total_num_tmp.append(foreground_num + background_num)
            if (foreground_num + background_num):
                foreground_ratio_temp.append(foreground_num/(foreground_num + background_num))
            else:
                foreground_ratio_temp.append(0)

        knowledge_foreground_num.append(foreground_num_temp)
        knowledge_background_num.append(background_num_temp)
        knowledge_total_num.append(total_num_tmp)
        knowledge_foreground_ratio.append(foreground_ratio_temp)

    np.save(result_root + '/knowledge_foreground_num.npy', np.array(knowledge_foreground_num))
    np.save(result_root + '/knowledge_background_num.npy', np.array(knowledge_background_num))
    np.save(result_root + '/knowledge_total_num.npy', np.array(knowledge_total_num))

    plt_knowdge(model, result_root, knowledge_foreground_num, fore_back_total=0, ratio=False)
    plt_knowdge(model, result_root, knowledge_background_num, fore_back_total=1, ratio=False)
    plt_knowdge(model, result_root, knowledge_total_num, fore_back_total=2, ratio=False)
    plt_knowdge(model, result_root, knowledge_foreground_ratio, fore_back_total=0, ratio=True)
    print(np.mean(knowledge_foreground_num, axis=0), np.mean(knowledge_background_num, axis=0), np.mean(knowledge_total_num, axis=0))

    fw.write('---------------------------' + '\n')
    fw.write('foreground num: '+str(np.mean(knowledge_foreground_num, axis=0))+'\n')
    fw.write('background num: '+str(np.mean(knowledge_background_num, axis=0))+'\n')
    fw.write('total num: ' + str(np.mean(knowledge_total_num, axis=0)) + '\n')
    fw.write('foreground ratio: ' + str(np.mean(knowledge_foreground_ratio, axis=0)) + '\n')
    print("plot complete!\n")


def plt_knowdge(model, result_root, knowledge, fore_back_total=0, ratio=True):

    if fore_back_total == 0:
        name = 'foreground'
    elif fore_back_total == 1:
        name = 'background'
    elif fore_back_total == 2:
        name = 'total'

    figure, ax = plt.subplots(figsize=(20, 10))
    # 设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=20)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置横纵坐标的名称以及对应字体格式
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 25,
             }

    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(knowledge))))
    for i in range(len(knowledge)):
        offset = i * 0.002
        know = [kn + offset for kn in knowledge[i]]
        plt.plot(checkpoints, know, color=next(color))
    plt.plot(checkpoints, np.mean(knowledge, axis=0), color='red', marker='o', linewidth=8)
    if ratio:
        plt.title('{} {} knowledge ratio'.format(model, name), font2)
        plt.ylim(0, 1)
    else:
        plt.title('{} {} knowledge num'.format(model, name), font2)
        plt.ylim(0, knowledge_num_max)
    plt.xlabel('model epochs', font2)

    if ratio:
        plt.ylabel('knowledge ratio', font2)
    else:
        plt.ylabel('knowledge number', font2)

    if ratio:
        img_name = result_root + '{}_{}_knowledge_ratio'.format(model, name)
    else:
        img_name = result_root + '{}_{}_knowledge_num'.format(model, name)
    if not peak:
        img_name = img_name + '_no_peak'
    plt.savefig(img_name)


def entropy(sigma):
    entropy = torch.log(sigma) + 0.5 * torch.log(torch.tensor(2 * math.pi * math.e))
    return entropy


def find_convergence(img_names, sigma_size, threshold, downsample_size, total_background_mean, teahcer_checkpoint_path, label_checkpoint_path, distil_checkpoint_path, layer, submode, counter):

    save_root = save_path + '/find_convergence/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    if layer == 'conv':
        index = 0
    elif layer == 'fc1':
        index = 1
    elif layer == 'fc2':
        index = 2
    elif layer == 'fc3':
        index = 3
    else:
        sys.exit('layer error')

    teacher_weight = np.load(teahcer_checkpoint_path + '/weight_diff.npy')
    teacher_weight_pro = np.zeros(teacher_weight.shape[1])
    for epoch_idx in range(teacher_weight.shape[1]):
        teacher_weight_pro[epoch_idx] = np.sum(teacher_weight[index][:epoch_idx + 1])

    label_weight = np.load(label_checkpoint_path + '/weight_diff.npy')
    label_weight_pro = np.zeros(label_weight.shape[1])
    for epoch_idx in range(label_weight.shape[1]):
        label_weight_pro[epoch_idx] = np.sum(label_weight[index][:epoch_idx + 1])

    distil_weight = np.load(distil_checkpoint_path + '/weight_diff.npy')
    distil_weight_pro = np.zeros(distil_weight.shape[1])
    for epoch_idx in range(distil_weight.shape[1]):
        distil_weight_pro[epoch_idx] = np.sum(distil_weight[index][:epoch_idx + 1])


    convergence_x_max = max(teacher_weight_pro[-1], label_weight_pro[-1], distil_weight_pro[-1]) + 0.1

    if submode == 'teacher':
        model_weight_pro = teacher_weight_pro
    elif submode == 'label_net':
        model_weight_pro = label_weight_pro
    elif submode == 'distil_net':
        model_weight_pro = distil_weight_pro
    else:
        sys.exit('submode error')

    full_name_list = open(sigma_root + 'full_name_list.txt').readlines()
    fore_knowledge_value = []
    back_knowledge_value = []
    fore_max_index = []
    back_max_index = []
    fore_slope = []
    back_slope = []
    index = 0

    for img_name in img_names:

        fore_knowledge_value.append([])
        back_knowledge_value.append([])

        img_idx = full_name_list.index(img_name)
        path_idx = 'img_' + str(img_idx) + '_effect'
        print('image index is:', img_idx)

        if threshold_mode == 'entropy':
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_entropy.npy'
        else:
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_feature.npy'
        best_sigma_list = np.load(sigma_path)

        if dataset == 'CUB' or dataset == 'VOC':
            if dataset == 'CUB':
                seg_img = transform(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L')).squeeze()
            if dataset == 'VOC':
                seg_orig = np.array(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L'))
                seg_img = np.zeros(shape=(int(seg_orig.shape[0] * 1.2), int(seg_orig.shape[1] * 1.2)))
                a = int((seg_img.shape[0] - seg_orig.shape[0]) / 2)
                b = int((seg_img.shape[1] - seg_orig.shape[1]) / 2)
                seg_img[a: seg_orig.shape[0] + a, b: seg_orig.shape[1] + b] = seg_orig
                seg_img = transform(Image.fromarray(seg_img)).squeeze()
            if fore_expand:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=downsample_size, mode='nearest').squeeze()[1:sigma_size + 1, 1:sigma_size + 1]
            else:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=sigma_size, mode='nearest').squeeze()
        elif dataset == 'ILSVRC':
            seg_img = torch.zeros((sigma_size, sigma_size))
            seg_img[int(sigma_size / 6):int(sigma_size * 5 / 6) + 1, int(sigma_size / 6):int(sigma_size * 5 / 6) + 1] = 1

        foreground_pos = (seg_img > 0).nonzero()
        background_pos = (seg_img == 0).nonzero()

        for checkpoint_idx in range(best_sigma_list.shape[0]):
            if 'resnet' in model_name:
                if checkpoint_idx == 0:
                    continue

            highlight_back, highlight_fore = 0, 0  # foreground highlight position and entropy
            best_entropy = entropy(torch.from_numpy(best_sigma_list[checkpoint_idx, counter]))
            whole_pic_entropy = total_background_mean[checkpoint_idx] - best_entropy
            highlight_pixel = (whole_pic_entropy > threshold).nonzero()

            for pos in highlight_pixel.numpy().tolist():
                if pos in background_pos.numpy().tolist():
                    highlight_back += 1
                else:
                    highlight_fore += 1
            fore_knowledge_value[index].append(highlight_fore)
            back_knowledge_value[index].append(highlight_back)

        fore = np.argmax(fore_knowledge_value[index])
        fore_max_index.append(fore)
        fore_slope.append(fore_knowledge_value[index][fore] / model_weight_pro[checkpoints[fore]])

        back = np.argmax(back_knowledge_value[index])
        back_max_index.append(back)
        back_slope.append(back_knowledge_value[index][back] / model_weight_pro[checkpoints[back]])
        index += 1

    fore_max_mean = np.mean(model_weight_pro[np.array(checkpoints)[fore_max_index]])
    fore_max_var = np.var(model_weight_pro[np.array(checkpoints)[fore_max_index]])
    fore_max_slope = np.mean(fore_slope)
    back_max_mean = np.mean(model_weight_pro[np.array(checkpoints)[back_max_index]])
    back_max_var = np.var(model_weight_pro[np.array(checkpoints)[back_max_index]])
    back_max_slope = np.mean(back_slope)

    fw.write('---------------------------' + '\n')
    fw.write('fore max mean: '+str(fore_max_mean)+'\n')
    fw.write('fore max var: ' + str(fore_max_var) + '\n')
    fw.write('fore max slope: ' + str(fore_max_slope) + '\n')
    fw.write('back max mean: '+str(back_max_mean)+'\n')
    fw.write('back max var: ' + str(back_max_var) + '\n')
    fw.write('back max slope: ' + str(back_max_slope) + '\n')

    ## plot curves ##
    plt.figure(figsize=(20, 10))
    plt.subplot(211)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, index)))
    if 'resnet' in model_name:
        for fore in fore_knowledge_value:
            plt.plot(model_weight_pro[checkpoints[1:]], fore, color=next(color))
        plt.plot(model_weight_pro[checkpoints[1:]], np.mean(fore_knowledge_value, axis=0), color='black', marker='o',
                 linewidth=8)
    else:
        for fore in fore_knowledge_value:
            plt.plot(model_weight_pro[checkpoints], fore, color=next(color))
        plt.plot(model_weight_pro[checkpoints], np.mean(fore_knowledge_value, axis=0), color='black', marker='o', linewidth=8)
    plt.xlabel('Weight Distance')
    plt.ylabel('Foreground Knowledge Number')
    title = 'Foreground Entropy Sigma Convergence ' + '{} {} {}'.format(model, layer, str(THRESHOLD))
    plt.title(title)
    plt.xlim(0, convergence_x_max)
    plt.ylim(0, convergence_y_max)

    plt.subplot(212)
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(back_knowledge_value))))
    if 'resnet' in model_name:
        for back in back_knowledge_value:
            plt.plot(model_weight_pro[checkpoints[1:]], back, color=next(color))
        if back_knowledge_value:
            plt.plot(model_weight_pro[checkpoints[1:]], np.mean(back_knowledge_value, axis=0), color='black', marker='o',
                     linewidth=8)
    else:
        for back in back_knowledge_value:
            plt.plot(model_weight_pro[checkpoints], back, color=next(color))
        if back_knowledge_value:
            plt.plot(model_weight_pro[checkpoints], np.mean(back_knowledge_value, axis=0), color='black', marker='o',
                     linewidth=8)
    plt.xlabel('Weight Diff')
    plt.ylabel('Background Knowledge Number')
    title = 'BackgroundEntropy Sigma Convergence ' + '{} {} {}'.format(model, layer, str(THRESHOLD))
    plt.title(title)
    plt.xlim(0, convergence_x_max)
    plt.ylim(0, convergence_y_max)

    img_name = save_root + 'convergence_' + model + '_' + layer + '_' + str(THRESHOLD)+'.jpg'
    plt.savefig(img_name)
    print("plot complete!")


def find_percent(img_names,sigma_size,threshold,downsample_size,total_background_mean, counter):
    full_name_list = open(sigma_root + 'full_name_list.txt').readlines()

    workbook_wt = xlwt.Workbook()
    sheet_wt = workbook_wt.add_sheet('Sheet1')

    sheet_wt.write_merge(0, 0, 0, 6, model)
    sheet_wt.write(1, 0, 'Image 1 Name')
    sheet_wt.write(1, 1, 'foreground_num')
    sheet_wt.write(1, 2, 'foreground_ratio')
    sheet_wt.write(1, 3, 'background_num')
    sheet_wt.write(1, 4, 'background_ratio')
    sheet_wt.write(1, 5, 'both_num')
    sheet_wt.write(1, 6, 'both_ratio')

    save_root = sigma_root + '/find_percent/'
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    row_index = 2
    img_names.sort()
    total_fore, total_back, total_total = 0, 0, 0
    for img_name in img_names:

        sheet_wt.write(row_index, 0, img_name)

        img_idx = full_name_list.index(img_name)
        path_idx = 'img_' + str(img_idx) + '_effect'
        print('image index is:', img_idx)

        if threshold_mode == 'entropy':
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_entropy.npy'
        else:
            sigma_path = sigma_root + path_idx + '/npy_files/best_sigma_feature.npy'
        best_sigma_list = np.load(sigma_path)

        if dataset == 'CUB' or dataset == 'VOC':
            if dataset == 'CUB':
                seg_img = transform(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L')).squeeze()
            if dataset == 'VOC':
                seg_orig = np.array(Image.open(segdir + img_name.replace('jpg\n', 'png')).convert('L'))
                seg_img = np.zeros(shape=(int(seg_orig.shape[0] * 1.2), int(seg_orig.shape[1] * 1.2)))
                a = int((seg_img.shape[0] - seg_orig.shape[0]) / 2)
                b = int((seg_img.shape[1] - seg_orig.shape[1]) / 2)
                seg_img[a: seg_orig.shape[0] + a, b: seg_orig.shape[1] + b] = seg_orig
                seg_img = transform(Image.fromarray(seg_img)).squeeze()
            if fore_expand:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=downsample_size, mode='nearest').squeeze()[1:sigma_size + 1, 1:sigma_size + 1]
            else:
                seg_img = F.interpolate(seg_img[np.newaxis, np.newaxis, :, :].float(), size=sigma_size, mode='nearest').squeeze()
        elif dataset == 'ILSVRC':
            seg_img = torch.zeros((sigma_size, sigma_size))
            seg_img[int(sigma_size / 6):int(sigma_size * 5 / 6) + 1, int(sigma_size / 6):int(sigma_size * 5 / 6) + 1] = 1

        foreground_pos = (seg_img > 0).nonzero()
        background_pos = (seg_img == 0).nonzero()

        total_foreground_list, total_background_list, total_list = [], [], []
        foreground_num, background_num = 0, 0
        for checkpoint_idx in range(best_sigma_list.shape[0]):
            print(checkpoints[checkpoint_idx])

            best_sigma = best_sigma_list[checkpoint_idx, counter]  # (16,16)
            best_entropy = entropy(torch.from_numpy(best_sigma))
            whole_pic_entropy = total_background_mean[checkpoint_idx] - best_entropy
            highlight_pixel = (whole_pic_entropy > threshold).nonzero()

            for pos in highlight_pixel.numpy().tolist():
                if pos not in total_list:
                    total_list.append(pos)
                if pos in foreground_pos.numpy().tolist():
                    if checkpoint_idx == best_sigma_list.shape[0] - 1:
                        foreground_num += 1
                    if pos not in total_foreground_list:
                        total_foreground_list.append(pos)
                else:
                    if checkpoint_idx == best_sigma_list.shape[0] - 1:
                        background_num += 1
                    if pos not in total_background_list:
                        total_background_list.append(pos)

        str_num = str(foreground_num) + '/' + str(len(total_foreground_list))
        sheet_wt.write(row_index, 1, str_num)
        if total_foreground_list:
            sheet_wt.write(row_index, 2, foreground_num/len(total_foreground_list))
        else:
            sheet_wt.write(row_index, 2, 'null')

        str_num = str(background_num) + '/' + str(len(total_background_list))
        sheet_wt.write(row_index, 3, str_num)
        if total_background_list:
            sheet_wt.write(row_index, 4, background_num/len(total_background_list))
        else:
            sheet_wt.write(row_index, 4, 'null')

        str_num = str(foreground_num + background_num) + '/' + str(len(total_list))
        sheet_wt.write(row_index, 5, str_num)
        if total_list:
            sheet_wt.write(row_index, 6, (foreground_num + background_num)/len(total_list))
        else:
            sheet_wt.write(row_index, 6, 'null')

        row_index += 1
        if total_foreground_list:
            total_fore += (foreground_num / len(total_foreground_list))
        else:
            total_fore += 0
        if total_background_list:
            total_back += (background_num / len(total_background_list))
        else:
            total_back += 0
        if total_list:
            total_total += ((foreground_num + background_num) / len(total_list))
        else:
            total_total += 0

    fw.write('---------------------------' + '\n')
    fw.write('fore percent: ' + str(total_fore/(row_index-2))+'\n')
    fw.write('back percent: ' + str(total_back / (row_index - 2))+'\n')
    fw.write('total percent: ' + str(total_total / (row_index - 2)) + '\n')

    workbook_wt.save(save_root+model+"_percent.xls")
    print("knowledge percent calculated complete!")




def choose_ckpt_list(ckpt_root):
    ckpt_list1 = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 130, 150]
    ckpt_list2 = [0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 90, 110, 130, 150, 170, 190, 240, 280, 320, 360, 400]
    if os.path.exists(ckpt_root + '400.pth.tar'):
        return ckpt_list2
    else:
        return ckpt_list1


model_name = 'vgg16'
layer = 'conv'
date = '0514'
teacher_sigma_root = './KD/sigma_result/CUB_vgg16_teacher_conv_0201/'
teahcer_checkpoint_path = './KD/trained_model/CUB_vgg16_pretrain_106/'

label_sigma_root = './KD/sigma_result/ILSVRC_vgg16_label_net_conv_0415/'
distil_sigma_root = './KD/sigma_result/ILSVRC_vgg16_distil_net_conv_0415/'
label_checkpoint_path = './KD/trained_model/ILSVRC_vgg16_without_pretrain_1018/'
distil_checkpoint_path = './KD/trained_model/ILSVRC_vgg16_distil_conv_0415/'

teacher_checkpoints = choose_ckpt_list(teahcer_checkpoint_path)
label_checkpoints = choose_ckpt_list(label_checkpoint_path)
distil_checkpoints = choose_ckpt_list(distil_checkpoint_path)
dataset = 'ILSVRC'

if dataset == 'CUB':
    segdir = './Dataset/CUB/seg/train/'
if dataset == 'VOC':
    segdir = './Dataset/VOCdevkit/seg/train/seg/'

if model_name == 'alexnet':
    transform = transforms.Compose([transforms.Resize((227, 227)), transforms.ToTensor()])
else:
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

total_txt_root = './KD/knowledge_txt/' + dataset + '_' + model_name + '_' + layer + '_' + date + '_results.txt'
entropy_thresh = [-2.8]
for THRESHOLD in [0.2]:
    for counter in range(len(entropy_thresh)):
        print('\nentropy threshold: ', entropy_thresh[counter], '\n')
        for submode in ['distil_net', 'label_net']:
            distil_file_path = distil_sigma_root + '/effect_name_list.txt'
            label_file_path = label_sigma_root + '/effect_name_list.txt'

            if submode == 'teacher':
                model = model_name + '_teacher_' + layer + '_' + dataset
                sigma_root = teacher_sigma_root
                distil_file_path = sigma_root + '/effect_name_list.txt'
                checkpoints = teacher_checkpoints

            elif submode == 'label_net':
                model = model_name + '_label_net_' + layer + '_' + dataset
                sigma_root = label_sigma_root
                checkpoints = label_checkpoints

            elif submode == 'distil_net':
                model = model_name + '_distil_' + layer + '_' + dataset
                sigma_root = distil_sigma_root
                checkpoints = distil_checkpoints
            else:
                sys.exit('submode error')

            fw = open(total_txt_root, 'a')
            fw.write('***************************' + '\n')
            fw.write(model + '_'+str(entropy_thresh[counter])+ '_'+str(THRESHOLD)+'\n')
            save_path = sigma_root

            if FIND_SAME_SAMPLE:
                img_names = find_sample(label_file_path, distil_file_path)
            else:
                img_names = open(sigma_root + 'effect_name_list.txt').readlines()

            if FIND_TOTAL_BACKGROUND:
                total_background_mean, checkpoint_num, img_names = find_total_background(img_names, counter, entropy_thresh)

            if FIND_THRESHOLD:
                find_threshold(img_names, total_background_mean, checkpoint_num, counter, entropy_thresh)

            if FIND_PEAK:
                find_peak(img_names, sigma_size, THRESHOLD, PEAK_SIZE, downsample_size, KNOWLEDGE_POINT, total_background_mean, counter, entropy_thresh)

            if FIND_CONVERGENCE:
                find_convergence(img_names, sigma_size, THRESHOLD, downsample_size,
                                 total_background_mean, teahcer_checkpoint_path, label_checkpoint_path, distil_checkpoint_path, layer, submode, counter)

            if FIND_PERCENT:
                find_percent(img_names, sigma_size, THRESHOLD, downsample_size, total_background_mean, counter)

            

            fw.write('***************************' + '\n' + '\n')
            fw.close()
        fw = open(total_txt_root, 'a')
        fw.write('\n')
        fw.close()
    fw = open(total_txt_root, 'a')
    fw.write('\n' + '\n')
    fw.close()
fw.close()
