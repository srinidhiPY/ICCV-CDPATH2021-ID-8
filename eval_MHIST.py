"""
Evaluation (testing) - MHIST dataset
"""
import argparse
import os
import time
import random
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch.backends.cudnn as cudnn

import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from util import AverageMeter
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, roc_auc_score

from dataset import DatasetMHIST_test
import models.net as net
from albumentations import Compose

#####
def test(args, model, classifier, test_loader):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    total_pred = []
    total_target = []
    total_pred_score = []

    with torch.no_grad():

        end = time.time()

        for batch_idx, (input, target) in enumerate(tqdm(test_loader, disable=False)):

            # Get inputs and target
            input, target = input.float(), target.long()

            # Move the variables to Cuda
            input, target = input.cuda(), target.cuda()

            # compute output ###############################
            feats = model(input)
            output = classifier(feats)
            pred_score = torch.softmax(output.detach_(), dim=-1)

            #######
            loss = F.cross_entropy(output, target, reduction='mean')

            # compute loss and accuracy
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)

            pred = torch.argmax(output, dim=1)
            acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

            # Save pred, target to calculate metrics
            total_pred.append(pred)
            total_target.append(target)
            total_pred_score.append(pred_score)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    batch_idx, len(test_loader), batch_time=batch_time, loss=losses, acc=acc))

        # Pred and target for performance metrics
        final_predictions = torch.cat(total_pred).to('cpu')
        final_targets = torch.cat(total_target).to('cpu')
        final_pred_score = torch.cat(total_pred_score).to('cpu')

    return final_predictions, final_targets, final_pred_score


###############
def parse_args():

    parser = argparse.ArgumentParser('Argument for MHIST evaluation script')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1, help='save frequency')
    parser.add_argument('--gpu', default='0', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='evaluation', help='test the performance')
    parser.add_argument('--num_classes', type=int, default=2, help='# of classes.')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size.')

    parser.add_argument('--model_path', type=str,
                        default='.pt',
                        help='path to load SSL pretrained model')

    # Data paths
    parser.add_argument('--image_pth', default='./MHIST/images/')
    parser.add_argument('--annotation_pth', default='./MHIST/annotations.csv')

    # Tiling parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')

    args = parser.parse_args()

    return args


#########
def main():

    # parse the args
    args = parse_args()

    # Set the data loaders (test)

    ## MHIST dataset ####

    if args.mode == 'evaluation':

        # test Set
        test_dataset = DatasetMHIST_test(args.image_pth, args.annotation_pth, args.image_size)
        print('total number of test samples in the dataset', len(test_dataset))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=args.num_workers, pin_memory=True)
    else:
        raise NotImplementedError('invalid mode {}'.format(args.mode))

    ############################################

    # set the model
    if args.model == 'resnet18':

        model = net.TripletNet_Finetune(args.model)
        classifier = net.FinetuneResNet(args.num_classes)

        if args.mode == 'evaluation':

            # Load fine-tuned model
            state = torch.load(args.model_path)
            model.load_state_dict(state['model'])

            # Load fine-tuned classifier
            classifier.load_state_dict(state['classifier'])

        else:
            raise NotImplementedError('invalid training {}'.format(args.mode))

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

####
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    final_predictions, final_targets, final_pred_score = test(args, model, classifier, test_loader)

    final_predictions = final_predictions.numpy()
    final_targets = final_targets.numpy()
    final_pred_score = final_pred_score.numpy()

    # Performance statistics of test data
    confusion_mat = confusion_matrix(final_targets, final_predictions)

    tn = confusion_mat[0, 0]
    tp = confusion_mat[1, 1]
    fp = confusion_mat[0, 1]
    fn = confusion_mat[1, 0]

    se = tp / (tp + fn)
    sp = tn / (tn + fp)
    acc = (tp + tn) / (tp + tn + fp + fn)

    auc_score = roc_auc_score(final_targets, final_pred_score[:, 1])

    # Print stats
    print('Confusion Matrix', confusion_mat)
    print('Sensitivity =', se)
    print('Specificity =', sp)
    print('Accuracy =', acc)
    print('AUC_score =', auc_score)



##################
if __name__ == "__main__":

    args = parse_args()
    print(vars(args))

    # Force the pytorch to create context on the specific device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.gpu:
            torch.cuda.manual_seed_all(args.seed)

    # Main function
    main()
