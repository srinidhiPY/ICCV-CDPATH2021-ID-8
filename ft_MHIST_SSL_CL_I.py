"""
Finetuning Self-Supervised model on MHIST dataset with Curriculum - I (i.e., easy-to-hard) stage
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

from dataset import DatasetMHIST_train
import models.net as net
from albumentations import Compose


#####
def train(args, model, classifier, train_loader, optimizer, epoch):

    """
    Fine-tuning the pre-trained SSL model
    """

    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    end = time.time()

    for batch_idx, (input, target) in enumerate(tqdm(train_loader, disable=False)):

        # Get inputs and target
        input, target = input.float(), target.long()

        # Reshape augmented tensors
        input, target = input.reshape(-1, 3, args.image_size, args.image_size), target.reshape(-1, )

        # Move the variables to Cuda
        input, target = input.cuda(), target.cuda()

        # compute output ###############################
        feats = model(input)
        output = classifier(feats)

        ###### Calculate the loss ##################
        loss = F.cross_entropy(output, target, reduction='none')

        'Sort the loss in descending order'
        loss_sorted, indices = torch.sort(loss, descending=True)

        top_k = round(args.alpha * target.size(0))   # Select top_K values for determining the hardness in mini-batch (alpha x batch_size)

        # Calculate the adaptive hardness threshold (thres as in Eq. 1 in the paper)
        a = 0.7
        b = 0.2
        thres = a*(1-(batch_idx/len(train_loader))) + b
        # print('thres', thres)
        # print('current_batch', batch_idx)
        # print('max_iteration', len(train_loader))

        # Select the hardness in each mini-batch based on the threshold (thres)
        hard_samples = loss_sorted[0:top_k]
        total_sum_hard_samples = sum(hard_samples)

        # Check whether total sum exceeds the threshold and update the loss accordingly (Eq. 2 in the paper)
        if total_sum_hard_samples > (thres * sum(loss_sorted)):
            output = output[indices, :]
            target = target[indices]
            top_k_output = output[0:top_k]
            tok_k_target = target[0:top_k]
            loss = F.cross_entropy(top_k_output, tok_k_target, reduction='mean')
            print('curriculum loss')
        else:
            loss = F.cross_entropy(output, target, reduction='mean')

        # compute gradient and do SGD step #############
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # compute loss and accuracy ####################
        batch_size = target.size(0)
        losses.update(loss.item(), batch_size)

        pred = torch.argmax(output, dim=1)
        acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print statistics and write summary every N batch
        if (batch_idx + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, batch_idx + 1, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                acc=acc))

    return losses.avg, acc.avg

#####
def validate(args, model, classifier, val_loader, criterion, epoch):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    with torch.no_grad():

        end = time.time()

        for batch_idx, (input, target) in enumerate(tqdm(val_loader, disable=False)):

            # Get inputs and target
            input, target = input.float(), target.long()

            # Reshape augmented tensors
            input, target = input.reshape(-1, 3, args.image_size, args.image_size), target.reshape(-1, )

            # Move the variables to Cuda
            input, target = input.cuda(), target.cuda()

            # compute output ###############################
            feats = model(input)
            output = classifier(feats)
            loss = criterion(output, target)

            # compute loss and accuracy ####################
            batch_size = target.size(0)
            losses.update(loss.item(), batch_size)

            pred = torch.argmax(output, dim=1)
            acc.update(torch.sum(target == pred).item() / batch_size, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % args.print_freq == 0:
                print('Val: [{0}][{1}/{2}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, batch_idx + 1, len(val_loader), batch_time=batch_time, data_time=data_time, loss=losses,
                    acc=acc))

    return losses.avg, acc.avg


#####
def parse_args():

    parser = argparse.ArgumentParser('Finetuning Self-Supervised model on MHIST dataset with Curriculum - I (i.e., easy-to-hard) stage')

    parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10, help='save frequency')
    parser.add_argument('--gpu', default='0', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--mode', type=str, default='fine-tuning', help='linear-classifier/fine-tuning/evaluation')
    parser.add_argument('--modules', type=int, default=0, help='which modules to freeze for fine-tuning the pretrained model. (full-finetune(0), fine-tune only classifier(64), layer4(45), layer3(30), layer2(15), layer1(3) - Resnet18')
    parser.add_argument('--num_classes', type=int, default=2, help='# of classes.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size - 64.')

    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate. - 1e-5(Adam)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay/weights regularizer for sgd. - 1e-4')
    parser.add_argument('--beta1', default=0.9, type=float, help='momentum for sgd, beta1 for adam.')
    parser.add_argument('--beta2', default=0.999, type=float, help=' beta2 for adam.')

    parser.add_argument('--model_path', type=str,
                        default='./models/cam_SSL_pretrained_model.pt',
                        help='path to load SSL pretrained model')
    parser.add_argument('--model_save_pth', type=str,
                        default='./Save_Results/',
                        help='path to save fine-tuned model')
    parser.add_argument('--save_loss', type=str, default='./Save_Results/',
                        help='path to save loss and other performance metrics')
    parser.add_argument('--resume', type=str, default='./Save_Results/',
                        metavar='PATH', help='path to latest checkpoint - model.pth (default: none)')

    # Data paths
    parser.add_argument('--image_pth', default='./MHIST/images/')
    parser.add_argument('--annotation_pth', default='./MHIST/annotations.csv')

    parser.add_argument('--validation_split', default=0.2, type=float, help='portion of the data that will be used for validation')

    # Parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')
    parser.add_argument('--alpha', default=0.1, type=int, help='portion of hard samples in each mini-batch to select top-K hard examples for curriculum-I fine-tuning')

    args = parser.parse_args()

    return args


#########
def main():

    # parse the args
    args = parse_args()

    # Set the data loaders (train, val)

    ## MHIST dataset ####

    if args.mode == 'fine-tuning':

        # Train set
        train_dataset = DatasetMHIST_train(args.image_pth, args.annotation_pth, args.image_size)

        # train and validation split
        num_train = len(train_dataset.datalist)
        indices = list(range(num_train))
        split = int(np.floor(args.validation_split * num_train))
        np.random.shuffle(indices)
        train_idx, val_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        print('total number of train samples in the dataset', len(train_idx))
        print('total number of val samples in the dataset', len(val_idx))


        # Data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
                                                   shuffle=True if train_sampler is None else False,
                                                   num_workers=args.num_workers, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, sampler=val_sampler,
                                                 shuffle=False, num_workers=args.num_workers, pin_memory=True)

    else:
        raise NotImplementedError('invalid mode {}'.format(args.mode))

    ############################################


    # set the model
    if args.model == 'resnet18':

        model = net.TripletNet_Finetune(args.model)

        if args.mode == 'fine-tuning':

            # original model saved file with DataParallel (Multi-GPU)
            state_dict = torch.load(args.model_path)

            # create new OrderedDict that does not contain `module.`
            new_state_dict = OrderedDict()

            for k, v in state_dict['model'].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            # Load pre-trained model
            print('==> loading pre-trained model')
            model.load_state_dict(new_state_dict)

            # look at the contents of the model and its parameters
            idx = 0
            for layer_name, param in model.named_parameters():
                print(layer_name, '-->', idx)
                idx += 1

            # Freezing the specific layer weights in the model and fine tune it
            for name, param in enumerate(model.named_parameters()):
                if name < args.modules:  # No of layers(modules) to be freezed
                    print("module", name, "was frozen")
                    param = param[1]
                    param.requires_grad = False
                else:
                    print("module", name, "was not frozen")
                    param = param[1]
                    param.requires_grad = True

            print('==> finetuning classification')
            classifier = net.FinetuneResNet(args.num_classes)

            # Multi-GPU
            if torch.cuda.is_available() and torch.cuda.device_count() > 1:
                model = torch.nn.DataParallel(model)
                classifier = torch.nn.DataParallel(classifier)

        else:
            raise NotImplementedError('invalid training {}'.format(args.mode))

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

##########

    # loss fn
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    # Optimiser & scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters()) + list(classifier.parameters())), lr=args.lr,
                           betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800], gamma=0.95)

    # Training Model
    start_epoch = 1
    best_val_acc = -1

    'check resume from a checkpoint'
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            classifier.load_state_dict(checkpoint['classifier'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint['val_acc']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Start log (writing into XL sheet)
    with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'w') as f:
        f.write('epoch, train_loss, train_acc, val_loss, val_acc\n')

    # Routine
    for epoch in range(start_epoch, args.num_epoch + 1):

        if args.mode == 'fine-tuning':

            print("==> fine-tuning the pretrained SSL model...")

            time_start = time.time()

            train_losses, train_acc = train(args, model, classifier, train_loader, optimizer, epoch)
            print('Epoch time: {:.2f} s.'.format(time.time() - time_start))

            print("==> validating the fine-tuned model...")
            val_losses, val_acc = validate(args, model, classifier, val_loader, criterion, epoch)

            # Log results
            with open(os.path.join(args.save_loss, 'fine_tuned_results.csv'), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,\n' % ((epoch + 1), train_losses, train_acc, val_losses, val_acc))

            'adjust learning rate --- Note that step should be called after validate()'
            scheduler.step()

            # Save model every 10 epochs
            if epoch % args.save_freq == 0:
                print('==> Saving...')
                state = {
                    'args': args,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_losses
                }
                torch.save(state, '{}/fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))

            # Save model for the best val
            if val_acc > best_val_acc:
                print('==> Saving...')
                state = {
                    'args': args,
                    'model': model.state_dict(),
                    'classifier': classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_losses,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'val_loss': val_losses
                }
                torch.save(state, '{}/best_fine_tuned_model_{}.pt'.format(args.model_save_pth, epoch))
                best_val_acc = val_acc

                # help release GPU memory
                del state

            torch.cuda.empty_cache()

        else:
            raise NotImplementedError('mode not supported {}'.format(args.mode))


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
