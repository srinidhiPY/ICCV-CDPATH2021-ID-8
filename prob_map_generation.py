"""
Camelyon16 Heat-map generation Script
"""
import argparse
import os
import time
import random
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import openslide
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
from util import AverageMeter
from collections import OrderedDict
from torchvision import transforms, datasets
import torch.nn.functional as F
import models.net as net
import matplotlib.pyplot as plt
import matplotlib.cm as cm


############
class DatasetTBLymph_test(Dataset):

    def __init__(self, data_path, mask_path, image_size):

            """
            Camelyon16/TBLN dataset class wrapper
                    data_path: string, path to pre-sampled images
                    json_path: string, path to the annotations in json format
            """

            self.data_path = data_path
            self.mask_path = mask_path
            self.image_size = image_size
            self.preprocess()

    def preprocess(self):

        self.mask = np.load(self.mask_path)
        self.slide = openslide.OpenSlide(self.data_path)

        X_slide, Y_slide = self.slide.level_dimensions[0]
        X_mask, Y_mask = self.mask.shape

        if round(X_slide / X_mask) != round(Y_slide / Y_mask):
            raise Exception('Slide/Mask dimension does not match ,'
                            ' X_slide / X_mask : {} / {},'
                            ' Y_slide / Y_mask : {} / {}'
                            .format(X_slide, X_mask, Y_slide, Y_mask))

        self.resolution = round(X_slide * 1.0 / X_mask)

        if not np.log2(self.resolution).is_integer():
            raise Exception('Resolution (X_slide / X_mask) is not power of 2 :'
                            ' {}'.format(self.resolution))

        # all the indices for tissue region from the tissue mask
        self.X_idcs, self.Y_idcs = np.where(self.mask)

    def __len__(self):
        return len(self.X_idcs)

    def __getitem__(self, idx):

        x_mask, y_mask = self.X_idcs[idx], self.Y_idcs[idx]

        x_center = int((x_mask) * self.resolution)
        y_center = int((y_mask) * self.resolution)

        x = int(x_center - self.image_size / 2)
        y = int(y_center - self.image_size / 2)

        img = self.slide.read_region((x, y), 0, (self.image_size, self.image_size)).convert('RGB')
        img = np.array(img, dtype=np.float32).transpose((2, 0, 1))

        return img, x_mask, y_mask


def test(args, model, classifier, test_loader):

    # switch to evaluate mode
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()

    with torch.no_grad():

        end = time.time()
        probs_map = np.zeros(test_loader.dataset.mask.shape)

        for batch_idx, (input, x_mask, y_mask) in enumerate(tqdm(test_loader, disable=False)):

            # Get inputs and target
            input = input.cuda()
            x_mask = x_mask.data.numpy()
            y_mask = y_mask.data.numpy()

            # compute output ############
            feats = model(input)
            output = classifier(feats)

            #######
            probs = torch.softmax(output, dim=1).cpu()
            probs = probs[:, -1]  # second column 'tumor'
            probs = probs.data.numpy()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print statistics and write summary every N batch
            if (batch_idx + 1) % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'BT {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_idx, len(test_loader), batch_time=batch_time))

            probs_map[x_mask, y_mask] = probs

    return probs_map


def parse_args():

    parser = argparse.ArgumentParser('Argument for Camelyon16 train/val/test heat-map predictions')

    parser.add_argument('--gpu', default='0', help='GPU id to use.')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers to use.')
    parser.add_argument('--seed', type=int, default=42, help='seed for initializing training.')

    # model definition
    parser.add_argument('--model', type=str, default='resnet18', help='choice of network architecture.')
    parser.add_argument('--num_classes', type=int, default=2, help='# of classes.')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size - 4 (32).')

    parser.add_argument('--finetune_model_path', type=str,
                        default='./models/cam_curriculum_II_finetuned_model.pt',
                        help='path to load fine-tuned model')

    # Data paths
    parser.add_argument('--tumor_train_image_pth', default='./CAMELYON16/finetune/val/tumor_val/')
    parser.add_argument('--tumor_mask_pth', default='./CAMELYON16/finetune/val/tumor_val_mask/')

    parser.add_argument('--normal_train_image_pth', default='./CAMELYON16/finetune/val/normal_val/')
    parser.add_argument('--normal_mask_pth', default='./CAMELYON16/finetune/val/normal_val_mask/')

    parser.add_argument('--probs_map_path', default='./Save_Results/')

    # Tiling parameters
    parser.add_argument('--image_size', default=256, type=int, help='patch size width 256')

    args = parser.parse_args()

    return args


#####################

def main():

    # parse the args
    args = parse_args()

    # set the model
    if args.model == 'resnet18':

        # Load fine-tuned model
        state = torch.load(args.finetune_model_path)
        model.load_state_dict(state['model'])
        classifier.load_state_dict(state['classifier'])

    else:
        raise NotImplementedError('model not supported {}'.format(args.model))

    # Load model to CUDA
    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

    ##############################################

    tr_wsipaths = []
    tr_maskpaths = []
    nr_wsipaths = []
    nr_maskpaths = []

    for file_ext in ['tif', 'svs', 'npy']:

        tr_wsipaths = tr_wsipaths + glob.glob('{}/*.{}'.format(args.tumor_train_image_pth, file_ext))
        tr_maskpaths = tr_maskpaths + glob.glob('{}/*.{}'.format(args.tumor_mask_pth, file_ext))
        tr_wsipaths, tr_maskpaths = sorted(tr_wsipaths), sorted(tr_maskpaths)

        nr_wsipaths = nr_wsipaths + glob.glob('{}/*.{}'.format(args.normal_train_image_pth, file_ext))
        nr_maskpaths = nr_maskpaths + glob.glob('{}/*.{}'.format(args.normal_mask_pth, file_ext))
        nr_wsipaths, nr_maskpaths = sorted(nr_wsipaths), sorted(nr_maskpaths)

        wsipaths = tr_wsipaths + nr_wsipaths
        maskpaths = tr_maskpaths + nr_maskpaths

    for file_ID in range(len(wsipaths)):

        wsi_pth = wsipaths[file_ID]
        mask_pth = maskpaths[file_ID]

        wsi_id = str(os.path.split(wsi_pth)[-1])
        wsi_id = os.path.splitext(wsi_id)[0]

        # Test set
        test_dataset = DatasetTBLymph_test(wsi_pth, mask_pth, args.image_size)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        #####
        n_data = len(test_dataset)
        print('number of testing samples: {}'.format(n_data))

        #################

        # Testing Model
        print("==> testing final test data...")
        probs_map = test(args, model, classifier, test_loader)

        # Save predictions
        np.save(os.path.join(args.probs_map_path, wsi_id), probs_map)

        probs_map = np.transpose(probs_map)
        predicted_img = Image.fromarray(np.uint8(probs_map * 255))
        predicted_img.save(os.path.join(args.probs_map_path, wsi_id + "." + 'png'), "PNG")
        predicted_img.close()

        # Save Heat-map
        cmapper = cm.get_cmap('jet')
        probs_heatmap = Image.fromarray(np.uint8(cmapper(np.clip(probs_map, 0, 1)) * 255))
        probs_heatmap.save(os.path.join(args.probs_map_path, wsi_id + "_" + 'heatmap' + "." + 'png'), "PNG")
        probs_heatmap.close()

        del probs_map, predicted_img



############################

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
