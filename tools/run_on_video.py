# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os
import sys
import argparse

import cv2

import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

import _init_paths
import models
from config import config
from config import update_config
from utils.drawing import get_colored_frame


def parse_args():
    parser = argparse.ArgumentParser(description='segmentation HRNetV2 inference')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    update_config(config, args)

    return args


class VideoSequence:
    def __init__(self, sequence_directory):
        self.images = sorted(os.listdir(sequence_directory))
        self.images = list(map(lambda x: os.path.join(sequence_directory, x), self.images))
        self.indexer = 0
        self.len = len(self.images)

    def read(self):
        if self.indexer < self.len:
            ret = True
            img = cv2.imread(self.images[self.indexer])
            self.indexer += 1
        else:
            ret = False
            img = None
        return ret, img


def main():
    parse_args()

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    if config.TEST.MODEL_FILE:
        model_state_file = config.TEST.MODEL_FILE
    else:
        print('weights model file has not been set!')
        print('Please, update your config file')
        sys.exit(1)

    model = getattr(models, config.MODEL.NAME)(config)
    model.init_weights(model_state_file)
    model = model.cuda()

    # prepare data
    test_size = (config.TEST.IMAGE_SIZE[1], config.TEST.IMAGE_SIZE[0])

    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize(test_size, interpolation=2),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    if config.TEST.VIDEO_IS_SEQUENCE:
        cap = VideoSequence(config.TEST.VIDEO_DIR)
    else:
        cap = cv2.VideoCapture(config.TEST.VIDEO_DIR)

    model.eval()
    overlay_coef = 0.5
    ret = True
    with torch.no_grad():
        while ret:
            ret, frame = cap.read()
            base_img = frame.copy()

            frame = transform(frame)
            frame = torch.unsqueeze(frame, dim=0)
            frame = frame.cuda()
            output = model(frame)

            predicted = get_colored_frame(frame=base_img, predicted=output, overlay_coef=overlay_coef)

            cv2.imshow('video', predicted)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                ret = False

    print('Done')


if __name__ == '__main__':
    main()
