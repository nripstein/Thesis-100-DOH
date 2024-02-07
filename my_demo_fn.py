# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
import os
import sys
import numpy as np
import argparse
#NR ADDITION START
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
#NR ADDITION END
import pdb
import time
import cv2
import torch
# from torch.autograd import Variable
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from PIL import Image

# import torchvision.transforms as transforms
# import torchvision.datasets as dset
# from scipy.misc import imread
# from roi_data_layer.roidb import combined_roidb
# from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
# (1) here add a function to viz
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
# import pdb


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")
    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save results',
                        default="images_det")
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true',
                        default=True, required=False)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument(
        '--parallel_type',
        dest='parallel_type',
        help='which part of model to parallel, 0: all, 1: model before roi pooling',
        default=0,
        type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=8, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=132028, type=int, required=False)
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        default=True)
    parser.add_argument('--webcam_num', dest='webcam_num',
                        help='webcam ID number',
                        default=-1, type=int)
    parser.add_argument('--thresh_hand',
                        type=float, default=0.5,
                        required=False)
    parser.add_argument('--thresh_obj', default=0.5,
                        type=float,
                        required=False)
    # NR ADDITION START. Hacky solution to avoid refactoring and enable calls from python interpreter, not just command line
    import sys
    sys.argv = ['']
    del sys
    # NR ADDITION END
    args = parser.parse_args()
    return args


def zoom_image(image: np.ndarray, zoom_percentage: float) -> np.ndarray:
    """
    Zooms in on an image by a specified percentage while keeping the center fixed.

    Parameters:
        image (numpy.ndarray): The input image represented as a numpy array.
        zoom_percentage (float): The percentage by which to zoom in (e.g., 10 for 10% zoom in).

    Returns:
        numpy.ndarray: The zoomed-in image with the center fixed.
    """
    # Calculate the zoom factor based on the zoom_percentage
    scale_factor = 1 + (zoom_percentage / 100.0)

    # Calculate the new dimensions after zooming
    new_height = int(image.shape[0] * scale_factor)
    new_width = int(image.shape[1] * scale_factor)

    # Calculate the cropping box to keep the center of the image
    left = (new_width - image.shape[1]) // 2
    right = new_width - image.shape[1] - left
    top = (new_height - image.shape[0]) // 2
    bottom = new_height - image.shape[0] - top

    # Apply zoom with cropping to the center
    zoomed_image = zoom(image, zoom=(scale_factor, scale_factor, 1), mode='nearest',
                        prefilter=False)

    # Crop the zoomed image to keep the center
    zoomed_image = zoomed_image[top:-bottom, left:-right]

    return zoomed_image


lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY


def _get_image_blob(im: np.ndarray):
    """Converts an image into a network input.
    Arguments:
      im (ndarray): a color image in BGR order
    Returns:
      blob (ndarray): a data blob holding an image pyramid
      im_scale_factors (list): list of image scales (relative to im) used
        in the image pyramid
    """
    try:
        im_orig = im.astype(np.float32, copy=True)
    except Exception:
        print(type(im))
        print(im)
    # im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []
    im_scale_factors = []

    for target_size in cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
            im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, np.array(im_scale_factors)


def vis_detections_filtered_objects_PIL_NR(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01, font_path='lib/model/utils/times_b.ttf'):
    color_rgb = [(255,255,0), (255, 128,0), (128,255,0), (0,128,255), (0,0,255), (127,0,255), (255,0,255), (255,0,127), (255,0,0), (255,204,153), (255,102,102), (153,255,153), (153,153,255), (0,0,153)]
    color_rgba = [(255,255,0,70), (255, 128,0,70), (128,255,0,70), (0,128,255,70), (0,0,255,70), (127,0,255,70), (255,0,255,70), (255,0,127,70), (255,0,0,70), (255,204,153,70), (255,102,102,70), (153,255,153,70), (153,153,255,70), (0,0,153,70)]


    hand_rgb = [(0, 90, 181), (220, 50, 32)] 
    hand_rgba = [(0, 90, 181, 70), (220, 50, 32, 70)]

    obj_rgb = (255, 194, 10)
    obj_rgba = (255, 194, 10, 70)


    side_map = {'l':'Left', 'r':'Right'}
    side_map2 = {0:'Left', 1:'Right'}
    side_map3 = {0:'L', 1:'R'}
    state_map = {0:'No Contact', 1:'Self Contact', 2:'Another Person', 3:'Portable Object', 4:'Stationary Object'}
    state_map2 = {0:'N', 1:'S', 2:'O', 3:'P', 4:'F'}

    
    from PIL import Image, ImageDraw, ImageFont

    def draw_line_point(draw, side_idx, hand_center, object_center):
        draw.line([hand_center, object_center], fill=hand_rgb[side_idx], width=4)
        x, y = hand_center[0], hand_center[1]
        r=7
        draw.ellipse((x-r, y-r, x+r, y+r), fill=hand_rgb[side_idx])
        x, y = object_center[0], object_center[1]
        draw.ellipse((x-r, y-r, x+r, y+r), fill=obj_rgb)

    def filter_object(obj_dets, hand_dets):
        filtered_object = []
        object_cc_list = []
        for j in range(obj_dets.shape[0]):
            object_cc_list.append(calculate_center(obj_dets[j,:4]))
        object_cc_list = np.array(object_cc_list)
        img_obj_id = []
        for i in range(hand_dets.shape[0]):
            if hand_dets[i, 5] <= 0:
                img_obj_id.append(-1)
                continue
            hand_cc = np.array(calculate_center(hand_dets[i,:4]))
            point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])])
            dist = np.sum((object_cc_list - point_cc)**2,axis=1)
            dist_min = np.argmin(dist)
            img_obj_id.append(dist_min)
        return img_obj_id
    
    def calculate_center(bb):
        return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

    def draw_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height, font):

        mask = Image.new('RGBA', (width, height))
        pmask = ImageDraw.Draw(mask)
        pmask.rectangle(obj_bbox, outline=obj_rgb, width=4, fill=obj_rgba) 
        image.paste(mask, (0,0), mask)  

        draw.rectangle(
            [obj_bbox[0], max(0, obj_bbox[1] - 30), obj_bbox[0] + 32, max(0, obj_bbox[1] - 30) + 30],  # left, top, right, bottom
            fill=(255, 255, 255),  # Fill color of the rectangle (white in this case).
            outline=obj_rgb,        # Outline color of the rectangle (specified elsewhere in the code).
            width=4                # Width of the outline (4 pixels in this case).
        )
        draw.text((obj_bbox[0]+5, max(0, obj_bbox[1]-30)-2), f'O', font=font, fill=(0,0,0)) #

        return image


    def draw_hand_mask(image, draw, hand_idx, hand_bbox, hand_score, side, state, width, height, font):
        if side == 0:
            side_idx = 0
        elif side == 1:
            side_idx = 1
        mask = Image.new('RGBA', (width, height))
        pmask = ImageDraw.Draw(mask)
        pmask.rectangle(hand_bbox, outline=hand_rgb[side_idx], width=4, fill=hand_rgba[side_idx])
        image.paste(mask, (0, 0), mask)

        # text
        draw = ImageDraw.Draw(image)
        draw.rectangle([hand_bbox[0], max(0, hand_bbox[1]-30), hand_bbox[0]+62, max(0, hand_bbox[1]-30)+30], fill=(255, 255, 255), outline=hand_rgb[side_idx], width=4)
        draw.text((hand_bbox[0]+6, max(0, hand_bbox[1]-30)-2), f'{side_map3[int(float(side))]}-{state_map2[int(float(state))]}', font=font, fill=(0,0,0)) # 

        return image

    # convert to PIL
    im = im[:,:,::-1]
    image = Image.fromarray(im).convert("RGBA")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, size=30)
    width, height = image.size

    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4]) # left, top, right, bottom
            # print(f"type()")
            score = obj_dets[i, 4]
            if score > thresh_obj and i in img_obj_id:
                # viz obj by PIL
                print(f"BBOX! {type(bbox)}, {bbox}")
                image = draw_obj_mask(image, draw, obj_idx, bbox, score, width, height, font)

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, -1]
            state = hand_dets[i, 5]
            if score > thresh_hand:
                # viz hand by PIL
                image = draw_hand_mask(image, draw, hand_idx, bbox, score, lr, state, width, height, font)

                if state > 0:  # in contact hand

                    obj_cc, hand_cc = calculate_center(obj_dets[img_obj_id[i], :4]), calculate_center(bbox)
                    # viz line by PIL
                    if lr == 0:
                        side_idx = 0
                    elif lr == 1:
                        side_idx = 1
                    draw_line_point(draw, side_idx, (int(hand_cc[0]), int(hand_cc[1])), (int(obj_cc[0]), int(obj_cc[1])))
    elif hand_dets is not None:
        im = vis_detections(im, 'hand', hand_dets, thresh)
    return im if isinstance(im, Image.Image) else Image.fromarray(im) if isinstance(im, np.ndarray) else None
    # return im


def main(verbose=False, save_imgs=False, img_dir=None):
    contact_label_map_english = {0: 'No Contact', 1: 'Self Contact', 2: 'Other Person Contact', 3: 'Portable Object Contact', 4: 'Stationary Object Contact'}
    output_dict = {}
    output_dict_new = {
        'No Contact': [],
        'Self Contact': [],
        'Other Person Contact': [],
        'Portable Object Contact': [],
        'Stationary Object Contact': []
    }
    df_row_list = []
    args = parse_args()

    if img_dir is not None:
        args.image_dir = img_dir

    # print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda
    np.random.seed(cfg.RNG_SEED)

    # load model
    model_dir = args.load_dir + "/" + args.net + "_handobj_100K" + "/" + args.dataset
    if not os.path.exists(model_dir):
        raise Exception(
            'There is no input directory for loading network from ' +
            model_dir)
    load_name = os.path.join(
        model_dir,
        'faster_rcnn_{}_{}_{}.pth'.format(
            args.checksession,
            args.checkepoch,
            args.checkpoint))

    pascal_classes = np.asarray(['__background__', 'targetobject', 'hand'])
    args.set_cfgs = [
        'ANCHOR_SCALES',
        '[8, 16, 32, 64]',
        'ANCHOR_RATIOS',
        '[0.5, 1, 2]']

    # initilize the network here.
    if args.net == 'vgg16':
        fasterRCNN = vgg16(
            pascal_classes,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
        fasterRCNN = resnet(
            pascal_classes,
            101,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
        fasterRCNN = resnet(
            pascal_classes,
            50,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
        fasterRCNN = resnet(
            pascal_classes,
            152,
            pretrained=False,
            class_agnostic=args.class_agnostic)
    else:
        print("network is not defined")
        pdb.set_trace()

    fasterRCNN.create_architecture()

    if verbose:
        print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
        checkpoint = torch.load(load_name)
    else:
        checkpoint = torch.load(
            load_name, map_location=(
                lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    if verbose:
        print('loaded model successfully!')

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    box_info = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    with torch.no_grad():
        if args.cuda > 0:
            cfg.CUDA = True

        if args.cuda > 0:
            fasterRCNN.cuda()

        fasterRCNN.eval()

        start = time.time()
        max_per_image = 100
        thresh_hand = args.thresh_hand
        thresh_obj = args.thresh_obj
        vis = args.vis

        # print(f'thresh_hand = {thresh_hand}')
        # print(f'thnres_obj = {thresh_obj}')

        webcam_num = args.webcam_num
        # Set up webcam or get image directories
        if webcam_num >= 0:
            cap = cv2.VideoCapture(webcam_num)
            num_images = 0
        else:
            if verbose:
                print(f'image dir = {args.image_dir}')
                print(f'save dir = {args.save_dir}')
            imglist = os.listdir(args.image_dir)
            num_images = len(imglist)

        print(f'Loaded {num_images} images.')

        progress_bar = tqdm(total=num_images, desc='Processing Images')

        while (num_images >= 0):  # was >=. I think > is appropriate

            total_tic = time.time()
            if webcam_num == -1:
                num_images -= 1
                progress_bar.update(1)

            # Get image from the webcam
            if webcam_num >= 0:
                if not cap.isOpened():
                    raise RuntimeError(
                        "Webcam could not open. Please check connection.")
                ret, frame = cap.read()
                im_in = np.array(frame)
            # Load the demo image
            else:
                im_file = os.path.join(args.image_dir, imglist[num_images])
                im_in = cv2.imread(im_file)
            # bgr
            im = im_in
            # NR ADDITION START
            if im is None:
                continue
            # im = zoom_image(im, 25)
            # NR ADDITION END
            blobs, im_scales = _get_image_blob(im)
            assert len(im_scales) == 1, "Only single-image batch implemented"
            im_blob = blobs
            im_info_np = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

            im_data_pt = torch.from_numpy(im_blob)
            im_data_pt = im_data_pt.permute(0, 3, 1, 2)
            im_info_pt = torch.from_numpy(im_info_np)

            with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()
                box_info.resize_(1, 1, 5).zero_()

            # pdb.set_trace()
            det_tic = time.time()

            rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

            scores = cls_prob.data
            boxes = rois.data[:, :, 1:5]

            # extact predicted params
            contact_vector = loss_list[0][0]  # hand contact state info
            # offset vector (factored into a unit vector and a magnitude)
            offset_vector = loss_list[1][0].detach()
            lr_vector = loss_list[2][0].detach()  # hand side info (left/right)

            # get hand contact
            _, contact_indices = torch.max(contact_vector, 2)
            contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

            # get hand side
            lr = torch.sigmoid(lr_vector) > 0.5
            lr = lr.squeeze(0).float()

            if cfg.TEST.BBOX_REG:
                # Apply bounding-box regression deltas
                box_deltas = bbox_pred.data
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    # Optionally normalize targets by a precomputed mean and
                    # stdev
                    if args.class_agnostic:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda(
                            ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                        box_deltas = box_deltas.view(1, -1, 4)
                    else:
                        if args.cuda > 0:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda(
                            ) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                        else:
                            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(
                                cfg.TRAIN.BBOX_NORMALIZE_STDS) + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                        box_deltas = box_deltas.view(
                            1, -1, 4 * len(pascal_classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
            else:
                # Simply repeat the boxes, once for each class
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= im_scales[0]

            scores = scores.squeeze()
            pred_boxes = pred_boxes.squeeze()
            det_toc = time.time()
            detect_time = det_toc - det_tic
            misc_tic = time.time()
            if vis:
                im2show = np.copy(im)
            obj_dets, hand_dets = None, None
            for j in range(1, len(pascal_classes)):
                # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
                if pascal_classes[j] == 'hand':
                    inds = torch.nonzero(scores[:, j] > thresh_hand).view(-1)
                elif pascal_classes[j] == 'targetobject':
                    inds = torch.nonzero(scores[:, j] > thresh_obj).view(-1)

                # if there is det
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    if args.class_agnostic:
                        cls_boxes = pred_boxes[inds, :]
                    else:
                        cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                    cls_dets = torch.cat(
                        (cls_boxes,
                         cls_scores.unsqueeze(1),
                         contact_indices[inds],
                            offset_vector.squeeze(0)[inds],
                            lr[inds]),
                        1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_boxes[order, :],
                               cls_scores[order], cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if pascal_classes[j] == 'targetobject':
                        obj_dets = cls_dets.cpu().numpy()
                    if pascal_classes[j] == 'hand':
                        hand_dets = cls_dets.cpu().numpy()

                    # NR ADDITION START
                    if verbose:
                        print(f"----------------------{imglist[num_images]}----------------------")
                        print(pascal_classes[j])
                    contact_label_map = {0: 'N', 1: 'S', 2: 'O', 3: 'P', 4: 'F'}  # This is actual mapping
                    contact_label_map_english = {0: 'No Contact', 1: 'Self Contact', 2: 'Other Person Contact', 3: 'Portable Object Contact', 4: 'Stationary Object Contact'}
                    contact_indices_for_dets = contact_indices[inds]
                    for det_index, det in enumerate(cls_dets):
                        contact_index = contact_indices_for_dets[det_index].item()  # Get the contact index for this detection
                        contact_label = contact_label_map.get(contact_index, 'Unknown')  # Get the contact label
                        if verbose:
                            print(f'Detection {det_index}: Contact Type - {contact_label}')
                        contact_label_english = contact_label_map_english.get(contact_index, 'Unknown')
                    output_dict[imglist[num_images]] = contact_label_english  # this loses info from printed stuff, but they all seem to be repeats
                    if imglist[num_images] not in output_dict_new[contact_label_english]:
                        output_dict_new[contact_label_english].append(imglist[num_images])
                        df_row_list.append({'image': imglist[num_images], 'contact_label': contact_label_english})
                    # NR ADDITION END

            if vis:
                # visualization
                # im2show = vis_detections_filtered_objects_PIL_NR(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)  # doesnt give output for some reason
                
                # NR ADDITION START NOT DONE
                # for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
                #     bbox = list(int(np.round(x)) for x in obj_dets[i, :4]) # left, top, right, bottom
                # NR ADDITION END

                im2show = vis_detections_filtered_objects_PIL(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)

                # print(type(obj_dets), obj_dets.shape)
                # print(obj_dets)


            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            if webcam_num == -1:
                if verbose:
                    sys.stdout.write(f'im_detect: {num_images + 1}/{len(imglist)} | Detect time: {detect_time:.3f}s NMS time: {nms_time:.3f}s   \r')
                    sys.stdout.flush()

            if save_imgs:
                if vis and webcam_num == -1:
                    folder_name = args.save_dir
                    os.makedirs(folder_name, exist_ok=True)
                    result_path = os.path.join(
                        folder_name, imglist[num_images][:-4] + "_det.png")

                    im2show.save(result_path)
                else:
                    im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
                    cv2.imshow("frame", im2showRGB)
                    total_toc = time.time()
                    total_time = total_toc - total_tic
                    frame_rate = 1 / total_time
                    print('Frame rate:', frame_rate)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        # progress_bar.close()

        if webcam_num >= 0:
            cap.release()
            cv2.destroyAllWindows()
    # return output_dict_new
    return pd.DataFrame(df_row_list)


if __name__ == "__main__":
    results = main(save_imgs=False)
    # results = results.iloc[results['image'].map(lambda x: int(x.split('_')[0])).argsort()].reset_index(drop=True) # sorts by image number if I'm using my image format
    # print(results)
    # Print the counts for each unique value
    print("--------results (ran from __name__ == __main__ ----------)")
    value_counts = results['contact_label'].value_counts()
    for label, count in value_counts.items():
        print(f'{label}: {count}')
