# --------------------------------------------------------
# Adopted from Tensorflow Faster R-CNN code
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
from PIL import Image
from nr_utils.bbox_draw import draw_standard_bboxes, draw_pretty_bboxes
#NR ADDITION END
import pdb
import time
import cv2
import torch
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


def parse_args():
    """
    Parse input arguments, for use if running from command line
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


def get_blue_bbox_proportion(img: np.ndarray, bbox: list, vis_debug: bool = False) -> float:
    """
    Args:
        img (np.ndarray): BGR from OpenCV
        bbox (list[int]): Of the form [left, top, right, bottom]
        vis_debug (bool): if True, a visual of the bounding box will appear
    """
    def visualize_masked_bbox(img, bbox, blue_mask):
        # Create a mask image with the same dimensions as the bounding box area
        mask_img = np.zeros((bbox[3] - bbox[1], bbox[2] - bbox[0], 3), dtype=np.uint8)
        # Fill the bounding box area in the mask with the blue mask
        mask_img[:, :, 0] = blue_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        # Apply the mask to the original image within the bounding box area
        masked_img = cv2.bitwise_and(img[bbox[1]:bbox[3], bbox[0]:bbox[2]], mask_img)
        # Display the masked image
        cv2.imshow('Blue Masked Bounding Box', masked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Convert the image to HSV color space
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Define the lower and upper bounds of the "blue" color in HSV
    LOWER_BLUE = np.array([90, 50, 50])
    UPPER_BLUE = np.array([130, 255, 255])
    # Create a mask that isolates the pixels within the specified blue range
    blue_mask = cv2.inRange(hsv_img, LOWER_BLUE, UPPER_BLUE)
    if vis_debug:
        # Call the inner function to visualize the masked bounding box
        visualize_masked_bbox(img, bbox, blue_mask)
    # Extract the bounding box area
    bbox_area = blue_mask[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    # Count the number of blue pixels within the bounding box
    blue_pixels = np.count_nonzero(bbox_area)
    # Calculate the total area of the bounding box
    total_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    # Calculate the percentage of blue pixels within the bounding box
    blue_proportion = blue_pixels / total_area
    if vis_debug:
        print(blue_pixels, total_area, blue_proportion)
    return blue_proportion


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


def extract_frames(video_path: str) -> str:
    """
    Extracts frames from video and returns their dir
    """
    # Check if the file exists
    if not os.path.isfile(video_path):
        print(f"The file {video_path} does not exist.")
        return

    # Extract the directory, video name, and extension
    video_dir, video_filename = os.path.split(video_path)
    video_name, video_ext = os.path.splitext(video_filename)

    # Define the directory to store images
    images_dir = os.path.join(video_dir, f"{video_name}_imgs")

    # Create the directory if it does not exist
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with tqdm(total=total_frames, desc=f'Extracting frames from {video_filename}') as pbar:
        while True:
            # Read a frame
            ret, frame = cap.read()

            # If frame is read correctly ret is True
            if not ret:
                break

            # Define the image filename
            image_filename = os.path.join(images_dir, f"{frame_count}_{video_name}.png")

            # Save the frame as a PNG file
            cv2.imwrite(image_filename, frame)

            frame_count += 1
            pbar.update(1)  # Update progress bar

    cap.release()
    return images_dir


def dir_or_video(path):
    """Determines if path is a video or directory which contains images"""
    def is_video_file(filename):
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.flv']
        _, file_extension = os.path.splitext(filename)
        return file_extension.lower() in video_extensions

    if os.path.isdir(path):
        return "dir"
    elif is_video_file(path):
        return "video"
    else:
        raise TypeError(f"TYPE OF FILE UNKNOWN: {type(path)}")


def main(verbose=False, save_imgs=False, img_dir=None, blue_refine=True):
    df_row_list = []
    args = parse_args()

    if img_dir is not None:
        args.image_dir = img_dir

        dir_type = dir_or_video(img_dir)
        if dir_type == "video":
            new_img_dir = extract_frames(img_dir)
            args.image_dir = new_img_dir
    from pathlib import Path
    args.save_dir = os.path.join(Path(args.image_dir).parent, "images_det")

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
            # imglist = [f for f in os.listdir(args.image_dir) if not f.startswith('.')]  # idk why there are some hidden files, but this avoids them
            num_images = len(imglist)

        if verbose:
            print(f'Loaded {num_images} images.')

        progress_bar = tqdm(total=num_images, desc='Processing Images')

        while num_images > 0:  # was >=. I think > is appropriate

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
            if hand_dets is not None:
                blue_status_arr = np.zeros(hand_dets.shape[0])  # length = number of hands
                for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
                    bbox = list(int(np.round(x)) for x in hand_dets[i, :4])  # left, top, right, bottom
                    score = hand_dets[i, 4]
                    lr = hand_dets[i, 9]
                    state = hand_dets[i, 5]
                    state_map2 = {0: 'N', 1: 'S', 2: 'O', 3: 'P', 4: 'F'}
                    side_map2 = {0: 'Left', 1: 'Right'}
                    state_map = {0: 'No Contact', 1: 'Self Contact', 2: 'Other Person Contact', 3: 'Portable Object', 4: 'Stationary Object Contact'}
                    # if the hand is touching portable object, check how much blue is in it because we might need to refine that prediction to rule out glvoed hands

                    if verbose:
                        print(f"{imglist[num_images]}: {side_map2[lr]} hand: {state_map2[state]} {score:.2f}")
                        print(bbox)

                    blue_status = "NA"
                    if blue_refine:
                        hand_percent_blue = get_blue_bbox_proportion(im, bbox=bbox) * 100
                        if hand_percent_blue > 50:
                            state = 0  # no contact
                            score = 1  # certain
                            blue_status = "blue_discard"
                    # add the data (even if it doesn't reach the threshold)

                    df_row_dict = {'frame_id': imglist[num_images],
                                   'contact_label_pred': state_map[state],
                                   'probability': int(score * 100),
                                   'bbox': bbox,
                                   'type': 'hand',
                                   'which': side_map2[lr],
                                   'other_detail': blue_status}
                    df_row_list.append(df_row_dict)

                    # if we're visualizing, we need the blue hand data included in hand_dets
                    if save_imgs:
                        blue_status_arr[i] = 1 if blue_status == "blue_discard" else 0

                # Reshape blue_status_arr to have a single column
                blue_status_arr = blue_status_arr.reshape(-1, 1)
                hand_dets = np.hstack((hand_dets, blue_status_arr))  # now blue_status = hand_dets[i, 10]

            if obj_dets is not None:
                for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
                    bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
                    score = obj_dets[i, 4]
                    df_row_dict = {'frame_id': imglist[num_images],
                                   'contact_label_pred': "NA",
                                   'probability': int(score * 100),
                                   'bbox': bbox,
                                   'type': 'obj',
                                   'which': "NA",
                                   'other_detail': 'NA'}

                    df_row_list.append(df_row_dict)

            misc_toc = time.time()
            nms_time = misc_toc - misc_tic

            if webcam_num == -1:
                if verbose:
                    sys.stdout.write(f'im_detect: {num_images + 1}/{len(imglist)} | Detect time: {detect_time:.3f}s NMS time: {nms_time:.3f}s   \r')
                    sys.stdout.flush()

            if save_imgs:
                # im2show = draw_standard_bboxes(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
                # im2show = draw_pretty_bboxes(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
                from nr_utils.bbox_draw import draw_presentation_bboxes
                im2show = draw_presentation_bboxes(im2show, obj_dets, hand_dets, thresh_hand, thresh_obj)
                im2show = Image.fromarray(cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB))  # convert to PIL image

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
    # need to use condense_dataframe() from caller.ipynb to get meaningful info out of it!
    print("--------results (ran from __name__ == __main__ ----------)")
    value_counts = results['contact_label_pred'].value_counts()
    for label, count in value_counts.items():
        print(f'{label}: {count}')
