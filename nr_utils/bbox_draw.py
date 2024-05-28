from model.utils.net_utils import filter_object
import numpy as np
from bounding_box import bounding_box as bb


def draw_bbox(img, coords, label, color: str) -> np.ndarray:
    # bb.add(image, left, top, right, bottom, label, color)
    bb.add(img, coords[0], coords[1], coords[2], coords[3], label, color)
    return img


def bbox_area(bbox: list) -> int:
    """
    Calculate the area of a bounding box.

    Args:
    - box (list): A list containing the coordinates of the bounding box in the format [left, top, right, bottom].
    """
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    return width * height


def draw_standard_bboxes(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01):
    side_map3 = {0: 'L', 1: 'R'}
    state_map2 = {0: 'N', 1: 'S', 2: 'O', 3: 'P', 4: 'F'}

    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        for obj_idx, i in enumerate(range(np.minimum(10, obj_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            if score > thresh_obj and i in img_obj_id:
                im = draw_bbox(im, bbox, "O", "yellow")

        for hand_idx, i in enumerate(range(np.minimum(10, hand_dets.shape[0]))):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, 9]
            state = hand_dets[i, 5]
            if score > thresh_hand:
                color = "red" if side_map3[lr] == "R" else "green"
                im = draw_bbox(im, bbox, f"{side_map3[lr]}-{state_map2[state]}", color)
    return im


def draw_pretty_bboxes(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01):
    """
    Colors better than default

    - Doesn't draw bounding boxes for objects which take up 50+% of the frame.
    - Only puts a bounding box on the smallest object.
    """
    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        smallest_obj_idx = None
        smallest_obj_area = float('inf')

        for i in range(np.minimum(10, obj_dets.shape[0])):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # Check conditions: score threshold, object ID filter, and not too large
            if score > thresh_obj and i in img_obj_id:
                if obj_area < smallest_obj_area and obj_area < 0.5 * im.shape[0] * im.shape[1]:
                    smallest_obj_area = obj_area
                    smallest_obj_idx = i

        # Draw bbox for the smallest object
        if smallest_obj_idx is not None:
            bbox = list(int(np.round(x)) for x in obj_dets[smallest_obj_idx, :4])
            im = draw_bbox(im, bbox, "O", "yellow")

        # Process hand detections as before
        for i in range(np.minimum(10, hand_dets.shape[0])):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, 9]
            state = hand_dets[i, 5]
            if score > thresh_hand:
                color = "navy"  # Assuming side_map2 is defined elsewhere like side_map3 in the initial function
                im = draw_bbox(im, bbox, f"{side_map2[lr]}-{state_map2[state]}", color)

    return im


def draw_presentation_bboxes(im, obj_dets, hand_dets, thresh_hand=0.8, thresh_obj=0.01):
    """
    Colors better than default

    - Doesn't draw bounding boxes for objects which take up 50+% of the frame.
    - Only puts a bounding box on the smallest object.
    """

    # we want to draw smallest obj if takes up less than hand size! not yet added

    if (obj_dets is not None) and (hand_dets is not None):
        img_obj_id = filter_object(obj_dets, hand_dets)
        smallest_obj_idx = None
        smallest_obj_area = float('inf')

        for i in range(np.minimum(10, obj_dets.shape[0])):
            bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
            score = obj_dets[i, 4]
            obj_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            # Check conditions: score threshold, object ID filter, and not too large
            if score > thresh_obj and i in img_obj_id:
                if obj_area < smallest_obj_area and obj_area < 0.5 * im.shape[0] * im.shape[1]:
                    smallest_obj_area = obj_area
                    smallest_obj_idx = i

        # Draw bbox for the smallest object
        if smallest_obj_idx is not None:
            smallest_obj_bbox = list(int(np.round(x)) for x in obj_dets[smallest_obj_idx, :4])

            im = draw_bbox(im, smallest_obj_bbox, "Object", "aqua")

        # Process hand detections as before
        for i in range(np.minimum(10, hand_dets.shape[0])):
            bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
            score = hand_dets[i, 4]
            lr = hand_dets[i, 9]
            state = hand_dets[i, 5]
            blue_status = hand_dets[i, 10]
            # print(blue_status)
            if score > thresh_hand:
                if blue_status == 1:  # Check if the blue status indicates a discard
                    color = "silver"
                    label = "Experimenter"
                else:
                    color = "black"  # Normal case without blue status
                    label = f"{side_map2[lr]}"
                
                im = draw_bbox(im, bbox, label, color)

    return im


side_map = {'l': 'Left', 'r': 'Right'}
side_map2 = {0: 'Left', 1: 'Right'}
side_map3 = {0: 'L', 1: 'R'}
state_map = {0: 'No Contact', 1: 'Self Contact', 2: 'Another Person', 3: 'Portable Object', 4: 'Stationary Object'}
state_map2 = {0: 'N', 1: 'S', 2: 'O', 3: 'P', 4: 'F'}
