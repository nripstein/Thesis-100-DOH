import os, json, glob, random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
random.seed(0)

def ratio2coord(ratio, width, height): 
    """
    ratio = [x1, y1, x2, y2]
    return image infos
    """

    x1, y1, x2, y2 = int(float(ratio[0])*width), int(float(ratio[1])*height), int(float(ratio[2])*width), int(float(ratio[3])*height)

    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, width)
    y2 = min(y2, height)
    
    bbox = [x1, y1, x2, y2]

    return bbox

def bbox2center(bbox):
    return (int((bbox[0]+bbox[2])/2), int((bbox[1]+bbox[3])/2))


def draw_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height, font):

    mask = Image.new('RGBA', (width, height))
    pmask = ImageDraw.Draw(mask)
    pmask.rectangle(obj_bbox, outline=obj_rgb, width=4, fill=obj_rgba) 
    image.paste(mask, (0,0), mask)  

    draw.rectangle([obj_bbox[0], max(0, obj_bbox[1]-30), obj_bbox[0]+32, max(0, obj_bbox[1]-30)+30], fill=(255, 255, 255), outline=obj_rgb, width=4)
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

# uses cv2 to avoid checkerboard. by NR
# import cv2 # uses cv2 to avoid checkerboard. by NR
# def draw_obj_mask(image, draw, obj_idx, obj_bbox, obj_score, width, height, font): # uses cv2 to avoid checkerboard. by NR
#     image = np.array(image)[:, :, ::-1].copy()
#     obj_alpha = 0.5
#     font = 0
#     # Create an empty mask with the same number of channels as the image
#     mask = np.zeros_like(image)
    
#     # Draw the filled rectangle on the mask
#     cv2.rectangle(mask, (obj_bbox[0], obj_bbox[1]), (obj_bbox[2], obj_bbox[3]), obj_rgb, thickness=cv2.FILLED)
    
#     # Overlay the mask on the image using alpha blending
#     result = cv2.addWeighted(image, 1, mask, obj_alpha, 0)
    
#     # Draw the 'O' character and the rectangle on the image
#     cv2.rectangle(result, (obj_bbox[0], max(0, obj_bbox[1] - 30)), (obj_bbox[0] + 32, max(0, obj_bbox[1] - 30) + 30), (255, 255, 255), thickness=cv2.FILLED)
#     cv2.putText(result, 'O', (obj_bbox[0] + 5, max(0, obj_bbox[1] - 30) + 25), font, 1, (0, 0, 0), thickness=2)

#     # Convert the result back to a Pillow image
#     result_pil = Image.fromarray(result[:, :, ::-1])
    
#     return result_pil

def draw_line_point(draw, side_idx, hand_center, object_center):
    draw.line([hand_center, object_center], fill=hand_rgb[side_idx], width=4)
    x, y = hand_center[0], hand_center[1]
    r=7
    draw.ellipse((x-r, y-r, x+r, y+r), fill=hand_rgb[side_idx])
    x, y = object_center[0], object_center[1]
    draw.ellipse((x-r, y-r, x+r, y+r), fill=obj_rgb)


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
