# Pytorch
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms as T

import os
import cv2
import numpy as np
from PIL import Image

# ZED
import pyzed.sl as sl

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

zed_camera = sl.Camera()

class_list = list()
color_list = list()
threshold=0.7

input_image = np.ndarray
output_image = np.ndarray
output_mask = torch.tensor([])
output_model = torch.tensor([])
output_tracking = list()

def calc_IOU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def get_names(path):
    class_list.clear()
    color_list.clear()

    with open(path,'r') as f:
        lines = f.readlines()

        for line in lines:
            class_list.append(line.replace("\n",""))
            color = (np.random.randint(255),np.random.randint(255),np.random.randint(255))
            color_list.append(color)
    
def get_model(path):
    num_classes = len(class_list)+1
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                    hidden_layer,
                                                    num_classes)

    model.to(device)

    # load trained model
    if os.path.isfile(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("load pretrained model:",path)
    else:
        if not path == "":
            print("cannot find pretrained model:",path)
    
    model.eval()

def initialize(model_path,label_path):
    # ZED camera initialize
    zed_init_params = sl.InitParameters()
    zed_init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    zed_init_params.camera_resolution = sl.RESOLUTION.HD720
    zed_init_params.coordinate_units = sl.UNIT.METER
    zed_init_params.sdk_gpu_id = -1

    err = zed_camera.open(zed_init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(repr(err))
        exit()
    
    get_names(label_path)
    get_model(model_path)
    
def get_image():
    if (zed_camera.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS):
        global input_image

        image = sl.Mat()
        zed_camera.retrieve_image(image)

        image_cv = image.get_data()
        image_cv = cv2.cvtColor(image_cv,cv2.COLOR_RGBA2RGB)

        input_image = image_cv
                
def forward():
    transform = T.Compose([T.ToTensor()])

    img = Image.fromarray(input_image)
    img = transform(img)

    img = img.unsqueeze(0)
    img = img.to(device)

    global output_model
    output_model = model(img)[0]

def draw_output():
    height,width,_ = input_image.shape
    
    img = input_image*.7
    img = img.astype('uint8')

    boxes = output_model['boxes']
    labels = output_model['labels']
    masks = output_model['masks']
    scores = output_model['scores']

    num_obj = labels.shape[0]

    total_masks = {}

    for class_name in class_list:
        total_masks[class_name] = np.zeros((height,width),dtype='uint8')

    for i in range(num_obj):
        bbox = boxes[i].detach().cpu().numpy()
        bbox = [int(x) for x in bbox]
        label = labels[i]
        mask = masks[i]
        score = scores[i].item()
        t_id = int()
        
        for j in range(len(output_tracking)):
            track_result = output_tracking[j]
            track_obj_id, track_prob, track_x, track_y, track_w,track_h,track_id = track_result
            track_bbox = [track_x,track_y,track_x+track_w,track_y+track_h]
            track_bbox = [int(x) for x in track_bbox]
            iou = calc_IOU(bbox,track_bbox)
            
            if iou >= threshold and label.item() == track_obj_id:
                bbox = track_bbox
                score = track_prob
                t_id = int(track_id)

        if score >= threshold:
            x1,y1,x2,y2 = bbox
            
            mask = mask.squeeze().detach().cpu().numpy()
            mask[mask>=threshold] = 1
            mask[mask<threshold] = 0
            mask = np.array(mask,dtype='uint8')
            
            color = color_list[int(label.item())-1]
            class_name = class_list[int(label.item())-1]
            
            total_masks[class_name] += mask

            cv2.rectangle(img,(x1,y1),(x2,y2),color,1)
            
            if len(output_tracking) == 0:
                cv2.putText(img,class_name+":"+str(round(score*100))+"%",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
            else:
                cv2.putText(img,"ID:"+str(t_id),(x1,y1-40),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
                cv2.putText(img,class_name+":"+str(round(score*100))+"%",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,1,color,2)
         
    for cnt in range(len(class_list)):
        class_name = class_list[cnt]
        mask = total_masks[class_name]
        
        color = color_list[cnt]
        mask[mask>=1] = 1

        r = mask*color[0]
        g = mask*color[1]
        b = mask*color[2]

        mask = cv2.merge([r,g,b])*.3

        img += mask.astype('uint8')

    global output_image
    output_image = img

def toss_output():
    output = output_model
    
    total_output = list()

    boxes = output['boxes']
    labels = output['labels']
    masks = output['masks']
    scores = output['scores']

    num_obj = labels.shape[0]

    for i in range(num_obj):
        bbox = boxes[i]
        label = labels[i]
        score = scores[i].item()

        if score >= threshold:
            x1,y1,x2,y2 = bbox

            info = [float(label),float(score),float(x1),float(y1),float(x2),float(y2)]
            total_output.append(info)
        
    return total_output

def inference():
    get_image()
    forward()
    
    total_output = toss_output()
    
    return total_output

def take_bbox(bboxes):
    global output_tracking
    output_tracking = bboxes
    
def show():
    draw_output()

    cv2.imshow("Instance seg. result",output_image)
    k = cv2.waitKey(1)
    
    # ESC
    if k==27:
        return 0
    else:
        return 1