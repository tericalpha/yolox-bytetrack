from ast import arg
from collections import deque
import sys
from venv import create

sys.path.insert(0, './YOLOX')
import torch
import numpy as np
import cv2
import time
from utils.couting import *
from YOLOX.yolox.data.data_augment import preproc
from YOLOX.yolox.data.data_augment import ValTransform
from YOLOX.yolox.data.datasets import COCO_CLASSES
from YOLOX.yolox.exp.build import get_exp_by_name,get_exp_by_file
from YOLOX.yolox.utils import postprocess
from utils.visualize import vis
from YOLOX.yolox.utils.visualize import plot_tracking
from YOLOX.yolox.tracker.byte_tracker import BYTETracker
from torch2trt import TRTModule

COCO_MEAN = (0.485, 0.456, 0.406)
COCO_STD = (0.229, 0.224, 0.225)

class Detector():
    """ 图片检测器 """
    def __init__(self, model=None, ckpt=None):
        super(Detector, self).__init__()

        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        print("device = ",self.device)
        self.cls_names = COCO_CLASSES

        self.preproc = ValTransform(legacy=False)
        self.exp = get_exp_by_name(model)
        self.exp.test_size = (640,640)

        self.test_size = self.exp.test_size  # TODO: 改成图片自适应大小
        self.model = self.exp.get_model()
        self.model.to(self.device)
        self.model.eval()
        checkpoint = torch.load(ckpt, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])

        # self.trt_file = "YOLOX/YOLOX_outputs/yolox_s/model_trt.pth"
        self.model.head.decode_in_inference = False
        self.decoder = self.model.head.decode_outputs

        # self.load_modelTRT()

    def load_modelTRT(self):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(self.trt_file))
        x = torch.ones(1, 3, self.exp.test_size[0], self.exp.test_size[1]).cuda()
        self.model(x)
        self.model = model_trt

    def detect(self, img):
        img_info = {"id": 0}

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img, _ = self.preproc(img, None, self.test_size)

        img_info["ratio"] = ratio
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.exp.num_classes,  self.exp.test_conf, self.exp.nmsthre,
                class_agnostic=True
            )
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        info = {}
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            info['boxes'], info['scores'], info['class_ids'],info['box_nums']=None,None,None,0
            return img,info

        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        info['boxes'] = bboxes
        info['scores'] = scores
        info['class_ids'] = cls
        info['box_nums'] = output.shape[0]

        return vis_res,info

class Args():
    def __init__(self) -> None:
        self.track_thresh = 0.4
        self.track_buffer = 30
        self.match_thresh = 0.8
        self.aspect_ratio_thresh = 1.6
        self.min_box_area = 10
        self.mot20 = False
        self.tsize = None
        self.name = 'yolox-s'
        self.ckpt = 'epoch_34_ckpt.pth'
        self.exp_file = None
        
def is_crossing_line(tracked_object, line_point1, line_point2):
    # Check if the object's bounding box crosses the specified line
    x1, y1 = line_point1
    x2, y2 = line_point2

    tlwh = tracked_object.tlwh
    x_left = int(tlwh[0])
    y_top = int(tlwh[1])
    x_right = int(tlwh[0] + tlwh[2])
    y_bottom = int(tlwh[1] + tlwh[3])

    return (x_left < x2 and x_right > x1 and ((y1 - y_top) * (x2 - x1) - (x1 - x_left) * (y2 - y1)) * (
            (y1 - y_bottom) * (x2 - x1) - (x1 - x_right) * (y2 - y1)) < 0)

if __name__=='__main__':
    args = Args()
    detector = Detector(model=args.name,ckpt=args.ckpt)
    tracker = BYTETracker(args, frame_rate=22)
    exp = get_exp_by_name(args.name)

    cap = cv2.VideoCapture('2.mp4')  # open one video

    # VideoWriter settings
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('yolox-person-output_video.avi', fourcc, 35.0, (1920, 1080))

    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    frame_id = 0
    results = []
    fps = 0

    # create filter class
    filter_class = [0]

    # init variable for counting object
    memory = {}
    angle = -1
    in_count = 0
    out_count = 0
    already_counted = deque(maxlen=50)

    crossing_counter = 0
    crossed_ids = set()

    while True:
        _, im = cap.read() # read frame from video

        if im is None:
            break
        
        outputs, img_info = detector.detect(im)

        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, filter_class)

            # draw line for couting object
            line = [(0, 600), (int(im.shape[1]), 600)]
            # print(line)
            # line = [(0, 0), (int(im.shape[1]), int(im.shape[0]))]
            cv2.line(im, line[0], line[1], (0, 255, 0), 2)

            online_tlwhs = []
            online_ids = []
            online_scores = []
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id                
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f},{ t.score:.2f}, -1, -1, -1\n")
                # print(online_tlwhs)
                # Check if the object crosses the line
                 # Check if the object crosses the line
                if is_crossing_line(t, line[0], line[1]) and tid not in crossed_ids:
                    in_count += 1  # Assuming crossing from top to bottom
                    crossed_ids.add(tid)
                    print(f"Object crossed the line! Counter: {in_count}, Tracker ID: {tid}")

                    # Highlight the line-crossing event (e.g., change color)
                    cv2.putText(im, "Crossing!", (int(tlwh[0]), int(tlwh[1] - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # results.append(f"{frame_id}, {tid}, {tlwh[0]:.2f}, {tlwh[1]:.2f}, {tlwh[2]:.2f}, {tlwh[3]:.2f}, {t.score:.2f}, -1, -1, -1\n")
            # print(online_tlwhs)
            
                
                    
            online_im = plot_tracking(img_info['raw_img'], online_tlwhs, online_ids, frame_id=frame_id + 1,fps=fps, in_count=in_count, out_count=out_count)
        else:
            online_im = img_info['raw_img']

        # font which we will be using to display FPS
        font = cv2.FONT_HERSHEY_SIMPLEX
        # time when we finish processing for this frame
        new_frame_time = time.time()
    
        # Calculating the fps
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
    
        online_im = cv2.resize(online_im,(1920,1080))
        out.write(online_im)  # Write frame to the output video

        cv2.imshow('demo', online_im)	# imshow

        cv2.waitKey(1)
        if cv2.getWindowProperty('demo', cv2.WND_PROP_AUTOSIZE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()
