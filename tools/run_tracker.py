# encoding: utf-8
# Copyright (c) Lian Zhang and its affiliates.

import argparse
import os
import os.path as osp
import time
import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

from loguru import logger
from unismot.data.data_augment import preproc
from unismot.exp import get_exp
from unismot.utils import fuse_model, get_model_info, postprocess
from unismot.utils.visualize import plot_tracking
from unismot.tracker.unismot_tracker import UnismotTracker
from unismot.tracking_utils.timer import Timer
from unismot.tracker.position_encoding import build_position_encoding

seed = 31
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("UnismotTrack Demo!")
    parser.add_argument(
        "--demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        # ./datasets/UniRTL/MOT/train
        # ./datasets/RGBT_mix/GTOT
        # ./datasets/RGBT_mix/LasHeR
        "-pt", "--path", default="./datasets/UniRTL/SOT/train", help="path to images or video"
    )
    parser.add_argument(
        "-tm",
        "--track_mode",
        default='SOT',
        help="Tracking node: SOT or MOT",
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--save_image_result",
        default=False,
        action="store_true",
        help="whether to save the inference result of jpg",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='exps/example/mot/unismot_l_RGBTL2.py',
        type=str,
        help="pls input your expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default='pretrained/unismot_l_RGBTL2.pth.tar', type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=None, type=float, help="test conf")
    parser.add_argument("--nms", default=None, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument("--fps", default=25, type=int, help="frame rate (fps)")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=True,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=True,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--nlow",
        dest="nlow",
        default=False,
        action="store_true",
        help="Add low light channels for training.",
    )
    parser.add_argument(
        "--use_FTP",
        dest="use_FTP",
        default=True,
        action="store_true",
        help="Add FTP module for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    # tracking args   yuanshi 0.5
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=40, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=100,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    ) # 1.6

    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    parser.add_argument("--ReID_folder", type=str, default='pretrained/ckpt.t7', help="reid model folder")
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names

def genConfig(seq_path, set_type):

    path, seqname = os.path.split(seq_path)


    if set_type == 'SOT':
        ############################################  have to refine #############################################
        if 'GTOT' in path:
            RGB_img_list = sorted(
                [seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if os.path.splitext(p)[1] == '.png'])
            T_img_list = sorted(
                [seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if os.path.splitext(p)[1] == '.png'])
            l_img_list = T_img_list
            RGBT_gt = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')
            x_min = np.min(RGBT_gt[:, [0, 2]], axis=1)[:, None]
            y_min = np.min(RGBT_gt[:, [1, 3]], axis=1)[:, None]
            x_max = np.max(RGBT_gt[:, [0, 2]], axis=1)[:, None]
            y_max = np.max(RGBT_gt[:, [1, 3]], axis=1)[:, None]
            RGBT_convert_gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)
            frame0 = RGBT_convert_gt[0]
        elif 'RGB-T234' in path:
            RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if
                                   os.path.splitext(p)[1] == '.jpg'])
            T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if
                                 os.path.splitext(p)[1] == '.jpg'])
            l_img_list = T_img_list
            RGBT_convert_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')
            frame0 = RGBT_convert_gt[0]
        elif 'LasHeR' in path:
            RGB_img_list = sorted([seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if
                                   os.path.splitext(p)[1] == '.jpg'])
            T_img_list = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if
                                 os.path.splitext(p)[1] == '.jpg'])
            l_img_list = T_img_list
            RGBT_convert_gt = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')
            frame0 = RGBT_convert_gt[0]
        elif 'UniRTL' in path:
            RGB_img_list = sorted(
                [seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if os.path.splitext(p)[1] == '.jpg'])
            T_img_list = sorted(
                [seq_path + '/IR/' + p for p in os.listdir(seq_path + '/IR') if os.path.splitext(p)[1] == '.jpg'])
            l_img_list = sorted(
                [seq_path + '/low/' + p for p in os.listdir(seq_path + '/low') if os.path.splitext(p)[1] == '.jpg'])

            RGBT_gt = np.loadtxt(seq_path + '/gt/' + '/gt_sot.txt', delimiter=',')
            RGBT_convert_gt = RGBT_gt[:, 2:6]
            frame0 = RGBT_gt[0][2:6]

    elif set_type == 'MOT':
        RGB_img_list = sorted(
            [seq_path + '/rgb/' + p for p in os.listdir(seq_path + '/rgb') if os.path.splitext(p)[1] == '.jpg'])
        T_img_list = sorted(
            [seq_path + '/IR/' + p for p in os.listdir(seq_path + '/IR') if os.path.splitext(p)[1] == '.jpg'])
        l_img_list = sorted(
            [seq_path + '/low/' + p for p in os.listdir(seq_path + '/low') if os.path.splitext(p)[1] == '.jpg'])

        RGBT_gt = np.loadtxt(seq_path + '/gt/' + '/gt_mot.txt', delimiter=',')
        RGBT_convert_gt = RGBT_gt[:, 2:6]
        frame0 = None

    return RGB_img_list, T_img_list, l_img_list, RGBT_convert_gt, frame0

def write_results(filename, results):
    # save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n' # for MOT
    save_format = '{x1},{y1},{w},{h}\n' # for SOT
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                #line = save_format.format(frame=frame_id, id=track_id, x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1), h=round(h, 1), s=round(score, 2))
                line = save_format.format(x1=round(x1, 1), y1=round(y1, 1), w=round(w, 1),h=round(h, 1))
                f.write(line)
    logger.info('save results to {}'.format(filename))

def get_label_map(boxes_gt, H, W, bs, device="cuda"):
    """boxes: (bs, 4)"""
    boxes = boxes_gt.copy()
    boxes[2:] = boxes[2:] + boxes[:2]
    labels = torch.zeros((bs, 1, H, W), dtype=torch.float32, device=device)
    x1, y1, x2, y2 = boxes.astype(int)
    x1, y1 = max(0, x1), max(0, y1)
    try:
        labels[0, 0, y1:y2, x1:x2] = 1.0
    except:
        print("too small bounding box")
        pass
    return labels # (bs, 1, H, W)

class UnismotPredictor(object):
    def __init__(
        self,
        model,
        exp,
        trt_file=None,
        decoder=None,
        device=torch.device("cpu"),
        fp16=False,
        use_FTP=True
    ):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16

        self.nlow = exp.nlow
        self.pos_emb = build_position_encoding()
        self.use_FTP = use_FTP
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]), device=device)
            self.model(x)
            self.model = model_trt

        # for IRlow -= mean
        # self.rgb_means = (0.309, 0.309, 0.309)
        # self.std = (0.250, 0.250, 0.250)

        # for RGBT234
        # self.rgb_means = (0.576, 0.576, 0.554)
        # self.std = (0.240, 0.238, 0.245)

        # # for LasHeR without Hist()
        # self.rgb_means = (0.456, 0.456, 0.425),
        # self.std = (0.281, 0.280, 0.286),

        # for IRrgb
        self.rgb_means = (0.403, 0.409, 0.375)
        self.std = (0.164, 0.161, 0.162)


    def inference(self, imgi, imgv, imgl, timer, frameID, gt_0):
        img_info = {"id": 0}
        if isinstance(imgi, str):
            img_info["file_IR_name"] = osp.basename(imgi)
            img_info["file_low_name"] = osp.basename(imgl)
            img_info["file_rgb_name"] = osp.basename(imgv)
            imgi = cv2.imread(imgi)
            imgl = cv2.imread(imgl)
            imgv = cv2.imread(imgv)
        else:
            img_info["file_name"] = None

        height, width = imgi.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_imgi"] = imgi
        img_info["raw_imgl"] = imgi
        img_info["raw_imgv"] = imgi

        imgi, imgl, imgv, ratio = preproc(imgi, imgl, imgv, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        if self.nlow:
            imgi = torch.from_numpy(imgi).unsqueeze(0).float().to(self.device)
            imgl = torch.from_numpy(imgl).unsqueeze(0).float().to(self.device)
            imgv = torch.from_numpy(imgv).unsqueeze(0).float().to(self.device)
        else:
            imgi = torch.from_numpy(imgi).unsqueeze(0).float().to(self.device)
            #################################################################################################################
            imgv = torch.from_numpy(imgv).unsqueeze(0).float().to(self.device)
            #################################################################################################################
        if self.fp16 and self.nlow:
            imgi = imgi.half()  # to FP16
            imgl = imgl.half()
            imgv = imgv.half()
        else:
            imgi = imgi.half()
            imgv = imgv.half()

        # FTP, position embedding for the current image
        FTP_RLM = gt_0
        img_info['FTP_init'] = FTP_RLM
        img_info['use_FTP'] = self.use_FTP
        if frameID == 1:
            H_d, W_d = img_info["height"]//8, img_info["width"]//8
            if FTP_RLM is not None:
                """get the positional encoding"""
                lable_map = get_label_map(FTP_RLM, img_info["height"], img_info["width"], bs=1, device="cuda")
                pos = self.pos_emb(1, img_info["height"], img_info["width"])  # (B, C, H, W)
                """Interpolate positional encoding according to input size"""
                pos = F.interpolate(pos, scale_factor=1/8, mode="bilinear", align_corners=False).flatten(-2).cuda()
                pos = pos.expand(-1, pos.size(-1), pos.size(-1))
                lable_pos = F.interpolate(lable_map, scale_factor=1/8, mode="bilinear", align_corners=False).flatten(-2).cuda()
                emb_feat = torch.bmm(lable_pos, pos).view(1, 1, H_d, W_d)
            else:
                emb_feat = torch.zeros((1, 1, H_d, W_d), device="cuda")
            feat_emb = (emb_feat,
                        F.interpolate(emb_feat, scale_factor=1 / 2, mode="bilinear", align_corners=False),
                        F.interpolate(emb_feat, scale_factor=1 / 4, mode="bilinear", align_corners=False))  # [8, 16, 32]
        else:
            feat_emb =None
        if not self.use_FTP:
            feat_emb = None


        with torch.no_grad():
            timer.tic()
            if self.nlow:
                outputs = self.model(imgi, imgv, imgl, targets=None, FTP=feat_emb)
            else:
                outputs = self.model(imgi, imgv, xl=None, targets=None, FTP=feat_emb)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

def border_limit(pre_tlwh, image_h, image_w):
    pre_tlbr = [0, 0, 0, 0]
    pre_tlbr[0] = pre_tlwh[0]
    pre_tlbr[1] = pre_tlwh[1]
    pre_tlbr[2] = pre_tlwh[0] + pre_tlwh[2]
    pre_tlbr[3] = pre_tlwh[1] + pre_tlwh[3]

    pre_tlbr[0] = max(1, min(image_w - 10, pre_tlbr[0]))
    pre_tlbr[2] = max(pre_tlbr[0] + 10, min(image_w, pre_tlbr[2]))
    pre_tlbr[1] = max(1, min(image_h - 10, pre_tlbr[1]))
    pre_tlbr[3] = max(pre_tlbr[1] + 10, min(image_h, pre_tlbr[3]))

    pre_tlwh[0] = pre_tlbr[0]
    pre_tlwh[1] = pre_tlbr[1]
    pre_tlwh[2] = pre_tlbr[2] - pre_tlbr[0]
    pre_tlwh[3] = pre_tlbr[3] - pre_tlbr[1]

    return pre_tlwh, pre_tlbr

def image_demo(predictor, vis_folder, current_time, args, seq):
    seq_path = args.path + '/' + seq
    if osp.isdir(seq_path):
        # files = get_image_list(args.path)
        RGB_img_list, T_img_list, L_img_list, RGBT_gt, frame0_gt = genConfig(seq_path, args.track_node)
    else:
        files = [args.path]
    RGB_img_list.sort()
    T_img_list.sort()
    L_img_list.sort()
    # files.sort()
    tracker = UnismotTracker(args, frame_rate=args.fps, use_cuda=True)
    timer = Timer()
    results = []

    if frame0_gt is not None:
        gt_tlbr = frame0_gt.copy()
        gt_tlbr[2:] = gt_tlbr[2:] + gt_tlbr[:2]
    else:
        gt_tlbr = RGBT_gt[0].copy()
        gt_tlbr[2:] = gt_tlbr[2:] + gt_tlbr[:2]
    pre_results = None
    start_results = None
    id_none = 0
    if args.track_node == 'MOT':
        frame0_gt = None
    res_file = osp.join(vis_folder, f"{seq}.txt")
    if not osp.exists(res_file):
        for frame_id, img_path in enumerate(zip(T_img_list, RGB_img_list, L_img_list), 1):

            imgi_path = img_path[0]
            imgv_path = img_path[1]
            imgl_path = img_path[2]
            pre_tlbrs = []
            pre_scores = []

            outputs, img_info = predictor.inference(imgi_path, imgv_path, imgl_path, timer, frame_id, frame0_gt)

            if outputs[0] is not None:
                id_none = 0
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, gt_frame0=img_info, img_name=imgi_path, mode='d')
            else:
                if frame0_gt is not None:
                    pre_results = np.append(gt_tlbr, 1) if frame_id == 1 or pre_results is None else pre_results
                    pre_results = np.expand_dims(pre_results, axis=0) if pre_results.ndim == 1 else pre_results
                    pre_results = pre_results.astype(np.float32)
                    online_targets = tracker.update(pre_results, [img_info['height'], img_info['width']], exp.test_size,
                                                    gt_frame0=img_info, img_name=imgi_path, mode='nd')
                    id_none += 1
                else:
                    online_targets = None
            online_tlwhs = []
            online_ids = []
            online_scores = []
            if outputs[0] is not None:
                bboxes_det = outputs[0][:, :4]
            else:
                bboxes_det = None
            if online_targets is not None:
                for t in online_targets:
                    tlwh = t.tlwh
                    tlwh, tlbr = border_limit(tlwh.tolist(), img_info['height'], img_info['width'])
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        pre_tlbrs.append(tlbr)
                        pre_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                if not online_targets and frame0_gt is not None:
                    results.append(
                        f"{frame_id},1,1,1,10,10,1,-1,-1,-1\n"
                    )
                last_results = np.column_stack((pre_tlbrs, pre_scores))
                timer.toc()
                online_im, online_imv = plot_tracking(
                    img_info['raw_imgi'], online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time, det_tlbr=bboxes_det, imageinfo=img_info
                )
                if id_none == 1:
                    start_results = last_results
                pre_results = last_results if last_results.size != 0 else pre_results
                if id_none == 3:
                    pre_results = start_results
                    id_none = 0
            elif frame0_gt is not None:
                results.append(
                    f"{frame_id},1,1,1,10,10,1,-1,-1,-1\n"
                )


            # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_image_result:
                timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
                save_folder = osp.join(vis_folder, seq+timestamp)
                os.makedirs(save_folder, exist_ok=True)
                cv2.imwrite(osp.join(save_folder, osp.basename(imgi_path)), online_im)
                cv2.imwrite(osp.join(save_folder, osp.basename(imgv_path)), online_imv)

            if frame_id % 20 == 0:
                logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        if args.save_result:
            with open(res_file, 'w') as f:
                f.writelines(results)
            logger.info(f"save results to {res_file}")




def imageflow_demo(predictor, vis_folder, current_time, args, gt=None):
    IR_video_path = osp.join(args.path, 'IR.mp4')
    Low_video_path = osp.join(args.path, 'low.mp4')
    RGB_video_path = osp.join(args.path, 'rgb.mp4')
    cap_IR = cv2.VideoCapture(IR_video_path if args.demo == "video" else args.camid)
    cap_low = cv2.VideoCapture(Low_video_path if args.demo == "video" else args.camid)
    cap_rgb = cv2.VideoCapture(RGB_video_path if args.demo == "video" else args.camid)
    width = cap_IR.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap_IR.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap_IR.get(cv2.CAP_PROP_FPS)
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = osp.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = osp.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    tracker = UnismotTracker(args, frame_rate=25)
    timer = Timer()
    frame_id = 0
    results = []
    while True:
        if frame_id % 20 == 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))
        ret_val_IR, frame_IR = cap_IR.read()
        ret_val_low, frame_low = cap_low.read()
        ret_val_rgb, frame_rgb = cap_rgb.read()

        if ret_val_IR:
            outputs, img_info = predictor.inference(frame_IR, frame_rgb, frame_low, timer)
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], [img_info['height'], img_info['width']], exp.test_size, gt)
                online_tlwhs = []
                online_ids = []
                online_scores = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                        online_tlwhs.append(tlwh)
                        online_ids.append(tid)
                        online_scores.append(t.score)
                        results.append(
                            f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                timer.toc()
                online_im = plot_tracking(
                    img_info['raw_imgi'], online_tlwhs, online_ids, frame_id=frame_id + 1, fps=1. / timer.average_time
                )
            else:
                timer.toc()
                online_im = img_info['raw_imgi']
            if args.save_result:
                vid_writer.write(online_im)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
        frame_id += 1

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        logger.info(f"save results to {res_file}")


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    if args.save_result:
        vis_folder = osp.join(output_dir, "track_UniRTL")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model(args).to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size, args.nlow)))
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = osp.join(output_dir, "best_ckpt.pth.tar")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.fp16:
        model = model.half()  # to FP16

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    predictor = UnismotPredictor(model, exp, trt_file, decoder, args.device, args.fp16, args.use_FTP)
    # current_time = time.localtime()

    seq_home = args.path
    seq_list = [f for f in os.listdir(seq_home) if osp.isdir(osp.join(seq_home, f))]
    seq_list.sort()
    for num, seq in enumerate(seq_list):
        current_time = time.localtime()
        if args.demo == "image":
            image_demo(predictor, vis_folder, current_time, args, seq)
        elif args.demo == "video" or args.demo == "webcam":
            imageflow_demo(predictor, vis_folder, current_time, args)



if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name, args.nlow)

    main(exp, args)
