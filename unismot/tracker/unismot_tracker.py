import numpy as np
from collections import deque
import os
import os.path as osp
import copy
import torch
import torch.nn.functional as F
import cv2
from .reid_model import Extractor
from .kalman_filter import KalmanFilter
from unismot.tracker import matching
from .basetrack import BaseTrack, TrackState

def border_limit(bbox_xywh, img_width, img_height):
    original_shape = bbox_xywh.shape
    bbox_xywh = np.reshape(bbox_xywh, (-1, 4))
    bbox_xywh[:, 0] = np.clip(bbox_xywh[:, 0], 1, img_width)
    bbox_xywh[:, 1] = np.clip(bbox_xywh[:, 1], 1, img_height)
    bbox_xywh[:, 2] = np.clip(bbox_xywh[:, 2], 5, img_width)
    bbox_xywh[:, 3] = np.clip(bbox_xywh[:, 3], 5, img_height)
    bbox_xywh = np.reshape(bbox_xywh, original_shape)

    return bbox_xywh

def _cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_cosine_distance(x, y):
    # if x.ndim == 2 and y.ndim == 2:
    #     x = np.squeeze(x)
    #     y = np.squeeze(y)
    distances = _cosine_distance(x, y)
    return distances.min(axis=0)

class NearestNeighborDistanceMetric(object):
    def __init__(self, metric, matching_threshold, budget=None):

        if metric == "cosine":
            self._metric = _nn_cosine_distance
        else:
            raise ValueError(
                "Invalid metric; must be either 'euclidean' or 'cosine'")
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, features, targets, features_track):
        cost_matrix = np.zeros((len(features_track), len(features)))
        for i, target in enumerate(targets):
            cost_matrix[i, :] = self._metric(features_track, features)
        return cost_matrix


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, mode, feature=None):

        # wait activate

        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.mode = mode
        self.kfpre_num = 1
        if feature is not None and self.mode == 'd':
            self.feature = np.asarray(feature, dtype=np.float32)
        else:
            self.feature = None

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, init_frame0=False):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if init_frame0:
            if frame_id == 1:
                self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id, kf=None):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        if new_track.feature is not None:
            self.feature = new_track.feature
        if kf is not None:
            self.kfpre_num += 1
        self.score = new_track.score

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)



class UnismotTracker(object):
    def __init__(self, args, frame_rate=25, use_cuda=True):
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args
        #self.det_thresh = args.track_thresh
        self.det_thresh = args.track_thresh
        if args.track_thresh >= 0.5:
            self.det_thresh_sec = args.track_thresh - 0.4  # + 0,1
        else:
            self.det_thresh_sec = 0.1
        self.buffer_size = int(frame_rate / 25.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.ReID_model = args.ReID_folder
        self.extractor = Extractor(args.ReID_folder, use_cuda=use_cuda)
        self.m_distance_thre = 20  # 20  # ReID maha distance
        self.pix_distance_thre = 108 # 108  # ReID pixel distance

        self.max_age = 1 # 1
        self.max_cosine_distance = 0.3  # metric Associations, larger than this value are disregarded.
        self.nn_budget = 100
        self.metric = NearestNeighborDistanceMetric("cosine", self.max_cosine_distance, self.nn_budget)


    def update(self, output_results, img_info, img_size, gt_frame0=None, img_name=None, mode=None):
        gt = gt_frame0['FTP_init']
        if gt is not None:
            if mode == 'd':
                ori_img = cv2.imread(img_name)
                self.height, self.width = ori_img.shape[:2]
            if self.frame_id == 0:
                if mode =='nd':
                    ori_img = cv2.imread(img_name)
                    self.height, self.width = ori_img.shape[:2]
                features_frame0 = self._get_features(gt, ori_img)
                features_frame0 = np.squeeze(features_frame0)
                score_gt = 1
                score_gt = np.array(score_gt)
                gt_track = STrack(gt, score_gt, mode='d', feature=features_frame0)
                gt_track.renew_count()
                init_frame0 = gt_frame0['use_FTP']
                gt_track.activate(self.kalman_filter, self.frame_id + 1, init_frame0)
                self.tracked_stracks.append(gt_track)
        if gt is None:
            if self.frame_id == 0:
                gt_mot_init = [320, 240, 330, 250]
                gt_mot_init = np.array(gt_mot_init)
                score_gt_init = 1
                score_gt_init = np.array(score_gt_init)
                gt_track = STrack(gt_mot_init, score_gt_init, mode='nd')
                gt_track.renew_count()

        self.frame_id += 1
        activated_starcks = []
        activated_starcks_f = []
        refind_stracks = []
        refind_stracks_f = []
        lost_stracks = []
        removed_stracks = []

        if output_results.shape[1] == 5:
            scores = output_results[:, 4]
            bboxes = output_results[:, :4]
        else:
            if not isinstance(output_results, np.ndarray):
                output_results = output_results.cpu().numpy()
            try:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
            except IndexError:
                print('index out of bounds')
                return None
        img_h, img_w = img_info[0], img_info[1]
        scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
        bboxes /= scale

        remain_inds = scores > self.args.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.args.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_first = bboxes[remain_inds]
        dets_second = bboxes[inds_second]
        scores_keep = scores[remain_inds]
        scores_second = scores[inds_second]

        ''' init '''
        if len(dets_first) > 0:
            '''Detections'''
            if gt is not None and mode == 'd':
                features_first = self._get_features(dets_first, ori_img)
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, mode, f) for
                              (tlbr, s, f) in zip(dets_first, scores_keep, features_first)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, mode) for
                              (tlbr, s) in zip(dets_first, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        if mode =='d' and gt is not None:
            matches_f, unmatched_tracks_f, unmatched_detections_f = self._match(strack_pool, detections)

        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)

        if mode == 'd' and gt is not None:
            for itracked, idet in matches_f:
                if itracked == 0:
                    if matches.size == 0 or (len(matches.shape) == 2 and not np.any(matches[:, 1] == idet)):
                        dict_matches = dict(matches)
                        dict_matches_f = dict(matches_f)
                        for key in dict_matches_f.keys():
                            dict_matches[key] = dict_matches_f[key]
                        matches = list(dict_matches.items())
                        u_track = u_track[u_track != itracked]
                        u_detection = u_detection[u_detection != idet]
                        break
                    mask = (matches[:,1] == idet) if len(matches.shape) == 2 else False
                    matched_idet = matches[mask] if np.any(mask) else None
                    if matched_idet is not None:
                        idtrack = matched_idet[0][0]
                        if idtrack == 0:
                            break
                        idet = matched_idet[0][1]
                        track = strack_pool[idtrack]
                        if track.tracklet_len <= 5:
                            matches = matches[matches[:, 1] != idet]
                            dict_matches = dict(matches)
                            dict_matches_f = dict(matches_f)
                            for key in dict_matches_f.keys():
                                dict_matches[key] = dict_matches_f[key]
                            matches = list(dict_matches.items())
                            u_track = u_track[u_track != itracked]
                            u_track = np.sort(np.append(u_track, idtrack))
                            u_detection = u_detection[u_detection != idet]


        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            if gt is not None and mode == 'd':
                features_second = self._get_features(dets_second, ori_img)
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, mode, f) for
                                     (tlbr, s, f) in zip(dets_second, scores_second, features_second)]
            else:
                detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, mode) for
                                     (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]

        if mode =='d' and gt is not None:
            if r_tracked_stracks and r_tracked_stracks[0].track_id == 1:
                matches_f, unmatched_tracks_f, unmatched_detections_f = self._match(r_tracked_stracks, detections_second)
            else:
                matches_f = []

        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)

        if mode == 'd' and gt is not None:
            for itracked, idet in matches_f:
                if itracked == 0:
                    if matches.size == 0 or (len(matches.shape) == 2 and not np.any(matches[:, 1] == idet)):
                        dict_matches = dict(matches)
                        dict_matches_f = dict(matches_f)
                        for key in dict_matches_f.keys():
                            dict_matches[key] = dict_matches_f[key]
                        matches = list(dict_matches.items())
                        u_track = u_track[u_track != itracked]
                        u_detection_second = u_detection_second[u_detection_second != idet]
                        break
                    mask = (matches[:, 1] == idet) if len(matches.shape) == 2 else False
                    matched_idet = matches[mask] if np.any(mask) else None
                    if matched_idet is not None:
                        idtrack = matched_idet[0][0]
                        if idtrack == 0:
                            break
                        idet = matched_idet[0][1]
                        track = strack_pool[idtrack]
                        if track.tracklet_len <= 5:
                            matches = matches[matches[:, 1] != idet]
                            dict_matches = dict(matches)
                            dict_matches_f = dict(matches_f)
                            for key in dict_matches_f.keys():
                                dict_matches[key] = dict_matches_f[key]
                            matches = list(dict_matches.items())
                            u_track = u_track[u_track != itracked]
                            u_track = np.sort(np.append(u_track, idtrack))
                            u_detection_second = u_detection_second[u_detection_second != idet]


        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            # if gt is not None and track.state == TrackState.Tracked:
            if track.state == TrackState.Tracked:
                if gt is not None:
                    if track.track_id == 1:
                        track.predict()
                        track.update(track, self.frame_id, 'kf_pre')
                        activated_starcks.append(track)
                else:
                    track.predict()
                    track.update(track, self.frame_id, 'kf_pre')
                    activated_starcks.append(track)


            if not track.state == TrackState.Lost and (track.kfpre_num % 15) == 0:
                if gt is None:
                    track.mark_lost()
                    lost_stracks.append(track)
            if not track.state == TrackState.Lost and (track.kfpre_num % 1) == 0:
                if gt is not None:
                    if track.track_id != 1:
                        track.mark_lost()
                        lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        # if gt is None and len(u_unconfirmed) > 0:
        if len(u_unconfirmed) > 0:
            r_unconfirmed = [unconfirmed[i] for i in u_unconfirmed if unconfirmed[i].state == TrackState.Tracked]
            detections_second = [detections_second[i] for i in u_detection_second]
            dists = matching.iou_distance(r_unconfirmed, detections_second)
            if not self.args.mot20:
                dists = matching.fuse_score(dists, detections_second)
            matches, u_unconfirmed, u_detection_second = matching.linear_assignment(dists, thresh=0.82)
            for itracked, idet in matches:
                r_unconfirmed[itracked].update(detections_second[idet], self.frame_id)
                activated_starcks.append(r_unconfirmed[itracked])

        for it in u_unconfirmed:
            if len(u_unconfirmed) > 0:
                track = r_unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)
            else:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if gt is not None:
                if track.score < self.det_thresh - 0.4:
                    continue
            else:
                if track.score < self.det_thresh:
                    continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        # if gt is None and len(u_detection_second) > 0:
        if len(u_detection_second) > 0:
            for inew in u_detection_second:
                track = detections_second[inew]
                if track.score < self.det_thresh_sec:
                    continue
                track.activate(self.kalman_filter, self.frame_id)
                activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # print('Ramained match {} s'.format(t4-t3))

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        if gt is not None:
            output_stracks = [track for track in self.tracked_stracks if track.track_id == 1]
        else:
            output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        return output_stracks

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        if np.any(bbox_xywh <= 0):
            self.height, self.width = ori_img.shape[:2]
            bbox_xywh = border_limit(bbox_xywh, self.width, self.height)
        if isinstance(bbox_xywh, np.ndarray) and bbox_xywh.ndim == 1:
            x1, y1, x2, y2 = self._tlwh_to_xyxy(bbox_xywh)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        else:
            for box in bbox_xywh:
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
                im = ori_img[y1:y2, x1:x2]
                im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def _match(self, strack_pool, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            features_track = np.array([tracks[i].feature for i in track_indices])
            cost_matrix = self.metric.distance(features, targets, features_track)
            cost_matrix = matching.gate_cost_matrix(
                self.kalman_filter, cost_matrix, tracks, dets, track_indices,
                detection_indices, self.m_distance_thre, self.pix_distance_thre)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(strack_pool) if t.state == TrackState.Tracked]
        unconfirmed_tracks = [
            i for i, t in enumerate(strack_pool) if not t.state == TrackState.Tracked]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            matching.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                strack_pool, detections, confirmed_tracks)

        matches = matches_a # + matches_b
        # unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        unmatched_tracks = unmatched_tracks_a
        return matches, unmatched_tracks, unmatched_detections



def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
