import re
import mindspore
from mindspore import Model
import mindspore.nn as nn

from .labeled_matching_layer_queue import LabeledMatchingLayerQueue
from .unlabeled_matching_layer import UnlabeledMatchingLayer
from .alignps import TripletLossFilter

INF = 1e8


class FCOSReidHeadFocalSubTriQueue3():

    def __init__(self,
                 num_classes,
                 in_channels,
                 #regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                 #                (512, INF)),
                 regress_ranges=((-1, INF), (-2, -1), (-2, -1), (-2, -1),
                                (-2, -1)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.background_id = -2

        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = loss_centerness
        self.loss_tri = TripletLossFilter()

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        #self._init_reid_convs()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        num_person = 5532
        # queue_size = 500
        queue_size = 5000
        #self.classifier_reid = nn.Linear(self.feat_channels, num_person)
        self.scales = nn([(1.0) for _ in self.strides])
        self.labeled_matching_layer = LabeledMatchingLayerQueue(num_persons=num_person, feat_len=self.in_channels) # for mot17half
        self.unlabeled_matching_layer = UnlabeledMatchingLayer(queue_size=queue_size, feat_len=self.in_channels)

    def _init_reid_convs(self):
        """Initialize classification conv layers of the head."""
        self.reid_convs = nn.Cells()
        #for i in range(self.stacked_convs):
        for i in range(1):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.reid_convs.append(
                nn.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    #norm_cfg=self.norm_cfg,
                    norm_cfg=dict(type='BN', requires_grad=True),
                    bias=self.conv_bias))

    def forward(self, feats):
        feats = list(feats)
        h, w = feats[0].shape[2], feats[0].shape[3]
        mean_value = nn.functional.adaptive_avg_pool2d(feats[0], 1)
        mean_value = mindspore.ops.upsample(input=mean_value, size=(h, w), mode='bilinear')
        feats[0] = feats[0] - mean_value
        return self.forward_single(feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        
        reid_feat = x
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = mindspore.nn.loss.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred, centerness, reid_feat

    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             reid_feats,
             gt_bboxes,
             gt_labels,
             gt_ids):

        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(reid_feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, ids, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels, gt_ids)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_reid = [
            reid_feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)
            for reid_feat in reid_feats
        ]
        flatten_cls_scores = mindspore.ops.cat(flatten_cls_scores)
        flatten_bbox_preds = mindspore.ops.cat(flatten_bbox_preds)
        flatten_centerness = mindspore.ops.cat(flatten_centerness)
        flatten_reid = mindspore.ops.cat(flatten_reid)
        #print("flatten reid", flatten_reid.shape)
        flatten_labels = mindspore.ops.cat(labels)
        flatten_ids = mindspore.ops.cat(ids)
        flatten_bbox_targets = mindspore.ops.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = mindspore.ops.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        #pos_inds = nonzero((flatten_labels >= 0) & (flatten_labels < bg_class_ind)).reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]


        pos_reid = flatten_reid[pos_inds]
        pos_reid = mindspore.ops.normalize(pos_reid)


        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = mindspore.ops.distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = mindspore.ops.distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)

            
            pos_reid_ids = flatten_ids[pos_inds]
            labeled_matching_scores, labeled_matching_reid, labeled_matching_ids = self.labeled_matching_layer(pos_reid, pos_reid_ids)
            labeled_matching_scores *= 10
            unlabeled_matching_scores = self.unlabeled_matching_layer(pos_reid, pos_reid_ids)
            unlabeled_matching_scores *= 10
            matching_scores = mindspore.ops.cat((labeled_matching_scores, unlabeled_matching_scores), dim=1)
            pid_labels = pos_reid_ids.clone()
            pid_labels[pid_labels == -2] = -1

            p_i = mindspore.nn.loss.softmax(matching_scores, dim=1)
            #focal_p_i = 0.25 * (1 - p_i)**2 * p_i.log()
            focal_p_i = (1 - p_i)**2 * p_i.log()

            #loss_oim = F.nll_loss(focal_p_i, pid_labels, reduction='none', ignore_index=-1)
            loss_oim = mindspore.nn.loss.F.nll_loss(focal_p_i, pid_labels, ignore_index=-1)

            pos_reid1 = mindspore.ops.cat((pos_reid, labeled_matching_reid), dim=0)
            pid_labels1 = mindspore.ops.cat((pid_labels, labeled_matching_ids), dim=0)
            loss_tri = self.loss_tri(pos_reid1, pid_labels1)
            
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_oim = pos_reid.sum()
            loss_tri = pos_reid.sum()
            print('no gt box')

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_oim=loss_oim,
            loss_tri=loss_tri), dict(pos_reid=pos_reid, pos_reid_ids=pos_reid_ids, out_preds=p_i)

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   reid_feats,
                   img_metas,
                   cfg=None,
                   rescale=None):

        assert len(cls_scores) == len(bbox_preds) == len(reid_feats)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            reid_feat_list = [
                reid_feats[i][img_id].detach() for i in range(num_levels)
            ]
            #print(img_metas)
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 centerness_pred_list,
                                                 reid_feat_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           reid_feats,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points) == len(reid_feats)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_reid_feats = []
        for cls_score, bbox_pred, centerness, points, reid_feat in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points, reid_feats):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            #reid_feat = reid_feat.permute(1, 2, 0).reshape(-1, 256)
            reid_feat = reid_feat.permute(1, 2, 0).reshape(-1, self.in_channels)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
                reid_feat = reid_feat[topk_inds, :]
            bboxes = mindspore.ops.distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
            mlvl_reid_feats.append(reid_feat)
        mlvl_bboxes = mindspore.ops.cat(mlvl_bboxes)
        mlvl_reid_feats = mindspore.ops.cat(mlvl_reid_feats)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = mindspore.ops.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = mindspore.ops.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = mindspore.ops.cat(mlvl_centerness)
        det_bboxes, det_labels, det_reid_feats = multiclass_nms_reid(
            mlvl_bboxes,
            mlvl_scores,
            mlvl_reid_feats,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        return det_bboxes, det_labels, det_reid_feats

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = mindspore.ops.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list, gt_ids_list):

        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = mindspore.ops.cat(expanded_regress_ranges, dim=0)
        concat_points = mindspore.ops.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, ids_list, bbox_targets_list = self._get_target_single(
            gt_bboxes_list,
            gt_labels_list,
            gt_ids_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        ids_list = [ids.split(num_points, 0) for ids in ids_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_ids = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                mindspore.ops.cat([labels[i] for labels in labels_list]))
            concat_lvl_ids.append(
                mindspore.ops.cat([ids[i] for ids in ids_list]))
            bbox_targets = mindspore.ops.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_ids, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, gt_ids, points, regress_ranges,
                           num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_ids.new_full((num_points,), self.background_id), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = mindspore.ops.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = mindspore.ops.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = mindspore.ops.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = mindspore.ops.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = mindspore.ops.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = mindspore.ops.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = mindspore.ops.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        ids = gt_ids[min_area_inds]
        labels[min_area == INF] = self.background_label  # set as BG
        ids[min_area == INF] = self.background_id # set as unannotated
        bbox_targets = bbox_targets[range(num_points), min_area_inds]

        return labels, ids, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return mindspore.ops.sqrt(centerness_targets)

def multiclass_nms_reid(multi_bboxes,
                   multi_scores,
                   multi_reid_feats,
                   score_thr,
                   nms_cfg,
                   max_num=-1,
                   score_factors=None):

    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)      
    scores = multi_scores[:, :-1]

    # filter out boxes with low scores
    valid_mask = scores > score_thr

    # We use masked_select for ONNX exporting purpose,
    # which is equivalent to bboxes = bboxes[valid_mask]
    # (TODO): as ONNX does not support repeat now,
    # we have to use this ugly code
    multi_reid_feats = multi_reid_feats[valid_mask.squeeze()]
    bboxes = mindspore.masked_select(
        bboxes,
        mindspore.ops.stack((valid_mask, valid_mask, valid_mask, valid_mask),
                    -1)).view(-1, 4)
    if score_factors is not None:
        scores = scores * score_factors[:, None]
    scores = mindspore.ops.masked_select(scores, valid_mask)
    labels = valid_mask.nonzero()[:, 1]

    if bboxes.numel() == 0:
        bboxes = multi_bboxes.new_zeros((0, 5))
        reid_feats = multi_reid_feats.new_zeros((0, 256))
        labels = multi_bboxes.new_zeros((0, ), dtype=mindspore.dtype.long)

        if mindspore.ops.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        return bboxes, labels, reid_feats

    dets, keep = mindspore.ops.batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    return dets, labels[keep], multi_reid_feats[keep]