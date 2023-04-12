import torch
from maskrcnn_benchmark.config import cfg

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Keypoints(object):
    def __init__(self, keypoints, size, mode=None):
        # FIXME remove check once we have better integration with device
        # in my version this would consistently return a CPU tensor
        device = keypoints.device if isinstance(keypoints, torch.Tensor) else torch.device('cpu')
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32, device=device)
        num_keypoints = keypoints.shape[0]
        if num_keypoints:
            keypoints = keypoints.view(num_keypoints, -1, 3)

        # TODO should I split them?
        # self.visibility = keypoints[..., 2]
        self.keypoints = keypoints  # [..., :2]

        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def crop(self, box):
        raise NotImplementedError()

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_w, ratio_h = ratios
        resized_data = self.keypoints.clone()
        resized_data[..., 0] *= ratio_w
        resized_data[..., 1] *= ratio_h
        keypoints = type(self)(resized_data, size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError(
                "Only FLIP_LEFT_RIGHT implemented")

        flip_inds = self.FLIP_INDS
        flipped_data = self.keypoints[:, flip_inds]
        width = self.size[0]
        TO_REMOVE = 1
        # Flip x coordinates
        flipped_data[..., 0] = width - flipped_data[..., 0] - TO_REMOVE

        # Maintain COCO convention that if visibility == 0, then x, y = 0
        inds = flipped_data[..., 2] == 0
        flipped_data[inds] = 0

        keypoints = type(self)(flipped_data, self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v)
        return keypoints

    def to(self, *args, **kwargs):
        keypoints = type(self)(self.keypoints.to(*args, **kwargs), self.size, self.mode)
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            keypoints.add_field(k, v)
        return keypoints

    def __getitem__(self, item):
        keypoints = type(self)(self.keypoints[item], self.size, self.mode)
        for k, v in self.extra_fields.items():
            keypoints.add_field(k, v[item])
        return keypoints

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'num_instances={}, '.format(len(self.keypoints))
        s += 'image_width={}, '.format(self.size[0])
        s += 'image_height={})'.format(self.size[1])
        return s


class PersonKeypoints(Keypoints):
    _NAMES = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle'
    ]
    _FLIP_MAP = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'
    }

    def __init__(self, *args, **kwargs):
        super(PersonKeypoints, self).__init__(*args, **kwargs)
        if len(cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME)>0:
            self.NAMES = cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME
            self.FLIP_MAP = {l:r for l,r in PersonKeypoints._FLIP_MAP.items() if l in cfg.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME}
        else:
            self.NAMES = PersonKeypoints._NAMES
            self.FLIP_MAP = PersonKeypoints._FLIP_MAP

        self.FLIP_INDS = self._create_flip_indices(self.NAMES, self.FLIP_MAP)
        self.CONNECTIONS = self._kp_connections(self.NAMES)

    def to_coco_format(self):
        coco_result = []
        for i in range(self.keypoints.shape[0]):
            coco_kps = [0]*len(PersonKeypoints._NAMES)*3
            for ki, name in enumerate(self.NAMES):
                coco_kps[3*PersonKeypoints._NAMES.index(name)] = self.keypoints[i,ki,0].item()
                coco_kps[3*PersonKeypoints._NAMES.index(name)+1] = self.keypoints[i,ki,1].item()
                coco_kps[3*PersonKeypoints._NAMES.index(name)+2] = self.keypoints[i,ki,2].item()
            coco_result.append(coco_kps)
        return coco_result

    def _create_flip_indices(self, names, flip_map):
        full_flip_map = flip_map.copy()
        full_flip_map.update({v: k for k, v in flip_map.items()})
        flipped_names = [i if i not in full_flip_map else full_flip_map[i] for i in names]
        flip_indices = [names.index(i) for i in flipped_names]
        return torch.tensor(flip_indices)


    def _kp_connections(self, keypoints):
        CONNECTIONS = [
            ['left_eye', 'right_eye'],
            ['left_eye', 'nose'],
            ['right_eye', 'nose'],
            ['right_eye', 'right_ear'],
            ['left_eye', 'left_ear'],
            ['right_shoulder', 'right_elbow'],
            ['right_elbow', 'right_wrist'],
            ['left_shoulder', 'left_elbow'],
            ['left_elbow', 'left_wrist'],
            ['right_hip', 'right_knee'],
            ['right_knee', 'right_ankle'],
            ['left_hip', 'left_knee'],
            ['left_knee', 'left_ankle'],
            ['right_shoulder', 'left_shoulder'],
            ['right_hip', 'left_hip'],
        ]

        kp_lines = [[keypoints.index(conn[0]), keypoints.index(conn[1])] for conn in CONNECTIONS
                    if conn[0] in self.NAMES and conn[1] in self.NAMES]
        return kp_lines



# TODO make this nicer, this is a direct translation from C2 (but removing the inner loop)
def keypoints_to_heat_map(keypoints, rois, heatmap_size):
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    return heatmaps, valid