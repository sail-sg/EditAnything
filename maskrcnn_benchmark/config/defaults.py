# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.RPN_ONLY = False
_C.MODEL.BOX_ON = True
_C.MODEL.MASK_ON = False
_C.MODEL.KEYPOINT_ON = False
_C.MODEL.DEVICE = "cuda"

_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

_C.MODEL.RPN_ARCHITECTURE = "RPN"
_C.MODEL.DEBUG = False  # add debug flag
_C.MODEL.ONNX = False  # add onnx flag

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in paths_catalog. Else, it will use it as the specified absolute
# path
_C.MODEL.WEIGHT = ""
_C.MODEL.PRETRAIN_NAME = ""

# If LINEAR_PROB = True, only the last linear layers in rpn and roi_head are trainable
_C.MODEL.LINEAR_PROB = False

# -----------------------------------------------------------------------------
# Multitask Training / Test specific parameters
# -----------------------------------------------------------------------------
_C.MODEL.MULTITASK = CN(new_allowed=True)

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.MIN_SIZE_TRAIN = 800  # (800,)
# Maximum size of the side of the image during training
_C.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
_C.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
_C.INPUT.TO_BGR255 = True
_C.INPUT.FORMAT = ''
_C.INPUT.FIX_RES = False

# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------
_C.AUGMENT = CN()
_C.AUGMENT.USE_RA = 0
_C.AUGMENT.FLIP_PROB_TRAIN = 0.5
_C.AUGMENT.VERTICAL_FLIP_PROB_TRAIN = 0.0
_C.AUGMENT.MULT_MIN_SIZE_TRAIN = ()

_C.AUGMENT.BRIGHTNESS = 0.0
_C.AUGMENT.CONTRAST = 0.0
_C.AUGMENT.SATURATION = 0.0
_C.AUGMENT.HUE = 0.0

_C.AUGMENT.CROP_PROB = 0.5
_C.AUGMENT.CROP_MIN_IOUS = (0.1, 0.3, 0.5, 0.7, 0.9)
_C.AUGMENT.CROP_MIN_SIZE = 0.3

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
# Use is_crowd label
_C.DATASETS.USE_CROWD = False
_C.DATASETS.CLASS_AGNOSTIC = False
_C.DATASETS.CLASS_CONCAT = False
_C.DATASETS.MAX_BOX = -1
_C.DATASETS.SAMPLE_RATIO = 0.0
_C.DATASETS.FEW_SHOT = 0
# SHUFFLE_SEED != 0 means shuffle the dataset in the few shot setting
_C.DATASETS.SHUFFLE_SEED = 0
_C.DATASETS.PREDEFINED_TEXT = ''
_C.DATASETS.ALTERNATIVE_TRAINING = False
_C.DATASETS.MULTISTAGE_TRAINING = False
_C.DATASETS.REGISTER = CN(new_allowed=True)
_C.DATASETS.BOX_THRESHOLD = 0.1
# Duplicate Dataset
_C.DATASETS.COCO_COPY = 1
_C.DATASETS.LVIS_COPY = 1
_C.DATASETS.FLICKR_COPY = 1
_C.DATASETS.MIXED_COPY = 1
_C.DATASETS.OBJECT365_COPY = 1
_C.DATASETS.VG_COPY = 1
_C.DATASETS.OI_COPY = 1
_C.DATASETS.IN_COPY = 1

# Duplicate Dataset
_C.DATASETS.COCO_COPY = 1
_C.DATASETS.FLICKR_COPY = 1
_C.DATASETS.MIXED_COPY = 1
_C.DATASETS.OBJECT365_COPY = 1
_C.DATASETS.VG_COPY = 1
_C.DATASETS.OI_COPY = 1
_C.DATASETS.IN_COPY = 1
_C.DATASETS.GENERAL_COPY = -1
_C.DATASETS.GENERAL_COPY_TEST = -1

# OD to Grounding
_C.DATASETS.RANDOM_SAMPLE_NEG = -1
_C.DATASETS.ADD_DET_PROMPT = False
_C.DATASETS.ADD_DET_PROMPT_ADVANCED = False
_C.DATASETS.USE_OD_AUG = False
_C.DATASETS.USE_COCO_FORMAT = False
_C.DATASETS.CONTROL_PROB = ()
_C.DATASETS.DISABLE_SHUFFLE = False
_C.DATASETS.PROMPT_VERSION = ""
_C.DATASETS.PROMPT_LIMIT_NEG = -1
_C.DATASETS.POS_QUESTION_PROB = 0.6
_C.DATASETS.NEG_QUESTION_PROB = 0.8
_C.DATASETS.FULL_QUESTION_PROB = 0.5
_C.DATASETS.ONE_HOT = False
_C.DATASETS.NO_MINUS_ONE_FOR_ONE_HOT = False

_C.DATASETS.DISABLE_CLIP_TO_IMAGE = False
_C.DATASETS.SEPARATION_TOKENS = " "

# LVIS
_C.DATASETS.LVIS_USE_NORMAL_AP = False
_C.DATASETS.SPECIAL_SAFEGUARD_FOR_COCO_GROUNDING = False

# Caption
_C.DATASETS.BING_INDEX_LIST = []
_C.DATASETS.CAPTION_MIN_BOX = 1
_C.DATASETS.REPLACE_CLEAN_LABEL = False
_C.DATASETS.FURTHER_SCREEN = False
_C.DATASETS.CAPTION_CONF = 0.9
_C.DATASETS.CAPTION_NMS = 0.9
_C.DATASETS.PACK_RANDOM_CAPTION_NUMBER = 0
_C.DATASETS.INFERENCE_CAPTION = False
_C.DATASETS.SAMPLE_NEGATIVE_FOR_GROUNDING_DATA = -1.0
_C.DATASETS.RANDOM_PACK_PROB = -1.0
_C.DATASETS.NO_RANDOM_PACK_PROBABILITY = 0.0
_C.DATASETS.SAFEGUARD_POSITIVE_CAPTION = True
_C.DATASETS.CAPTION_FORMAT_VERSION = "v1"
_C.DATASETS.LOCAL_DEBUG = False


# Od in the wild
_C.DATASETS.PREDEFINED_TEXT = None
_C.DATASETS.TRAIN_DATASETNAME_SUFFIX = ""
_C.DATASETS.TEST_DATASETNAME_SUFFIX = ""
_C.DATASETS.OVERRIDE_CATEGORY = None
_C.DATASETS.USE_OVERRIDE_CATEGORY = False
_C.DATASETS.SUPRESS_QUERY = None
_C.DATASETS.USE_SUPRESS_QUERY = False
_C.DATASETS.USE_CAPTION_PROMPT = False
_C.DATASETS.CAPTION_PROMPT = None

_C.DATASETS.FLICKR_GT_TYPE = "separate"

# VQA
_C.DATASETS.DIVER_BOX_FOR_VQA = False
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = True
# Define min number of keypoints required from GT, for example 10 out of 17
_C.DATALOADER.MIN_KPS_PER_IMS = 0
# Use random sampler during training
_C.DATALOADER.USE_RANDOM_SEED = False

_C.DATALOADER.DISTRIBUTE_CHUNK_AMONG_NODE = False
# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
# (e.g., 'FPN.add_fpn_ResNet101_conv5_body' to specify a ResNet-101-FPN
# backbone)
_C.MODEL.BACKBONE.CONV_BODY = "R-50-C4"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 2
_C.MODEL.BACKBONE.FREEZE = False
_C.MODEL.BACKBONE.GROUP = 1
_C.MODEL.BACKBONE.OUT_CHANNELS = 256 * 4
# Option to reset bn running statics
_C.MODEL.BACKBONE.RESET_BN = False
# Backbone Normalization Level
_C.MODEL.BACKBONE.NORM_LEVEL = 3
# BN for backbone
_C.MODEL.BACKBONE.USE_BN = False
# Sync BN for backbone
_C.MODEL.BACKBONE.USE_SYNCBN = False
_C.MODEL.BACKBONE.USE_NSYNCBN = False
# GN for backbone
_C.MODEL.BACKBONE.USE_GN = False
# Evo Norm for backbone
_C.MODEL.BACKBONE.USE_EN = False
# Layers for backbone
_C.MODEL.BACKBONE.USE_DFCONV = False
_C.MODEL.BACKBONE.USE_DYRELU = False
_C.MODEL.BACKBONE.USE_SE = False
_C.MODEL.BACKBONE.LAYER_SETUP = (3, 4, 6, 3)
_C.MODEL.BACKBONE.LAYER_SEARCH = CN(new_allowed=True)
_C.MODEL.BACKBONE.OUT_FEATURES = ("stage2", "stage3", "stage4", "stage5")
_C.MODEL.BACKBONE.FPN_LAYER = ()
_C.MODEL.BACKBONE.USE_CHECKPOINT = False
# Add JF efficient det cfgs
_C.MODEL.BACKBONE.EFFICIENT_DET_START_FROM = 3
_C.MODEL.BACKBONE.EFFICIENT_DET_COMPOUND = 0
_C.MODEL.BACKBONE.EFFICIENT_DET_BIFPN_VERSION = 0

_C.MODEL.LANGUAGE_BACKBONE = CN()
_C.MODEL.LANGUAGE_BACKBONE.WEIGHT = ""
_C.MODEL.LANGUAGE_BACKBONE.FREEZE = False
_C.MODEL.LANGUAGE_BACKBONE.USE_CHECKPOINT = False
_C.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE = "bert-base-uncased"
_C.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE = "bert-base-uncased"
_C.MODEL.LANGUAGE_BACKBONE.LANG_DIM = 768
_C.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN = 256
_C.MODEL.LANGUAGE_BACKBONE.N_LAYERS = 1
_C.MODEL.LANGUAGE_BACKBONE.UNUSED_TOKEN = 106
_C.MODEL.LANGUAGE_BACKBONE.MASK_SPECIAL = False

_C.MODEL.LANGUAGE_BACKBONE.RNN_TYPE = "lstm"
_C.MODEL.LANGUAGE_BACKBONE.VARIABLE_LENGTH = True
_C.MODEL.LANGUAGE_BACKBONE.WORD_EMBEDDING_SIZE = 512
_C.MODEL.LANGUAGE_BACKBONE.WORD_VEC_SIZE = 512
_C.MODEL.LANGUAGE_BACKBONE.HIDDEN_SIZE = 512
_C.MODEL.LANGUAGE_BACKBONE.BIDIRECTIONAL = True
_C.MODEL.LANGUAGE_BACKBONE.INPUT_DROPOUT_P = 0.5
_C.MODEL.LANGUAGE_BACKBONE.DROPOUT_P = 0.2
_C.MODEL.LANGUAGE_BACKBONE.CORPUS_PATH = ""
_C.MODEL.LANGUAGE_BACKBONE.VOCAB_SIZE = 0

_C.MODEL.LANGUAGE_BACKBONE.PAD_MAX = True
# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.FPN = CN()
_C.MODEL.FPN.FREEZE = False
_C.MODEL.FPN.USE_GN = False
_C.MODEL.FPN.USE_RELU = False
_C.MODEL.FPN.USE_DYRELU = False
_C.MODEL.FPN.DROP_BLOCK = True
_C.MODEL.FPN.DROP_PROB = 0.3
_C.MODEL.FPN.DROP_SIZE = 3
_C.MODEL.FPN.USE_SPP = False
_C.MODEL.FPN.USE_PAN = False
_C.MODEL.FPN.USE_DYHEAD = False
_C.MODEL.FPN.RETURN_SWINT_FEATURE_BEFORE_FUSION = False
# ---------------------------------------------------------------------------- #
# BIFPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.BIFPN = CN()
_C.MODEL.BIFPN.NUM_REPEATS = 1
_C.MODEL.BIFPN.USE_ATTENTION = True

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 16
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# Evo Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.EVO_NORM = CN()
# Number of groups in EvoNorm (-1 if using DIM_PER_GP)
_C.MODEL.EVO_NORM.NUM_GROUPS = 8
# EvoNorm's small constant in the denominator
_C.MODEL.EVO_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# RetinaNet Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()
# This is the number of foreground classes and background.
_C.MODEL.RETINANET.NUM_CLASSES = 81
# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4
# During inference, #locs to select based on cls score before NMS is performed
# per FPN level
_C.MODEL.RETINANET.PRE_NMS_TOP_N = 1000
# Prior prob for the positives at the beginning of training. This is used to set
# the bias init for the logits layer
_C.MODEL.RETINANET.PRIOR_PROB = 0.01
# Inference cls score threshold, anchors with score > INFERENCE_TH are
# considered for inference
_C.MODEL.RETINANET.INFERENCE_TH = 0.05
# NMS threshold used in RetinaNet
_C.MODEL.RETINANET.NMS_TH = 0.4
_C.MODEL.RETINANET.DETECTIONS_PER_IMG = 100

# ---------------------------------------------------------------------------- #
# Focal Loss Options (Follow the Detectron version)
# ---------------------------------------------------------------------------- #
_C.MODEL.FOCAL = CN()
# Weight for bbox_regression loss
_C.MODEL.FOCAL.BBOX_REG_WEIGHT = 4.0
# Smooth L1 loss beta for bbox regression
_C.MODEL.FOCAL.BBOX_REG_BETA = 0.11
# IoU overlap ratio for labeling an anchor as positive
# Anchors with >= iou overlap are labeled positive
_C.MODEL.FOCAL.FG_IOU_THRESHOLD = 0.5
# IoU overlap ratio for labeling an anchor as negative
# Anchors with < iou overlap are labeled negative
_C.MODEL.FOCAL.BG_IOU_THRESHOLD = 0.4
# Focal loss parameter: alpha
_C.MODEL.FOCAL.LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FOCAL.LOSS_GAMMA = 2.0

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
_C.MODEL.FCOS.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4
# if use deformable conv to align features
_C.MODEL.FCOS.USE_DFCONV = False

# if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
_C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
# IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
_C.MODEL.FCOS.IOU_LOSS_TYPE = "iou"

_C.MODEL.FCOS.NORM_REG_TARGETS = False
_C.MODEL.FCOS.CENTERNESS_ON_REG = False
_C.MODEL.FCOS.USE_GT_CENTER = False

_C.MODEL.FCOS.DETECTIONS_PER_IMG = 100
_C.MODEL.FCOS.USE_GN = False
_C.MODEL.FCOS.USE_BN = False

_C.MODEL.FCOS.INFERENCE_TH_TRAIN = 0.0
_C.MODEL.FCOS.PRE_NMS_TOP_N_TRAIN = 3000
_C.MODEL.FCOS.POST_NMS_TOP_N_TRAIN = 1000

# ---------------------------------------------------------------------------- #
# ATSS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.ATSS = CN()
_C.MODEL.ATSS.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.ATSS.PRIOR_PROB = 0.01
_C.MODEL.ATSS.INFERENCE_TH = 0.05
_C.MODEL.ATSS.NMS_TH = 0.6
_C.MODEL.ATSS.PRE_NMS_TOP_N = 1000

# the number of convolutions used in the cls and bbox tower
_C.MODEL.ATSS.NUM_CONVS = 4
# the channels of convolutions used in the cls and bbox tower
_C.MODEL.ATSS.CHANNELS = 128
# if use deformable conv to align features
_C.MODEL.ATSS.USE_DFCONV = False

# topk for selecting candidate positive samples from each level
_C.MODEL.ATSS.TOPK = 9

# Weight for bbox_regression loss
_C.MODEL.ATSS.REG_LOSS_WEIGHT = 2.0

_C.MODEL.ATSS.DETECTIONS_PER_IMG = 100
_C.MODEL.ATSS.USE_GN = False
_C.MODEL.ATSS.USE_BN = False

_C.MODEL.ATSS.USE_DYRELU = False
_C.MODEL.ATSS.USE_SE = False

_C.MODEL.ATSS.INFERENCE_TH_TRAIN = 0.0
_C.MODEL.ATSS.PRE_NMS_TOP_N_TRAIN = 3000
_C.MODEL.ATSS.POST_NMS_TOP_N_TRAIN = 1000
# ---------------------------------------------------------------------------- #
# DYHEAD Options
# ---------------------------------------------------------------------------- #
_C.MODEL.DYHEAD = CN()
_C.MODEL.DYHEAD.NUM_CLASSES = 81  # the number of classes including background
_C.MODEL.DYHEAD.PRIOR_PROB = 0.01

# the number of convolutions used in the cls and bbox tower
_C.MODEL.DYHEAD.NUM_CONVS = 4
# the channels of convolutions used in the cls and bbox tower
_C.MODEL.DYHEAD.CHANNELS = 128
_C.MODEL.DYHEAD.GROUPS = 1
# if use deformable conv to align features
_C.MODEL.DYHEAD.USE_DFCONV = False

# topk for selecting candidate positive samples from each level
_C.MODEL.DYHEAD.TOPK = 9

_C.MODEL.DYHEAD.SCORE_AGG = "MEAN"  # MEAN or MAX, for binary focal loss score aggregation

_C.MODEL.DYHEAD.LOG_SCALE = 0.0  # temperature (dot product)
_C.MODEL.DYHEAD.SHALLOW_LOG_SCALE = 0.0  # # temperature (shallow contrastive)

_C.MODEL.DYHEAD.USE_GN = False
_C.MODEL.DYHEAD.USE_NSYNCBN = False
_C.MODEL.DYHEAD.USE_SYNCBN = False

_C.MODEL.DYHEAD.USE_DYFUSE = False
_C.MODEL.DYHEAD.USE_DYRELU = False

_C.MODEL.DYHEAD.CONV_FUNC = ''

# CosineSimOutputLayers: https://github.com/ucbdrive/few-shot-object-detection/blob/master/fsdet/modeling/roi_heads/fast_rcnn.py#L448-L464
_C.MODEL.DYHEAD.COSINE_SCALE = -1.0

_C.MODEL.DYHEAD.FUSE_CONFIG = CN()
_C.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON = False
_C.MODEL.DYHEAD.FUSE_CONFIG.TYPE = ""
_C.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE = 256
_C.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE = 256
_C.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT = 0.1
_C.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS = 2

_C.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS = False

_C.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS = False
_C.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT = 1.0
_C.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA = 2.0
_C.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA = 0.25

_C.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS = False
_C.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS = False
_C.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM = 64
_C.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT = 1.0
_C.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT = 1.0
_C.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE = True
_C.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL = False
_C.MODEL.DYHEAD.FUSE_CONFIG.STABLE_SOFTMAX_2D = False

_C.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT = False

_C.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT = False

# Controls for 
_C.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW = False
_C.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW = False
_C.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW = False
_C.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW = False
_C.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT = False

# MLM Loss
_C.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS = False
_C.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_FOR_ONLY_POSITIVES = True
_C.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_OD = False
_C.MODEL.DYHEAD.FUSE_CONFIG.NO_MASK_FOR_GOLD = False
_C.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF = 1.0
_C.MODEL.DYHEAD.FUSE_CONFIG.MLM_OBJ_FOR_ONLY_POSITIVE  = False

# Shallow Contrastive Loss (FPN)
_C.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS = False
_C.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_MAX_POSITIVE_ANCHORS = 100
_C.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_ZERO_PADS = False
_C.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_HIDDEN_DIM = 64
_C.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT = 1.0

# Shallow Contrastive Loss (BACKBONE)
_C.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS = False

_C.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER = False

# use checkpoint to save memory
_C.MODEL.DYHEAD.USE_CHECKPOINT = False

# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.USE_FPN = False
# Base RPN anchor sizes given in absolute pixels w.r.t. the scaled network input
_C.MODEL.RPN.ANCHOR_SIZES = (32, 64, 128, 256, 512)
# Stride of the feature map that RPN is attached.
# For FPN, number of strides should match number of scales
_C.MODEL.RPN.ANCHOR_STRIDE = (16,)
# RPN anchor aspect ratios
_C.MODEL.RPN.ASPECT_RATIOS = (0.5, 1.0, 2.0)
# Anchor shift away ration from the center for r,t,l,d
_C.MODEL.RPN.ANCHOR_SHIFT = (0.0, 0.0, 0.0, 0.0)
# Use center to decide anchor size
_C.MODEL.RPN.USE_RELATIVE_SIZE = False
# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.STRADDLE_THRESH = 0
# Anchor scales per octave for complex anchors
_C.MODEL.RPN.OCTAVE = 2.0
_C.MODEL.RPN.SCALES_PER_OCTAVE = 3
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example)
_C.MODEL.RPN.FG_IOU_THRESHOLD = 0.7
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example)
_C.MODEL.RPN.BG_IOU_THRESHOLD = 0.3
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOP_N_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOP_N_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
_C.MODEL.RPN.POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOP_N_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.RPN.MIN_SIZE = 0
# Number of top scoring RPN proposals to keep after combining proposals from
# all FPN levels
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TRAIN = 2000
_C.MODEL.RPN.FPN_POST_NMS_TOP_N_TEST = 2000
# Custom rpn head, empty to use default conv or separable conv
_C.MODEL.RPN.RPN_HEAD = "SingleConvRPNHead"
_C.MODEL.RPN.FREEZE = False
_C.MODEL.RPN.FORCE_BOXES = False
_C.MODEL.RPN.RETURN_FUSED_FEATURES = False

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.USE_FPN = False
# Overlap threshold for an RoI to be considered foreground (if >= FG_IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.FG_IOU_THRESHOLD = 0.5
# Overlap threshold for an RoI to be considered background
# (class = 0 if overlap in [0, BG_IOU_THRESHOLD))
_C.MODEL.ROI_HEADS.BG_IOU_THRESHOLD = 0.5
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
# E.g., a common configuration is: 512 * 2 * 8 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
_C.MODEL.ROI_HEADS.SCORE_THRESH = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS = 0.5
# Maximum number of detections to return per image (100 is based on the limit
# established for the COCO dataset)
_C.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 100

_C.MODEL.ROI_BOX_HEAD = CN()
_C.MODEL.ROI_BOX_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_BOX_HEAD.PREDICTOR = "FastRCNNPredictor"
_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_BOX_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_BOX_HEAD.NUM_CLASSES = 81
# Hidden layer dimension when using an MLP for the RoI box head
_C.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM = 1024
# GN
_C.MODEL.ROI_BOX_HEAD.USE_GN = False
# Dilation
_C.MODEL.ROI_BOX_HEAD.DILATION = 1
_C.MODEL.ROI_BOX_HEAD.CONV_HEAD_DIM = 256
_C.MODEL.ROI_BOX_HEAD.NUM_STACKED_CONVS = 4
# Use D2 style ROIAlignV2
_C.MODEL.ROI_BOX_HEAD.POOLER_ALIGNED = False

_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.FEATURE_EXTRACTOR = "ResNet50Conv5ROIFeatureExtractor"
_C.MODEL.ROI_MASK_HEAD.PREDICTOR = "MaskRCNNC4Predictor"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_MASK_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_MASK_HEAD.CONV_LAYERS = (256, 256, 256, 256)
_C.MODEL.ROI_MASK_HEAD.RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True
# Whether or not resize and translate masks to the input image.
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS = False
_C.MODEL.ROI_MASK_HEAD.POSTPROCESS_MASKS_THRESHOLD = 0.5
# Dilation
_C.MODEL.ROI_MASK_HEAD.DILATION = 1
# GN
_C.MODEL.ROI_MASK_HEAD.USE_GN = False
# HG
_C.MODEL.ROI_MASK_HEAD.HG_SCALE = 1

_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.FEATURE_EXTRACTOR = "KeypointRCNNFeatureExtractor"
_C.MODEL.ROI_KEYPOINT_HEAD.PREDICTOR = "KeypointRCNNPredictor"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SCALES = (1.0 / 16,)
_C.MODEL.ROI_KEYPOINT_HEAD.MLP_HEAD_DIM = 1024
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_LAYERS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_CLASSES = 17
_C.MODEL.ROI_KEYPOINT_HEAD.KEYPOINT_NAME = ()  # If left empty, use default names
_C.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR = True

# ---------------------------------------------------------------------------- #
# ResNe[X]t options (ResNets = {ResNet, ResNeXt}
# Note that parts of a resnet may be used for both the backbone and the head
# These options apply to both
# ---------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.USE_STEM3X3 = False
_C.MODEL.RESNETS.WITH_SE = False
_C.MODEL.RESNETS.USE_AVG_DOWN = False

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Baseline width of each group
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for C2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Residual transformation function
_C.MODEL.RESNETS.TRANS_FUNC = "BottleneckWithFixedBatchNorm"
# ResNet's stem function (conv1 and pool1)
_C.MODEL.RESNETS.STEM_FUNC = "StemWithFixedBatchNorm"

# Apply dilation in stage "res5"
_C.MODEL.RESNETS.RES5_DILATION = 1

_C.MODEL.RESNETS.BACKBONE_OUT_CHANNELS = 256 * 4
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

_C.MODEL.RESNETS.REVISION = "resnet_light"
# Deformable convolutions
_C.MODEL.RESNETS.STAGE_WITH_DCN = (False, False, False, False)
_C.MODEL.RESNETS.WITH_MODULATED_DCN = False
_C.MODEL.RESNETS.DEFORMABLE_GROUPS = 1

# ---------------------------------------------------------------------------- #
# Swin Transformer
# ---------------------------------------------------------------------------- #
_C.MODEL.SWINT = CN()
_C.MODEL.SWINT.EMBED_DIM = 96
_C.MODEL.SWINT.OUT_CHANNELS = (96, 192, 384, 768)
_C.MODEL.SWINT.DEPTHS = (2, 2, 6, 2)
_C.MODEL.SWINT.NUM_HEADS = (3, 6, 12, 24)
_C.MODEL.SWINT.WINDOW_SIZE = 7
_C.MODEL.SWINT.MLP_RATIO = 4
_C.MODEL.SWINT.DROP_PATH_RATE = 0.2
_C.MODEL.SWINT.APE = False
_C.MODEL.SWINT.VERSION = "v1"
_C.MODEL.SWINT.OUT_NORM = True
_C.MODEL.SWINT.LAYER_SCALE = 0

# ---------------------------------------------------------------------------- #
# CVT SPEC
# ---------------------------------------------------------------------------- #
_C.MODEL.SPEC = CN(new_allowed=True)

# ---------------------------------------------------------------------------- #
# CLIP SPEC
# ---------------------------------------------------------------------------- #
_C.MODEL.CLIP = CN()
_C.MODEL.CLIP.CONTEXT_LENGTH = 256  # default 77
_C.MODEL.CLIP.WIDTH = 512
_C.MODEL.CLIP.LAYERS = 12
_C.MODEL.CLIP.HEADS = 8
_C.MODEL.CLIP.DROP_PATH = 0.0
_C.MODEL.CLIP.TOKENIZER = "clip"
_C.MODEL.CLIP.VOCAB_SIZE = 49408

# ---------------------------------------------------------------------------- #
# SEARCH
# ---------------------------------------------------------------------------- #

_C.SEARCH = CN()
_C.SEARCH.MAX_EPOCH = 20
_C.SEARCH.SELECT_NUM = 20
_C.SEARCH.POPULATION_NUM = 64
_C.SEARCH.MUTATION_NUM = 24
_C.SEARCH.CROSSOVER_NUM = 24
_C.SEARCH.MUTATION_PROB = 0.1

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.USE_AMP = False

_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.MULTI_MAX_ITER = ()  # set different max epoch for different stage
_C.SOLVER.MAX_EPOCH = 0  # any epoch number>0 will overwrite max_iter
_C.SOLVER.MULTI_MAX_EPOCH = ()  # set different max epoch for different stage

_C.SOLVER.OPTIMIZER = "SGD"  # "ADAMW"

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.LANG_LR = 0.00001
_C.SOLVER.BACKBONE_BODY_LR_FACTOR = 1.0

_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.GRAD_CLIP = 0.0
# D2 gradient clip
_C.SOLVER.CLIP_GRADIENTS = CN()
_C.SOLVER.CLIP_GRADIENTS.ENABLED = False
_C.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.0
_C.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "full_model"
_C.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0
_C.SOLVER.MODEL_EMA = 0.0

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0
_C.SOLVER.WEIGHT_DECAY_NORM_FACTOR = 1.0

# use cosine lr to replace default multistage
_C.SOLVER.USE_COSINE = False
_C.SOLVER.MIN_LR = 0.000001

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.USE_AUTOSTEP = False
_C.SOLVER.STEP_PATIENCE = 5

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 2500
_C.SOLVER.CHECKPOINT_PER_EPOCH = -1.0
_C.SOLVER.TEST_WITH_INFERENCE = False
_C.SOLVER.AUTO_TERMINATE_PATIENCE = -1
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 16
# This is the max negative ratio allowed per batch
_C.SOLVER.MAX_NEG_PER_BATCH = 0.1

_C.SOLVER.SEED = 0
_C.SOLVER.DISABLE_OUTPUT_DISTRIBUTED = False


_C.SOLVER.PROMPT_PROBING_LEVEL = -1.0 
# -1 means tuning the whole model; 
# 1 means tuning the whole language model; 1.5 means tuning the box head as well

_C.SOLVER.FIND_UNUSED_PARAMETERS = True
_C.SOLVER.DATASET_LENGTH = -1 # Just for logging purpose
_C.SOLVER.TUNING_HIGHLEVEL_OVERRIDE = None
_C.SOLVER.USE_EMA_FOR_MONITOR = False

_C.SOLVER.WEIGHT_DECAY_SCHEDULE = False
_C.SOLVER.WEIGHT_DECAY_SCHEDULE_RATIO = 0.667

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.EXPECTED_RESULTS = []
_C.TEST.EXPECTED_RESULTS_SIGMA_TOL = 4
_C.TEST.DURING_TRAINING = False
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.IMS_PER_BATCH = 16
# Special Test Configuration
_C.TEST.USE_MULTISCALE = False
# _C.TEST.SCALES = (400, 600, 800, 1000, 1200, 1400)
# _C.TEST.RANGES = ((96, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 192))
_C.TEST.SCALES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800)
_C.TEST.RANGES = ((96, 10000), (96, 10000), (64, 10000), (64, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256), (0, 256), (0, 192), (0, 192), (0, 96))
_C.TEST.MAX_SIZE = 2500
_C.TEST.FLIP = True
_C.TEST.SPECIAL_NMS = 'none'  # ('none', 'soft-nms', 'vote', 'soft-vote')
_C.TEST.TH = 0.6  # threshold for nms or vote
_C.TEST.PRE_NMS_TOP_N = 1000
_C.TEST.NUM_CLASSES = 81
_C.TEST.SELECT_CLASSES = ()

_C.TEST.EVAL_TASK = ""
_C.TEST.SUBSET = -1
_C.TEST.CHUNKED_EVALUATION = -1
_C.TEST.MDETR_STYLE_AGGREGATE_CLASS_NUM = -1
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "OUTPUT"

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")

# TensorBoard experiment location
_C.TENSORBOARD_EXP = "OUTPUT"


_C.GLIPKNOW = CN()
_C.GLIPKNOW.KNOWLEDGE_FILE = ""
_C.GLIPKNOW.KNOWLEDGE_TYPE = ""
_C.GLIPKNOW.MAX_NUM_CLASSES_PER_BATCH_TRAIN = -1
_C.GLIPKNOW.PARALLEL_LANGUAGE_INPUT = False
_C.GLIPKNOW.LAN_FEATURE_AGG_TYPE = "first"
_C.GLIPKNOW.GPT3_NUM = 5
_C.GLIPKNOW.WIKI_AND_GPT3 = False