import os
import os.path as op
import json
# import logging
import base64
import yaml
import errno
import io
import math
from PIL import Image, ImageDraw

from maskrcnn_benchmark.structures.bounding_box import BoxList
from .box_label_loader import LabelLoader


def load_linelist_file(linelist_file):
    if linelist_file is not None:
        line_list = []
        with open(linelist_file, 'r') as fp:
            for i in fp:
                line_list.append(int(i.strip()))
        return line_list


def img_from_base64(imagestring):
    try:
        img = Image.open(io.BytesIO(base64.b64decode(imagestring)))
        return img.convert('RGB')
    except ValueError:
        return None


def load_from_yaml_file(yaml_file):
    with open(yaml_file, 'r') as fp:
        return yaml.load(fp, Loader=yaml.CLoader)


def find_file_path_in_yaml(fname, root):
    if fname is not None:
        if op.isfile(fname):
            return fname
        elif op.isfile(op.join(root, fname)):
            return op.join(root, fname)
        else:
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), op.join(root, fname)
            )


def create_lineidx(filein, idxout):
    idxout_tmp = idxout + '.tmp'
    with open(filein, 'r') as tsvin, open(idxout_tmp, 'w') as tsvout:
        fsize = os.fstat(tsvin.fileno()).st_size
        fpos = 0
        while fpos != fsize:
            tsvout.write(str(fpos) + "\n")
            tsvin.readline()
            fpos = tsvin.tell()
    os.rename(idxout_tmp, idxout)


def read_to_character(fp, c):
    result = []
    while True:
        s = fp.read(32)
        assert s != ''
        if c in s:
            result.append(s[: s.index(c)])
            break
        else:
            result.append(s)
    return ''.join(result)


class TSVFile(object):
    def __init__(self, tsv_file, generate_lineidx=False):
        self.tsv_file = tsv_file
        self.lineidx = op.splitext(tsv_file)[0] + '.lineidx'
        self._fp = None
        self._lineidx = None
        # the process always keeps the process which opens the file.
        # If the pid is not equal to the currrent pid, we will re-open the file.
        self.pid = None
        # generate lineidx if not exist
        if not op.isfile(self.lineidx) and generate_lineidx:
            create_lineidx(self.tsv_file, self.lineidx)

    def __del__(self):
        if self._fp:
            self._fp.close()

    def __str__(self):
        return "TSVFile(tsv_file='{}')".format(self.tsv_file)

    def __repr__(self):
        return str(self)

    def num_rows(self):
        self._ensure_lineidx_loaded()
        return len(self._lineidx)

    def seek(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        try:
            pos = self._lineidx[idx]
        except:
            # logging.info('{}-{}'.format(self.tsv_file, idx))
            raise
        self._fp.seek(pos)
        return [s.strip() for s in self._fp.readline().split('\t')]

    def seek_first_column(self, idx):
        self._ensure_tsv_opened()
        self._ensure_lineidx_loaded()
        pos = self._lineidx[idx]
        self._fp.seek(pos)
        return read_to_character(self._fp, '\t')

    def get_key(self, idx):
        return self.seek_first_column(idx)

    def __getitem__(self, index):
        return self.seek(index)

    def __len__(self):
        return self.num_rows()

    def _ensure_lineidx_loaded(self):
        if self._lineidx is None:
            # logging.info('loading lineidx: {}'.format(self.lineidx))
            with open(self.lineidx, 'r') as fp:
                self._lineidx = [int(i.strip()) for i in fp.readlines()]

    def _ensure_tsv_opened(self):
        if self._fp is None:
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()

        if self.pid != os.getpid():
            # logging.info('re-open {} because the process id changed'.format(self.tsv_file))
            self._fp = open(self.tsv_file, 'r')
            self.pid = os.getpid()


class CompositeTSVFile():
    def __init__(self, file_list, seq_file, root='.'):
        if isinstance(file_list, str):
            self.file_list = load_list_file(file_list)
        else:
            assert isinstance(file_list, list)
            self.file_list = file_list

        self.seq_file = seq_file
        self.root = root
        self.initialized = False
        self.initialize()

    def get_key(self, index):
        idx_source, idx_row = self.seq[index]
        k = self.tsvs[idx_source].get_key(idx_row)
        return '_'.join([self.file_list[idx_source], k])

    def num_rows(self):
        return len(self.seq)

    def __getitem__(self, index):
        idx_source, idx_row = self.seq[index]
        return self.tsvs[idx_source].seek(idx_row)

    def __len__(self):
        return len(self.seq)

    def initialize(self):
        '''
        this function has to be called in init function if cache_policy is
        enabled. Thus, let's always call it in init funciton to make it simple.
        '''
        if self.initialized:
            return
        self.seq = []
        with open(self.seq_file, 'r') as fp:
            for line in fp:
                parts = line.strip().split('\t')
                self.seq.append([int(parts[0]), int(parts[1])])
        self.tsvs = [TSVFile(op.join(self.root, f)) for f in self.file_list]
        self.initialized = True


def load_list_file(fname):
    with open(fname, 'r') as fp:
        lines = fp.readlines()
    result = [line.strip() for line in lines]
    if len(result) > 0 and result[-1] == '':
        result = result[:-1]
    return result


class TSVDataset(object):
    def __init__(self, img_file, label_file=None, hw_file=None,
                 linelist_file=None, imageid2idx_file=None):
        """Constructor.
        Args:
            img_file: Image file with image key and base64 encoded image str.
            label_file: An optional label file with image key and label information.
                A label_file is required for training and optional for testing.
            hw_file: An optional file with image key and image height/width info.
            linelist_file: An optional file with a list of line indexes to load samples.
                It is useful to select a subset of samples or duplicate samples.
        """
        self.img_file = img_file
        self.label_file = label_file
        self.hw_file = hw_file
        self.linelist_file = linelist_file

        self.img_tsv = TSVFile(img_file)
        self.label_tsv = None if label_file is None else TSVFile(label_file, generate_lineidx=True)
        self.hw_tsv = None if hw_file is None else TSVFile(hw_file)
        self.line_list = load_linelist_file(linelist_file)
        self.imageid2idx = None
        if imageid2idx_file is not None:
            self.imageid2idx = json.load(open(imageid2idx_file, 'r'))

        self.transforms = None

    def __len__(self):
        if self.line_list is None:
            if self.imageid2idx is not None:
                assert self.label_tsv is not None, "label_tsv is None!!!"
                return self.label_tsv.num_rows()
            return self.img_tsv.num_rows()
        else:
            return len(self.line_list)

    def __getitem__(self, idx):
        img = self.get_image(idx)
        img_size = img.size  # w, h
        annotations = self.get_annotations(idx)
        # print(idx, annotations)
        target = self.get_target_from_annotations(annotations, img_size, idx)
        img, target = self.apply_transforms(img, target)

        if self.transforms is None:
            return img, target, idx, 1.0
        else:
            new_img_size = img.shape[1:]
            scale = math.sqrt(float(new_img_size[0] * new_img_size[1]) / float(img_size[0] * img_size[1]))
            return img, target, idx, scale

    def get_line_no(self, idx):
        return idx if self.line_list is None else self.line_list[idx]

    def get_image(self, idx):
        line_no = self.get_line_no(idx)
        if self.imageid2idx is not None:
            assert self.label_tsv is not None, "label_tsv is None!!!"
            row = self.label_tsv.seek(line_no)
            annotations = json.loads(row[1])
            imageid = annotations["img_id"]
            line_no = self.imageid2idx[imageid]
        row = self.img_tsv.seek(line_no)
        # use -1 to support old format with multiple columns.
        img = img_from_base64(row[-1])
        return img

    def get_annotations(self, idx):
        line_no = self.get_line_no(idx)
        if self.label_tsv is not None:
            row = self.label_tsv.seek(line_no)
            annotations = json.loads(row[1])
            return annotations
        else:
            return []

    def get_target_from_annotations(self, annotations, img_size, idx):
        # This function will be overwritten by each dataset to
        # decode the labels to specific formats for each task.
        return annotations

    def apply_transforms(self, image, target=None):
        # This function will be overwritten by each dataset to
        # apply transforms to image and targets.
        return image, target

    def get_img_info(self, idx):
        if self.imageid2idx is not None:
            assert self.label_tsv is not None, "label_tsv is None!!!"
            line_no = self.get_line_no(idx)
            row = self.label_tsv.seek(line_no)
            annotations = json.loads(row[1])
            return {"height": int(annotations["img_w"]), "width": int(annotations["img_w"])}

        if self.hw_tsv is not None:
            line_no = self.get_line_no(idx)
            row = self.hw_tsv.seek(line_no)
            try:
                # json string format with "height" and "width" being the keys
                data = json.loads(row[1])
                if type(data) == list:
                    return data[0]
                elif type(data) == dict:
                    return data
            except ValueError:
                # list of strings representing height and width in order
                hw_str = row[1].split(' ')
                hw_dict = {"height": int(hw_str[0]), "width": int(hw_str[1])}
                return hw_dict

    def get_img_key(self, idx):
        line_no = self.get_line_no(idx)
        # based on the overhead of reading each row.
        if self.imageid2idx is not None:
            assert self.label_tsv is not None, "label_tsv is None!!!"
            row = self.label_tsv.seek(line_no)
            annotations = json.loads(row[1])
            return annotations["img_id"]

        if self.hw_tsv:
            return self.hw_tsv.seek(line_no)[0]
        elif self.label_tsv:
            return self.label_tsv.seek(line_no)[0]
        else:
            return self.img_tsv.seek(line_no)[0]


class TSVYamlDataset(TSVDataset):
    """ TSVDataset taking a Yaml file for easy function call
    """

    def __init__(self, yaml_file, root=None, replace_clean_label=False):
        print("Reading {}".format(yaml_file))
        self.cfg = load_from_yaml_file(yaml_file)
        if root:
            self.root = root
        else:
            self.root = op.dirname(yaml_file)
        img_file = find_file_path_in_yaml(self.cfg['img'], self.root)
        label_file = find_file_path_in_yaml(self.cfg.get('label', None),
                                            self.root)
        hw_file = find_file_path_in_yaml(self.cfg.get('hw', None), self.root)
        linelist_file = find_file_path_in_yaml(self.cfg.get('linelist', None),
                                               self.root)
        imageid2idx_file = find_file_path_in_yaml(self.cfg.get('imageid2idx', None),
                                               self.root)

        if replace_clean_label:
            assert ("raw_label" in label_file)
            label_file = label_file.replace("raw_label", "clean_label")

        super(TSVYamlDataset, self).__init__(
            img_file, label_file, hw_file, linelist_file, imageid2idx_file)


class ODTSVDataset(TSVYamlDataset):
    """
    Generic TSV dataset format for Object Detection.
    """

    def __init__(self, yaml_file, extra_fields=(), transforms=None,
                 is_load_label=True, **kwargs):
        if yaml_file is None:
            return
        super(ODTSVDataset, self).__init__(yaml_file)

        self.transforms = transforms
        self.is_load_label = is_load_label
        self.attribute_on = False
        # self.attribute_on = kwargs['args'].MODEL.ATTRIBUTE_ON if "args" in kwargs else False

        if self.is_load_label:
            # construct maps
            jsondict_file = find_file_path_in_yaml(
                self.cfg.get("labelmap", None), self.root
            )
            if jsondict_file is None:
                jsondict_file = find_file_path_in_yaml(
                    self.cfg.get("jsondict", None), self.root
                )
            if "json" in jsondict_file:
                jsondict = json.load(open(jsondict_file, 'r'))
                if "label_to_idx" not in jsondict:
                    jsondict = {'label_to_idx': jsondict}
            elif "tsv" in jsondict_file:
                label_to_idx = {}
                counter = 1
                with open(jsondict_file) as f:
                    for line in f:
                        label_to_idx[line.strip()] = counter
                        counter += 1
                jsondict = {'label_to_idx': label_to_idx}
            else:
                assert (0)

            self.labelmap = {}
            self.class_to_ind = jsondict['label_to_idx']
            self.class_to_ind['__background__'] = 0
            self.ind_to_class = {v: k for k, v in self.class_to_ind.items()}
            self.labelmap['class_to_ind'] = self.class_to_ind

            if self.attribute_on:
                self.attribute_to_ind = jsondict['attribute_to_idx']
                self.attribute_to_ind['__no_attribute__'] = 0
                self.ind_to_attribute = {v: k for k, v in self.attribute_to_ind.items()}
                self.labelmap['attribute_to_ind'] = self.attribute_to_ind

            self.label_loader = LabelLoader(
                labelmap=self.labelmap,
                extra_fields=extra_fields,
            )

    def get_target_from_annotations(self, annotations, img_size, idx):
        if isinstance(annotations, list):
            annotations = {"objects": annotations}
        if self.is_load_label:
            return self.label_loader(annotations['objects'], img_size)

    def apply_transforms(self, img, target=None):
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
