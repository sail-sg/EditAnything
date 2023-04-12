
import time
import pickle
import logging
import os
import numpy as np
import torch
import torch.nn as nn


from collections import OrderedDict
from yaml import safe_dump
from yacs.config import load_cfg, CfgNode#, _to_dict
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.inference import _accumulate_predictions_from_multiple_gpus
from maskrcnn_benchmark.modeling.backbone.nas import get_layer_name
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, get_world_size, all_gather
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from maskrcnn_benchmark.utils.flops import profile


choice = lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))


def gather_candidates(all_candidates):
    all_candidates = all_gather(all_candidates)
    all_candidates = [cand for candidates in all_candidates for cand in candidates]
    return list(set(all_candidates))


def gather_stats(all_candidates):
    all_candidates = all_gather(all_candidates)
    reduced_statcs = {}
    for candidates in all_candidates:
        reduced_statcs.update(candidates) # will replace the existing key with last value if more than one exists
    return reduced_statcs


def compute_on_dataset(model, rngs, data_loader, device=cfg.MODEL.DEVICE):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(data_loader):
        images, targets, image_ids = batch
        with torch.no_grad():
            output = model(images.to(device), rngs=rngs)
            output = [o.to(cpu_device) for o in output]
        results_dict.update(
            {img_id: result for img_id, result in zip(image_ids, output)}
        )
    return results_dict


def bn_statistic(model, rngs, data_loader, device=cfg.MODEL.DEVICE, max_iter=500):
    for name, param in model.named_buffers():
        if 'running_mean' in name:
            nn.init.constant_(param, 0)
        if 'running_var' in name:
            nn.init.constant_(param, 1)

    model.train()
    for iteration, (images, targets, _) in enumerate(data_loader, 1):
        images = images.to(device)
        targets = [target.to(device) for target in targets]
        with torch.no_grad():
            loss_dict = model(images, targets, rngs)
        if iteration >= max_iter:
            break

    return model


def inference(
        model,
        rngs,
        data_loader,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):

    # convert to a torch.device for efficiency
    device = torch.device(device)
    dataset = data_loader.dataset
    predictions = compute_on_dataset(model, rngs, data_loader, device)
    # wait for all processes to complete before measuring the time
    synchronize()

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)


def fitness(cfg, model, rngs, val_loaders):
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    for data_loader_val in val_loaders:
        results = inference(
            model,
            rngs,
            data_loader_val,
            iou_types=iou_types,
            box_only=False,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
        )
        synchronize()

    return results


class EvolutionTrainer(object):
    def __init__(self, cfg, model, flops_limit=None, is_distributed=True):

        self.log_dir = cfg.OUTPUT_DIR
        self.checkpoint_name = os.path.join(self.log_dir,'evolution.pth')
        self.is_distributed = is_distributed

        self.states = model.module.mix_nums if is_distributed else model.mix_nums
        self.supernet_state_dict = pickle.loads(pickle.dumps(model.state_dict()))
        self.flops_limit = flops_limit
        self.model = model

        self.candidates = []
        self.vis_dict = {}

        self.max_epochs = cfg.SEARCH.MAX_EPOCH
        self.select_num = cfg.SEARCH.SELECT_NUM
        self.population_num = cfg.SEARCH.POPULATION_NUM/get_world_size()
        self.mutation_num = cfg.SEARCH.MUTATION_NUM/get_world_size()
        self.crossover_num = cfg.SEARCH.CROSSOVER_NUM/get_world_size()
        self.mutation_prob = cfg.SEARCH.MUTATION_PROB/get_world_size()

        self.keep_top_k = {self.select_num:[], 50:[]}
        self.epoch=0
        self.cfg = cfg

    def save_checkpoint(self):
        if not is_main_process():
            return
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info = {}
        info['candidates'] = self.candidates
        info['vis_dict'] = self.vis_dict
        info['keep_top_k'] = self.keep_top_k
        info['epoch'] = self.epoch
        torch.save(info, self.checkpoint_name)
        print('Save checkpoint to', self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        info = torch.load(self.checkpoint_name)
        self.candidates = info['candidates']
        self.vis_dict = info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']
        print('Load checkpoint from', self.checkpoint_name)
        return True

    def legal(self, cand):
        assert isinstance(cand,tuple) and len(cand)==len(self.states)
        if cand in self.vis_dict:
            return False

        if self.flops_limit is not None:
            net = self.model.module.backbone if self.is_distributed else self.model.backbone
            inp = (1, 3, 224, 224)
            flops, params = profile(net, inp, extra_args={'paths': list(cand)})
            flops = flops/1e6
            print('flops:',flops)
            if flops>self.flops_limit:
                return False

        return True

    def update_top_k(self, candidates, *, k, key, reverse=False):
        assert k in self.keep_top_k
        # print('select ......')
        t = self.keep_top_k[k]
        t += candidates
        t.sort(key=key,reverse=reverse)
        self.keep_top_k[k]=t[:k]

    def eval_candidates(self, train_loader, val_loader):
        for cand in self.candidates:
            t0 = time.time()

            # load back supernet state dict
            self.model.load_state_dict(self.supernet_state_dict)
            # bn_statistic
            model = bn_statistic(self.model, list(cand), train_loader)
            # fitness
            evals = fitness(cfg, model, list(cand), val_loader)

            if is_main_process():
                acc = evals[0].results['bbox']['AP']
                self.vis_dict[cand] = acc
                print('candiate ', cand)
                print('time: {}s'.format(time.time() - t0))
                print('acc ', acc)

    def stack_random_cand(self, random_func, *, batchsize=10):
        while True:
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                yield cand

    def random_can(self, num):
        # print('random select ........')
        candidates = []
        cand_iter = self.stack_random_cand(lambda:tuple(np.random.randint(i) for i in self.states))
        while len(candidates)<num:
            cand = next(cand_iter)

            if not self.legal(cand):
                continue
            candidates.append(cand)
            #print('random {}/{}'.format(len(candidates),num))

        # print('random_num = {}'.format(len(candidates)))
        return candidates

    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        # print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num*10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            for i in range(len(self.states)):
                if np.random.random_sample()<m_prob:
                    cand[i] = np.random.randint(self.states[i])
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            cand = next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            #print('mutation {}/{}'.format(len(res),mutation_num))
            max_iters-=1

        # print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        # print('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num

        def random_func():
            p1=choice(self.keep_top_k[k])
            p2=choice(self.keep_top_k[k])
            return tuple(choice([i,j]) for i,j in zip(p1,p2))

        cand_iter = self.stack_random_cand(random_func)
        while len(res)<crossover_num and max_iters>0:
            cand = next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            #print('crossover {}/{}'.format(len(res),crossover_num))
            max_iters-=1

        # print('crossover_num = {}'.format(len(res)))
        return res

    def train(self, train_loader, val_loader):
        logger = logging.getLogger("maskrcnn_benchmark.evolution")

        if not self.load_checkpoint():
            self.candidates = gather_candidates(self.random_can(self.population_num))

        while self.epoch<self.max_epochs:
            self.eval_candidates(train_loader, val_loader)
            self.vis_dict = gather_stats(self.vis_dict)

            self.update_top_k(self.candidates, k=self.select_num, key=lambda x:1-self.vis_dict[x])
            self.update_top_k(self.candidates, k=50, key=lambda x:1-self.vis_dict[x])

            if is_main_process():
                logger.info('Epoch {} : top {} result'.format(self.epoch+1, len(self.keep_top_k[self.select_num])))
                for i,cand in enumerate(self.keep_top_k[self.select_num]):
                    logger.info('     No.{} {} perf = {}'.format(i+1, cand, self.vis_dict[cand]))

            mutation = gather_candidates(self.get_mutation(self.select_num, self.mutation_num, self.mutation_prob))
            crossover = gather_candidates(self.get_crossover(self.select_num, self.crossover_num))
            rand = gather_candidates(self.random_can(self.population_num - len(mutation) - len(crossover)))

            self.candidates = mutation + crossover + rand

            self.epoch+=1
            self.save_checkpoint()

    def save_candidates(self, cand, template):
        paths = self.keep_top_k[self.select_num][cand-1]

        with open(template, "r") as f:
            super_cfg = load_cfg(f)

        search_spaces = {}
        for mix_ops in super_cfg.MODEL.BACKBONE.LAYER_SEARCH:
            search_spaces[mix_ops] = super_cfg.MODEL.BACKBONE.LAYER_SEARCH[mix_ops]
        search_layers = super_cfg.MODEL.BACKBONE.LAYER_SETUP

        layer_setup = []
        for i, layer in enumerate(search_layers):
            name, setup = get_layer_name(layer, search_spaces)
            if not isinstance(name, list):
                name = [name]
            name = name[paths[i]]

            layer_setup.append("('{}', {})".format(name, str(setup)[1:-1]))
        super_cfg.MODEL.BACKBONE.LAYER_SETUP = layer_setup

        cand_cfg = _to_dict(super_cfg)
        del cand_cfg['MODEL']['BACKBONE']['LAYER_SEARCH']
        with open(os.path.join(self.cfg.OUTPUT_DIR, os.path.basename(template)).replace('.yaml','_cand{}.yaml'.format(cand)), 'w') as f:
            f.writelines(safe_dump(cand_cfg))

        super_weight = self.supernet_state_dict
        cand_weight = OrderedDict()
        cand_keys = ['layers.{}.ops.{}'.format(i, c) for i, c in enumerate(paths)]

        for key, val in super_weight.items():
            if 'ops' in key:
                for ck in cand_keys:
                    if ck in key:
                        cand_weight[key.replace(ck,ck.split('.ops.')[0])] = val
            else:
                cand_weight[key] = val

        torch.save({'model':cand_weight}, os.path.join(self.cfg.OUTPUT_DIR, 'init_cand{}.pth'.format(cand)))
