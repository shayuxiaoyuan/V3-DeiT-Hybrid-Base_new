# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import importlib
import inspect

import torch
from util.samplers import RASampler
# import torchinfo
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.utils import accuracy, AverageMeter, ModelEma
# assert timm.__version__ == "0.5.4"  # version check
from timm.models.layers import trunc_normal_
import timm.optim.optim_factory as optim_factory
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import util.lr_decay_spikformer as lrd
import util.misc as misc
from util.datasets import build_dataset
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models
from engine_finetune import train_one_epoch, evaluate
from timm.data import create_loader

from util.kd_loss import DistillationLoss
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy



def get_args_parser():
    # important params
    parser = argparse.ArgumentParser(
        "MAE fine-tuning for image classification", add_help=False
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus",
    )
    parser.add_argument("--epochs", default=200, type=int)  # 20/30(T=4)
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--finetune", default="", help="finetune from checkpoint")
    parser.add_argument(
        "--data_path", default="./data", type=str, help="dataset path"
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="spikformer",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument(
        "--model_mode",
        default="ms",
        type=str,
        help="Mode of model to train",
    )

    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop_path",
        type=float,
        default=0.1,
        metavar="PCT",
        help="Drop path rate (default: 0.1)",
    )
    # Optimizer parameters
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        metavar="LR",
        help="learning rate (absolute lr)",
    )
    parser.add_argument(
        "--blr",
        type=float,
        default=6e-4,
        metavar="LR",  # 1e-5,2e-5(T=4)
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    parser.add_argument(
        "--layer_decay",
        type=float,
        default=1.0,
        help="layer-wise lr decay from ELECTRA/BEiT",
    )

    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0",
    )

    parser.add_argument(
        "--warmup_epochs", type=int, default=10, metavar="N", help="epochs to warmup LR"
    )

    # Augmentation parameters
    parser.add_argument(
        "--color_jitter",
        type=float,
        default=None,
        metavar="PCT",
        help="Color jitter factor (enabled only when not using Auto/RandAug)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)',
    ),
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup", type=float, default=0, help="mixup alpha, mixup enabled if > 0."
    )
    parser.add_argument(
        "--cutmix", type=float, default=0, help="cutmix alpha, cutmix enabled if > 0."
    )
    parser.add_argument(
        "--cutmix_minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup_prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup_switch_prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup_mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # * Finetuning params

    parser.add_argument("--global_pool", action="store_true")
    parser.set_defaults(global_pool=True)
    parser.add_argument(
        "--cls_token",
        action="store_false",
        dest="global_pool",
        help="Use class token instead of global pool for classification",
    )
    parser.add_argument("--time_steps", default=1, type=int)

    # Dataset parameters

    parser.add_argument(
        "--nb_classes",
        default=1000,
        type=int,
        help="number of the classification types",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir",
        default="./output_dir",
        help="path where to tensorboard log",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default=None, help="resume from checkpoint")

    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true",default=False, help="Perform evaluation only")
    parser.add_argument(
        "--repeated_aug",
        action="store_true",
        default=False,
        help="Three aug",
    )
    parser.add_argument(
        "--dist_eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation (recommended during training for faster monitor",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )

    # --- 新增: 基于响应的蒸馏相关参数 ---
    parser.add_argument('--teacher_model', default='regnety_160', type=str,
                        help='Name of teacher model to train (default: regnety_160)')
    parser.add_argument('--teacher_path', default='', type=str,
                        help='path to teacher model checkpoint')
    parser.add_argument('--distillation_type', default='hard', choices=['none', 'soft', 'hard'],
                        help='type of distillation')
    parser.add_argument('--lambda_dist', default=0.5, type=float,
                        help="weight of distillation loss")
    
    # --- 新增: 基于特征的蒸馏相关参数 ---
    parser.add_argument('--ta_model_module', default='', type=str,
                    help='Python module path of TA model, e.g. models_spike_v3')
    parser.add_argument('--ta_model_cls', default='', type=str,
                    help='Class name of TA model, e.g. SpikeV3_170M')
    parser.add_argument('--ta_path', default='', type=str,
                    help='Path to TA pretrained checkpoint (.pth)')
    parser.add_argument('--feat_kd_layers', default='conv2_2,stage3_3,stage4_last', type=str,
                    help='Comma-separated tap names to distill (must match your hook names)')
    parser.add_argument('--feat_kd_w', default=0.5, type=float,
                    help='Weight of feature-level KD loss')


    return parser


def main(args):
    misc.init_distributed_mode(args)

    print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(", ", ",\n"))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset(is_train=True, args=args)
    dataset_val = build_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )


    model = models.__dict__[args.model]()

    # 加载教师模型
    teacher_model = None
    if args.distillation_type != 'none':
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = timm.create_model(
            args.teacher_model,
            pretrained=True,  # 使用timm的预训练权重
            num_classes=args.nb_classes,
        )
        if args.teacher_path:
            checkpoint = torch.load(args.teacher_path, map_location='cpu')
            teacher_model.load_state_dict(checkpoint['model'])

        # 将教师模型移动到设备并设置为评估模式
        teacher_model.to(device)
        teacher_model.eval()

    # 加载助教模型
    ta_model = None
    if getattr(args, 'ta_model_module', '') and getattr(args, 'ta_model_cls', ''):
       import importlib
       ta_mod = importlib.import_module(args.ta_model_module)   # e.g. models_spike_v3
       TaClass = getattr(ta_mod, args.ta_model_cls)             # e.g. SpikeV3_170M
       #ta_model = TaClass(num_classes=args.nb_classes)          # 按你的实现补充必要init参数 
       obj = getattr(ta_mod, args.ta_model_cls)  # 可能是类，也可能是工厂函数
       if inspect.isclass(obj):
         # 直接是类（如 Spikformer）
          ta_model = obj(num_classes=args.nb_classes)
       else:
         # 是工厂函数（如 spikformer12_768）
          sig = inspect.signature(obj)
          if 'num_classes' in sig.parameters:
             ta_model = obj(num_classes=args.nb_classes)
          else:
             ta_model = obj()

       if args.ta_path:
          ckpt = torch.load(args.ta_path, map_location='cpu')
          state = ckpt.get('model', ckpt.get('state_dict', ckpt))
          # 去掉可能的 'module.' 前缀
          from collections import OrderedDict
          new_state = OrderedDict((k.replace('module.', ''), v) for k,v in state.items())
          ta_model.load_state_dict(new_state, strict=False)
          
       for p in ta_model.parameters():
          p.requires_grad = False
       ta_model.eval()
       # 显存紧张建议放 CPU（强烈推荐）；显存充足可放 GPU 或 half 到 GPU
       ta_model.to(device)           # 放到与学生同一张 GPU
       #ta_model.half()               # 半精度（需 AMP 支持）


    model.T = args.time_steps
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        msg = model.load_state_dict(checkpoint_model, strict=False)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print("number of params (M): %.2f" % (n_parameters / 1.0e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    
    # ========= 在“创建优化器之前”做：注册 hooks + 构建投影头 =========
    stu_feats, ta_feats = {}, {}
    def _hook(bag, name):
     def fn(_m, _i, o):
        # 如果 TA 输出是 5D (B,C,T,H,W)，这里先把时间维 T 压掉
        if isinstance(o, torch.Tensor) and o.dim() == 5:
            o = o.mean(dim=2)  # 平均时间维 → (B,C,H,W)
        bag[name] = o
     return fn
    
    # 学生模型：在底层 module 上注册（避免 AttributeError）
    model_without_ddp.ConvBlock2_2[0].register_forward_hook(_hook(stu_feats, 'conv2_2'))
    model_without_ddp.stage3_blocks[3].register_forward_hook(_hook(stu_feats, 'stage3_3'))
    model_without_ddp.stage4_blocks[-1].register_forward_hook(_hook(stu_feats, 'stage4_last'))
    
    # 助教模型（你放在 CPU），保持原样
    if ta_model is not None:
        ta_model.ConvBlock2_2[0].register_forward_hook(_hook(ta_feats, 'conv2_2'))
        ta_model.block3[3].register_forward_hook(_hook(ta_feats, 'stage3_3'))
        ta_model.block3[-1].register_forward_hook(_hook(ta_feats, 'stage4_last'))
    
    # 探测一次获取中间层尺寸
    import torch.nn as nn
    feat_layers = [x.strip() for x in args.feat_kd_layers.split(',') if x.strip()]
    proj_heads_2d = nn.ModuleDict()
    proj_heads_1d = nn.ModuleDict()
    
    samples_probe, _ = next(iter(data_loader_train))
    samples_probe = samples_probe.to(device, non_blocking=True)
    with torch.no_grad():
        if ta_model is not None:
            _ = ta_model(samples_probe.cpu() if next(ta_model.parameters()).device.type=='cpu' else samples_probe)
        _ = model(samples_probe)  # 用 DDP 外壳前向，hooks 仍会触发到底层
        
        def _to_tokens(x: torch.Tensor) -> torch.Tensor:
        # 统一到 (B, N, C)
         if x.dim() == 5:
            # (B,C,T,H,W) 或 (T,B,C,H,W) —— 先压时间维
            if x.shape[2] >= 2 and x.shape[0] in (1, samples_probe.shape[0]):
                x = x.mean(dim=2)                 # (B,C,H,W)
            else:
                x = x.mean(dim=0)                 # (B,C,H,W)
         if x.dim() == 4:
            # (T,B,C,N) 或 (B,C,H,W)
            if x.shape[0] <= 4 and x.shape[-1] >= 49 and int(x.shape[-1] ** 0.5) ** 2 == x.shape[-1]:
                x = x.mean(dim=0) if x.shape[0] > 1 else x.squeeze(0)   # (B,C,N)
                x = x.permute(0, 2, 1)                                   # (B,N,C)
                return x
            else:
                B, C, H, W = x.shape
                x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)           # (B,N,C)
                return x
         elif x.dim() == 3:
            # (B,C,N) -> (B,N,C)
            if x.shape[1] < x.shape[2]:
                x = x.permute(0, 2, 1)
            return x
         else:
            raise RuntimeError(f"Unexpected feature shape: {tuple(x.shape)}")

    for name in feat_layers:
        s = _to_tokens(stu_feats[name])    # (B,Ns,Cs)
        t = _to_tokens(ta_feats[name])     # (B,Nt,Ct)
        Cs, Ct = s.shape[-1], t.shape[-1]
        # 统一走 1D 线性投影（token 维度已对齐到 (B,N,·)）
        proj_heads_1d[name] = nn.Linear(Cs, Ct, bias=False).to(device)


# 把投影头挂在底层模型，确保参数进入 optimizer
    model_without_ddp.add_module('feat_kd_proj2d', proj_heads_2d)
    model_without_ddp.add_module('feat_kd_proj1d', proj_heads_1d)

     # ========= 到此为止，再去“构建 optimizer” =========
            
     
    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(
        model_without_ddp,
        args.weight_decay,
        # no_weight_decay_list=model_without_ddp.no_weight_decay(),
        layer_decay=args.layer_decay,
    )

    optimizer = optim_factory.Lamb(param_groups, trust_clip=True, lr=args.lr)
    loss_scaler = NativeScaler()
    if mixup_fn is not None:
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = torch.nn.CrossEntropyLoss()

    criterion = DistillationLoss(
        base_criterion=base_criterion,
        teacher_model=teacher_model,
        distillation_type=args.distillation_type,
        alpha=args.lambda_dist,
        tau=1.0  # 这个参数在硬蒸馏中不起作用
    )
    print("criterion = %s" % str(criterion))

    misc.load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
    )

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        exit(0)
    



    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    best_acc = 0
    best_epoch = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            # 传递教师模型和蒸馏参数
            teacher_model,
            args.lambda_dist,
            #--------------------
            # ------- 新增：TA特征蒸馏所需 -------
            ta_model, stu_feats, ta_feats,
            feat_layers, model_without_ddp.feat_kd_proj2d, model_without_ddp.feat_kd_proj1d, args.feat_kd_w,
            # -----------------------------------
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            mixup_fn,
            log_writer=log_writer,
            args=args,
        )
        if args.output_dir and (epoch % 50 == 0 or epoch + 1 == args.epochs):
            print("Saving model at epoch:", epoch)
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        test_stats = evaluate(data_loader_val, model, device)
        print(
            f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
        )
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"Max accuracy: {max_accuracy:.2f}%")
        if args.output_dir and test_stats["acc1"] > best_acc:
            print("Saving model at epoch:", epoch)
            misc.save_model(
                args=args,
                model=model,
                model_without_ddp=model_without_ddp,
                optimizer=optimizer,
                loss_scaler=loss_scaler,
                epoch=epoch,
            )

        if log_writer is not None:
            log_writer.add_scalar("perf/test_acc1", test_stats["acc1"], epoch)
            log_writer.add_scalar("perf/test_acc5", test_stats["acc5"], epoch)
            log_writer.add_scalar("perf/test_loss", test_stats["loss"], epoch)

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
