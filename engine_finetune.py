# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import argparse
import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


# 修改 train_one_epoch 的函数定义
def train_one_epoch(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        # 新增参数
        teacher_model: torch.nn.Module,
        lambda_dist: float,
        #------------
        # 新增：特征蒸馏（有默认值，保持兼容）
        ta_model: Optional[torch.nn.Module] = None,
        stu_feats: Optional[dict] = None,
        ta_feats: Optional[dict] = None,
        feat_layers: Optional[list] = None,
        proj_heads_2d: Optional[torch.nn.Module] = None,   # nn.ModuleDict
        proj_heads_1d: Optional[torch.nn.Module] = None,   # nn.ModuleDict
        feat_kd_w: float = 0.0,
        #------------
        data_loader: Iterable = None,
        optimizer: torch.optim.Optimizer = None,
        device: torch.device = None,
        epoch: int = 0,
        loss_scaler = None,
        max_norm: Optional[float] = None,
        mixup_fn: Optional[Mixup] = None,
        log_writer: Optional[SummaryWriter] = None,
        args: argparse.Namespace = None
):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
    ):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_nomix = targets
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            # 新模型会返回包含两个输出的元组：(分类预测, 蒸馏预测)
            outputs = model(samples)

            # 调用在main函数中定义的损失函数
            # 其接收 (原始输入, 模型输出, 真实标签)
            loss = criterion(samples, outputs, targets)
            # ====== Feature-level KD from TA (REPLACE THIS WHOLE BLOCK) ======
            if ta_model is not None and feat_kd_w > 0:
    # 1) 只做 TA 的前向，触发 hooks；TA 在 CPU 就把输入搬到 CPU
              with torch.no_grad():
                ta_dev = next(ta_model.parameters()).device
                ta_inp = samples.to(ta_dev, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=True):
                    _ = ta_model(ta_inp)
    # ---- 辅助函数：把 5D 特征压时间维，统一到 4D/3D ----
              def _squeeze_time(x: torch.Tensor) -> torch.Tensor:
                if torch.is_tensor(x) and x.dim() == 5:
            # 兼容 (B,C,T,H,W) 与 (T,B,C,H,W) 两种约定
            # 经验：你们之前 TA 在 conv2_2 的形状像 (B,C,T,H,W)
                  if x.shape[2] >= 2 and x.shape[0] in (1, samples.shape[0]):
                     return x.mean(dim=2)      # (B,C,H,W)
                  else:
                    return x.mean(dim=0)      # (B,C,H,W)
                return x
              def _to_tokens(x: torch.Tensor) -> torch.Tensor:
                  # 统一为 (B,N,C)
                  if x.dim() == 5:
                      x = _squeeze_time(x)                        # -> (B,C,H,W)
                  if x.dim() == 4:
                      # (T,B,C,N) or (B,C,H,W)
                      if x.shape[0] <= 4 and x.shape[-1] >= 49 and int(x.shape[-1]**0.5)**2 == x.shape[-1]:
                          x = x.mean(dim=0) if x.shape[0] > 1 else x.squeeze(0)   # (B,C,N)
                          x = x.permute(0,2,1)                                     # (B,N,C)
                          return x
                      else:
                          B,C,H,W = x.shape
                          x = x.permute(0,2,3,1).reshape(B, H*W, C)               # (B,N,C)
                          return x
                  if x.dim() == 3:
                      # (B,C,N) -> (B,N,C)
                      if x.shape[1] < x.shape[2]:
                          x = x.permute(0,2,1)
                      return x
                  raise RuntimeError(f"Unexpected feat dim: {tuple(x.shape)}")

              def _drop_special_tokens(feat: torch.Tensor) -> torch.Tensor:
                  # (B,N,C)：常见 786/788 = 784 + 2 (cls,dist)
                  N = feat.shape[1]
                  if N in (197, 198, 785, 786, 788):
                      return feat[:, 2:, :]
                  return feat

              def _pool_to(feat: torch.Tensor, target_N: int) -> torch.Tensor:
                  # (B,N,C) -> (B,target_N,C)，优先 28x28 -> 14x14 的 2x2 平均池化
                  B, N, C = feat.shape
                  if N == target_N:
                      return feat
                  S = int((N) ** 0.5)
                  T = int((target_N) ** 0.5)
                  if S*S == N and T*T == target_N and S % T == 0:
                      k = S // T
                      x = feat.view(B, S, S, C).permute(0,3,1,2)         # (B,C,S,S)
                      x = F.avg_pool2d(x, kernel_size=k, stride=k)       # (B,C,T,T)
                      x = x.permute(0,2,3,1).reshape(B, target_N, C)     # (B,target_N,C)
                      return x
                  # 兜底：截断
                  return feat[:, :target_N, :]

              feat_loss = 0.0
              for name in (feat_layers or []):
                s = None if stu_feats is None else stu_feats.get(name, None)
                t = None if ta_feats  is None else ta_feats.get(name, None)
                if s is None or t is None:
                  continue

        # 2) 压时间维 & 设备/数据类型对齐
                s = _squeeze_time(s)
                t = _squeeze_time(t).to(s.device, dtype=s.dtype)

                if s.dim() == 4 and t.dim() == 4:
            # ---- BCHW 分支：通道映射 + 空间对齐 + 池化成向量 ----
                  if (proj_heads_2d is not None) and (name in proj_heads_2d):
                    s = proj_heads_2d[name](s)  # (B,Ct,H,W)
                  if s.shape[2:] != t.shape[2:]:
                    t = F.adaptive_avg_pool2d(t, output_size=s.shape[2:])
                  s_vec = s.mean(dim=(2, 3))     # (B,C)
                  t_vec = t.mean(dim=(2, 3))     # (B,C)

                elif s.dim() in (3,4,5) or t.dim() in (3,4,5):
            # ---- 统一到 (B,N,C)：去特殊 token + 2x2 池化对齐 + 通道映射 ----
                  s = _to_tokens(s)
                  t = _to_tokens(t)
                  s = _drop_special_tokens(s)
                  # 对齐 token 数（例如 784 -> 196）
                  s = _pool_to(s, target_N=t.shape[1])
                  if (proj_heads_1d is not None) and (name in proj_heads_1d):
                      s = proj_heads_1d[name](s)  # (B, Nt, Ct)
                  # 按 token 取均值得到 (B,C) 再算损失（也可改为逐 token MSE）
                  s_vec = s.mean(dim=1)          # (B, Ct)
                  t_vec = t.mean(dim=1)          # (B, Ct)


                else:
            # 其他维度形状不支持，跳过该层
                 continue

        # 3) 通道交集上计算 MSE（稳妥）
                C = min(s_vec.shape[1], t_vec.shape[1])
                feat_loss = feat_loss + F.mse_loss(s_vec[:, :C], t_vec[:, :C], reduction='mean')

    # 4) 并入总损失
              loss = loss + feat_kd_w * feat_loss
# ====== END Feature-level KD ======


            outputs_acc = outputs[0]

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        torch.cuda.synchronize()
        batch_size = samples.shape[0]
        acc1, acc5 = accuracy(outputs_acc, targets_nomix, topk=(1, 5))

        metric_logger.update(loss=loss_value)
        min_lr = 10.0
        max_lr = 0.0
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar("loss", loss_value_reduce, epoch_1000x)
            log_writer.add_scalar("lr", max_lr, epoch_1000x)

            # ---  快照训练中断点 ---
            #if data_iter_step >= 2000:
            #     print("--- Snapshot training: finishing epoch early at 2000 steps for quick validation. ---")
            #    break
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print(
        "* Train_Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def cal_acc(metric_logger, output, target):
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
    metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    return metric_logger.acc1, metric_logger.acc5


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 500, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            # --- 修改: 模型会返回一个元组，我们只关心第一个分类输出 ---
            # 在eval模式下，我们的新模型只返回一个值，所以这里可能无需修改
            # 但为了代码健壮性，我们假设它可能返回元组
            output = model(images)
            if isinstance(output, tuple):
                output = output[0]  # 只取分类头的输出进行评估

            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
