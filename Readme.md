ANN2SNN蒸馏的初步实现 
注意，小模型的训练没有微调环节，也没有SSC模块 目前仅定义了一个参数量较小的简单模型 后期可以再做调整
以上工作和预训练参数均参考了E-spikeformer: [here](https://github.com/BICLab/Spike-Driven-Transformer-V3).

### Results on Imagenet-1K

Trained weights of 5.1M: [here](https://drive.google.com/file/d/1LMkOTPehDNpQE79bvB7jFTf6UzDjpAHQ/view?usp=drive_link).

Trained weights of 10M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).

Trained weights of  19M:  [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).

Others weights are coming soon.

### Results on Imagenet-1K

Trained weights of 171M_1x4: [here](https://drive.google.com/file/d/1sJAjirbjVaB7gLSybvy2Xz2wwQl6gZk7/view?usp=sharing).

Trained weights of 171M_1x8: [here](https://drive.google.com/file/d/18bcS2jQD41JyoJAW9lhZOkTgUb79uShf/view?usp=sharing).

Trained weights of  171M_1x8_384: [here](https://drive.google.com/file/d/1ooNGJRTi869e0ApZm8Oc84Mq02uXXyA8/view?usp=sharing).


Trained weights of 83M_1x4: [here](https://drive.google.com/file/d/1f9pFflYcMacnYJc2u8cHcgMqdibv8wAO/view?usp=sharing).

Trained weights of 83M_1x8: [here](https://drive.google.com/file/d/1sh4F9LWFbKIgIVa2u0QaixBWIcbDZ7h_/view?usp=sharing).

Trained weights of  83M_1x8_384: [here]().

### convert

convert:

```shell
python convert_checkpoint.py \
    --source_path /path/to/your/original_v3_checkpoint.pth \
    --dest_path /path/to/save/converted_hybrid_checkpoint.pth \
    --model_name V3_DeiT_Hybrid_l

```

### Train 

Train:

```shell
torchrun --nproc_per_node=8 --master_port=29666 \
  main_finetune.py \
  --model V3_DeiT_Hybrid_l \
  --finetune /share/home/ruiqi.zheng/v3/converted_hybrid_checkpoint.pth \
  --data_path /data/datasets/imagenet/ \
  --output_dir /share/home/ruiqi.zheng/v3/deit_out_1/ \
  --log_dir /share/home/ruiqi.zheng/v3/deit_logs_1/ \
  --batch_size 16 --accum_iter 4 \
  --epochs 200 \
  --blr 2e-3 \
  --warmup_epochs 25 \
  --teacher_model regnety_160 \
  --lambda_dist 0.3 \
  --ta_model_module spikformer \
  --ta_model_cls spikformer12_768 \
  --ta_path /share/home/ruiqi.zheng/v3/171M-1x8_86_2.pth \
  --feat_kd_layers conv2_2,stage3_3,stage4_last \
  --feat_kd_w 0.7 \


```




### Data Prepare

ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```
