ANN2SNN蒸馏的初步实现 
注意，小模型的训练没有微调环节，也没有SSC模块 目前仅定义了一个参数量较小的简单模型 后期可以再做调整
以上工作和预训练参数均参考了E-spikeformer: [here](https://github.com/BICLab/Spike-Driven-Transformer-V3).

### Results on Imagenet-1K

Trained weights of 5.1M: [here](https://drive.google.com/file/d/1LMkOTPehDNpQE79bvB7jFTf6UzDjpAHQ/view?usp=drive_link).

Trained weights of 10M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).

Trained weights of  19M:  [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).

Others weights are coming soon.
### Train 

Train:

```shell
torchrun   main_finetune.py \
    --model V3_DeiT_Hybrid_l \
    --finetune /path/to/your/converted_checkpoint.pth \
    --data_path /path/to/your/imagenet \
    --output_dir /path/to/your/output_directory \
    --log_dir /path/to/your/log_directory \
    --batch_size 64 \
    --epochs 200 \
    --blr 6e-4 \
    --warmup_epochs 10 \
    --teacher_model regnety_160 \
    --lambda_dist 0.5

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
