import torch
import argparse
import models  # 导入您定义模型的 models.py 文件


def convert_state_dict(v3_state_dict, new_model_state_dict):
    """
    执行“参数手术”，将 V3 state_dict 转换为 Hybrid 模型的 state_dict。
    Args:
        v3_state_dict (dict): 从 V3 检查点加载的 state_dict。
        new_model_state_dict (dict): 一个空的 Hybrid 模型实例的 state_dict，用作模板。
    Returns:
        dict: 填充了转换后权重的新 state_dict。
    """
    new_state_dict_converted = new_model_state_dict.copy()
    num_converted = 0

    # 遍历所有旧模型的权重
    for old_key, old_value in v3_state_dict.items():
        # --- 情况一: 名字和形状都完全匹配 (主要是卷积前端) ---
        if old_key in new_state_dict_converted and old_value.shape == new_state_dict_converted[old_key].shape:
            new_state_dict_converted[old_key] = old_value
            print(f"[Direct Copy] Copied: {old_key}")
            num_converted += 1
            continue

    # --- 情况二: 手动映射和转换 Transformer 后端 ---
    # 假设 Stage 3 有 6 个块，Stage 4 有 2 个块
    for stage_num, num_blocks, old_block_name, new_block_name in [(3, 6, 'block3', 'stage3_blocks'),
                                                                  (4, 2, 'block4', 'stage4_blocks')]:
        for i in range(num_blocks):
            # 1. 转换注意力层 (Attention) 的 QKV 权重
            # 旧 key 示例: block3.0.attn.q_conv.0.weight
            # 新 key 示例: stage3_blocks.0.attn.qkv.weight
            old_q_w_key = f'{old_block_name}.{i}.attn.q_conv.0.weight'
            old_k_w_key = f'{old_block_name}.{i}.attn.k_conv.0.weight'
            old_v_w_key = f'{old_block_name}.{i}.attn.v_conv.0.weight'
            new_qkv_w_key = f'{new_block_name}.{i}.attn.qkv.weight'

            if all(k in v3_state_dict for k in [old_q_w_key, old_k_w_key, old_v_w_key]):
                # V3 的 QKV 是 Conv2d(D, D, 1)，权重形状 [D, D, 1, 1]
                # Hybrid 的 QKV 是 Linear(D, 3*D)，权重形状 [3*D, D]
                q_w = v3_state_dict[old_q_w_key].squeeze()  # 形状变为 [D, D]
                k_w = v3_state_dict[old_k_w_key].squeeze()
                v_w = v3_state_dict[old_v_w_key].squeeze()

                # 拼接成新的 QKV 权重
                new_qkv_w = torch.cat([q_w, k_w, v_w], dim=0)  # 形状变为 [3*D, D]

                if new_qkv_w.shape == new_state_dict_converted[new_qkv_w_key].shape:
                    new_state_dict_converted[new_qkv_w_key] = new_qkv_w
                    print(f"[Surgery] Converted QKV weights for: {new_qkv_w_key}")
                    num_converted += 1
                else:
                    print(f"[Warning] Shape mismatch for QKV weights: {new_qkv_w_key}")

            # (可选) 转换 QKV 的 bias (如果存在)
            # ... 逻辑与权重类似 ...

            # 2. 转换 MLP 层
            # 旧 key 示例: block3.0.mlp.fc1_conv.weight
            # 新 key 示例: stage3_blocks.0.mlp.fc1.weight
            for mlp_layer_num in [1, 2]:
                old_mlp_w_key = f'{old_block_name}.{i}.mlp.fc{mlp_layer_num}_conv.weight'
                new_mlp_w_key = f'{new_block_name}.{i}.mlp.fc{mlp_layer_num}.weight'

                if old_mlp_w_key in v3_state_dict:
                    # V3 的 MLP 是 Conv1d(C_in, C_out, 1)，权重形状 [C_out, C_in, 1]
                    # Hybrid 的 MLP 是 Linear(C_in, C_out)，权重形状 [C_out, C_in]
                    old_w = v3_state_dict[old_mlp_w_key].squeeze()  # 形状变为 [C_out, C_in]
                    if old_w.shape == new_state_dict_converted[new_mlp_w_key].shape:
                        new_state_dict_converted[new_mlp_w_key] = old_w
                        print(f"[Surgery] Converted MLP weights for: {new_mlp_w_key}")
                        num_converted += 1
                    else:
                        print(f"[Warning] Shape mismatch for MLP weights: {new_mlp_w_key}")

            # 注意：V3 的 MS_Block_Spike_SepConv 中的 self.conv (SepConv_Spike) 在新模型中没有对应项，其权重会被自动忽略。

    print(f"\nConversion finished. Total {num_converted} parameter groups converted.")
    return new_state_dict_converted


def main():
    parser = argparse.ArgumentParser(description='Convert V3 checkpoint to V3-DeiT-Hybrid checkpoint')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the original V3 checkpoint file.')
    parser.add_argument('--dest_path', type=str, required=True, help='Path to save the new converted checkpoint file.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the new Hybrid model factory function in models.py.')
    args = parser.parse_args()

    print("--- Starting Checkpoint Conversion ---")

    # 1. 加载原始 V3 模型的 state_dict
    print(f"Loading source checkpoint from: {args.source_path}")
    source_checkpoint = torch.load(args.source_path, map_location='cpu')
    v3_state_dict = source_checkpoint['model']
    print(f"Source model has {len(v3_state_dict)} parameter groups.")

    # 2. 创建一个新的 Hybrid 模型实例，以获取其参数结构
    print(f"Creating new model instance: {args.model_name}")
    new_model = models.__dict__[args.model_name]()
    new_model_state_dict = new_model.state_dict()
    print(f"New model has {len(new_model_state_dict)} parameter groups.")

    # 3. 执行转换
    print("\n--- Performing Parameter Surgery ---")
    converted_state_dict = convert_state_dict(v3_state_dict, new_model_state_dict)

    # 4. (验证) 尝试将转换后的权重加载到新模型中
    print("\n--- Verifying converted checkpoint ---")
    msg = new_model.load_state_dict(converted_state_dict, strict=False)
    print("Loading message:")
    print("Missing keys:", msg.missing_keys)  # 应该主要是 token, pos_embed 和 head
    print("Unexpected keys:", msg.unexpected_keys)  # 应该为空，或为旧模型中被舍弃的层

    # 5. 保存新的检查点文件
    print(f"\nSaving new checkpoint to: {args.dest_path}")
    torch.save({'model': converted_state_dict}, args.dest_path)
    print("--- Conversion Successful! ---")


if __name__ == '__main__':
    main()