# debug_feat_kd.py
import torch, torch.nn as nn, torch.nn.functional as F
import importlib

# ===== 配置（按需改） =====
STU_MODULE = 'models'                 # 你的学生模型所在模块
STU_CLASS  = 'V3_DeiT_Hybrid_l'       # 学生类名（或工厂）
TA_MODULE  = 'spikformer'             # 助教模块
TA_CLASS   = 'spikformer12_768'       # 助教类名/工厂
IMG_SIZE   = 224
FEAT_LAYERS = ['conv2_2','stage3_3','stage4_last']
# =========================

# 1) 构建模型（CPU）
stu_mod = importlib.import_module(STU_MODULE)
ta_mod  = importlib.import_module(TA_MODULE)
Stu = getattr(stu_mod, STU_CLASS)
Ta  = getattr(ta_mod,  TA_CLASS)

# 兼容“类/工厂”的两种写法
import inspect
stu = Stu() if not inspect.isclass(Stu) else Stu(num_classes=1000)
ta  = Ta()  if not inspect.isclass(Ta)  else Ta(num_classes=1000)

stu.eval().to('cpu')
ta.eval().to('cpu')
for p in ta.parameters(): p.requires_grad = False

# 2) 注册 hooks
stu_feats, ta_feats = {}, {}
def hook(bag,name):
    def fn(_m,_i,o): bag[name]=o
    return fn

# 学生（V3_DeiT_Hybrid_l 的命名）
stu.ConvBlock2_2[0].register_forward_hook(hook(stu_feats, 'conv2_2'))
stu.stage3_blocks[3].register_forward_hook(hook(stu_feats, 'stage3_3'))
stu.stage4_blocks[-1].register_forward_hook(hook(stu_feats, 'stage4_last'))

# 助教（spikformer.py 的命名）
ta.ConvBlock2_2[0].register_forward_hook(hook(ta_feats, 'conv2_2'))
ta.block3[3].register_forward_hook(hook(ta_feats, 'stage3_3'))
ta.block3[-1].register_forward_hook(hook(ta_feats, 'stage4_last'))

# 3) 造一批小数据，前向触发 hooks
x = torch.randn(2,3,IMG_SIZE,IMG_SIZE)
with torch.no_grad():
    _ = stu(x)
    _ = ta(x)

print('[featKD][hooks] student keys:', list(stu_feats.keys()))
print('[featKD][hooks] TA      keys:', list(ta_feats.keys()))
for k in FEAT_LAYERS:
    print(f'[featKD][shape] {k}: student={tuple(stu_feats[k].shape)} , TA={tuple(ta_feats[k].shape)}')

# 4) 自动建投影头并计算 feat_loss
proj2d, proj1d = nn.ModuleDict(), nn.ModuleDict()
for k in FEAT_LAYERS:
    s,t = stu_feats[k], ta_feats[k]
    if s.dim()==4 and t.dim()==4:
        proj2d[k] = nn.Conv2d(s.shape[1], t.shape[1], kernel_size=1, bias=False)
    elif s.dim()==3 and t.dim()==3:
        proj1d[k] = nn.Linear(s.shape[-1], t.shape[-1], bias=False)
    else:
        raise RuntimeError(f'shape mismatch @ {k}: {s.shape} vs {t.shape}')

feat_loss = 0.0
for k in FEAT_LAYERS:
    s,t = stu_feats[k], ta_feats[k]
    if s.dim()==4:
        s_m = proj2d[k](s)
    else:
        s_m = proj1d[k](s)
    feat_loss = feat_loss + F.mse_loss(s_m, t, reduction='mean')

print(f'[featKD][loss] feat_loss={feat_loss.item():.6f}')
print('[featKD] ✅ Feature-level KD path is wired correctly.')
