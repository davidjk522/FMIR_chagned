"""
OASIS-only training entrypoint for regdino_mlp / FMIR.

*** THIS FILE DOES NOT EXIST IN THE PUBLIC FMIR REPO. ***
It is adapted from train_registration_all.py (which only wires up
acdcreg + abdomenreg) to drive the single 'oasisreg' dataset defined in
loaders/oasis_pkl_loader.py, using the OASIS folder shipped in this
checkout (./OASIS/imagesTr, labelsTr, ...).

Known gaps / assumptions carried over or introduced here, please review:
  - models/backbones/layers.py is a best-effort reconstruction (see its
    docstring) since the upstream repo is missing models/backbones/
    entirely. models/backbones/voxelmorph/torch/layers.py is the
    standard, well-known VoxelMorph implementation (low risk).
  - loaders/oasis_pkl_loader.py's train/val split (395/19 subjects,
    neighbor pairing) is an assumption based on common OASIS-L2R practice
    in the registration literature, not something confirmed from the
    paper. If your results disagree with the paper, check this first.
  - `embed_oasis()` below pads each 2D coronal-ish slice (H=160,W=224 by
    default) up to 256x256 before feeding the frozen DINO backbone -- this
    mirrors embed_abdomen()'s padding strategy in train_registration_all.py
    but sized for OASIS's larger volumes. It is NOT verified against the
    paper's actual preprocessing for OASIS (the paper likely only
    describes ACDC/Abdomen in detail).

DINOv3 backbone loading -- IMPORTANT DIFFERENCE FROM train_registration_all.py:
  The weight file actually shipped in ./weight/ is
  dinov3_vits16_pretrain_lvd1689m-08c60483.pth, Meta's ORIGINAL dinov3
  GitHub checkpoint format (verified: the "08c60483" hash matches
  dinov3_vits16's official hash exactly), NOT a HuggingFace `transformers`
  snapshot. `AutoModel.from_pretrained(...)` (what train_registration_all.py
  uses) cannot load this file at all -- the state_dict key names don't
  match HF's Dinov3 module structure.
  This script instead loads it via `torch.hub` against a local clone of
  https://github.com/facebookresearch/dinov3 (see ./third_party/dinov3/),
  using the model's native `get_intermediate_layers(..., reshape=True)`
  API, which was verified end-to-end on this machine (state_dict loads
  with strict=True, dummy forward pass produces the expected
  (1, 384, 14, 14) patch feature map for a 224x224 input).
  Also note: dinov3_vits16's embedding dim is 384, NOT 768 -- the
  original train_registration_all.py hardcodes 768 in a few places
  (apply_pca_to_3d_features_torch, the train-time channel dropout
  `random.sample(range(768), 256)`), which would silently break with
  this checkpoint. This script uses the backbone's real embed_dim
  throughout instead of a hardcoded constant.
"""

import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable

from utils.functions import AverageMeter, registerSTModel, dice_eval, get_downsampled_images
from utils.loss import Grad3d, BinaryDiceLoss, NccLoss
from utils import getters, setters


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(8)
torch.set_default_dtype(torch.float32)

from models.backbones.voxelmorph.torch import layers
from torchvision import transforms

transform_image = transforms.Compose([
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

###########dino backbone (frozen), loaded via torch.hub from a local clone
DINOV3_REPO_DIR = './third_party/dinov3'
DINOV3_WEIGHTS = './weight/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
DINOV3_LAYER_IDX = [2, 5, 8, 11]  # unused for now; embed_oasis() only reads the last block, see note below

backbone = torch.hub.load(DINOV3_REPO_DIR, 'dinov3_vits16', source='local', weights=DINOV3_WEIGHTS).cuda()
backbone.eval()
EMBED_DIM = backbone.embed_dim  # 384 for vits16 -- do NOT hardcode 768 here
for p in backbone.parameters():
    p.requires_grad = False
###########


def embed_oasis_lowres(x, target_size=256, dino_batch_size=32):
    """
    Decompose a (B=1, C, H, W, D) OASIS volume into D 2D slices along the
    last axis, pad/resize each slice for the frozen DINO backbone, run it
    through the backbone, and reassemble a (1, EMBED_DIM, patch_h, patch_w, D)
    feature volume AT THE BACKBONE'S NATIVE PATCH-GRID RESOLUTION (patch_h,
    patch_w ~ 32x32 here, NOT yet upsampled to (H, W)).

    Uses dinov3's native get_intermediate_layers(reshape=True) instead of
    HF's output_hidden_states/last_hidden_state (see module docstring).
    Only the LAST transformer block's patch tokens are used here (n=1),
    matching embed_abdomen()'s use of `last_hidden_state`; pass
    DINOV3_LAYER_IDX to get_intermediate_layers instead if you want to
    mirror embed_acdc()'s shallower-layer choice.

    Returns (features_lowres, (pad_h, pad_w, H, W)) -- the caller is
    responsible for upsampling back to (H, W) via _finish_embed() AFTER
    reducing the channel count (see coembed()). Deliberately split this
    way: EMBED_DIM=384 * full-res (256x256) * D=192 upsampled all at once
    is ~19GB and OOMs even a 12GB GPU; channels x2-6x fewer after
    reduction makes it ~3-13GB, and channel-selection/PCA (both linear,
    per-voxel-independent ops) commute exactly with spatial interpolation,
    so doing the reduction first changes nothing about the result.
    """
    B, C, H, W, D = x.shape
    assert B == 1, "embed_oasis_lowres assumes batch size 1 (matches the rest of this pipeline)"

    pad_h = (target_size - H) // 2
    pad_h_after = target_size - H - pad_h
    pad_w = (target_size - W) // 2
    pad_w_after = target_size - W - pad_w

    x = F.pad(x, pad=(0, 0, pad_w, pad_w_after, pad_h, pad_h_after), mode='constant', value=0).cuda()
    x = torch.permute(x, (0, 4, 1, 2, 3)).squeeze(0)  # (D, C, target_size, target_size)
    x = x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    mask = (x != 0)
    x = transform_image(x) * mask

    res = []
    with torch.no_grad():
        for i in range(0, x.shape[0], dino_batch_size):
            batch = x[i:i + dino_batch_size]
            # reshape=True already strips the CLS + register/storage
            # tokens and returns (batch, EMBED_DIM, patch_h, patch_w).
            feature_map = backbone.get_intermediate_layers(batch, n=1, reshape=True)[0]
            res.append(feature_map)
    patch_features = torch.cat(res, dim=0)  # (D, EMBED_DIM, patch_h, patch_w)
    patch_features = torch.permute(patch_features, (1, 2, 3, 0)).unsqueeze(0)  # (1, EMBED_DIM, patch_h, patch_w, D)
    return patch_features, (pad_h, pad_w, H, W)


def _finish_embed(features_lowres, crop):
    """Upsample an (already channel-reduced) low-res feature volume back
    up to native (H, W) resolution and crop off the symmetric padding."""
    pad_h, pad_w, H, W = crop
    feat = F.interpolate(features_lowres, scale_factor=(8, 8, 1), mode='trilinear', align_corners=True)
    feat = feat[:, :, pad_h:pad_h + H, pad_w:pad_w + W, :]
    return feat


def _torch_pca_coembed(x, y, n_components=256):
    """Dimension-agnostic replacement for train_registration_all.py's
    apply_pca_to_3d_features_torch(), which hardcodes 768 channels and
    would silently misbehave with this 384-dim vits16 backbone."""
    w, h, d = x.shape[2:]
    embed_dim = x.shape[1]
    x_flat = x.contiguous().view(embed_dim, -1).T
    y_flat = y.contiguous().view(embed_dim, -1).T
    U, S, V = torch.svd_lowrank(x_flat, q=n_components)
    components = V[:, :n_components]
    x_reduced = torch.matmul(x_flat, components).T.contiguous().view(1, n_components, w, h, d)
    y_reduced = torch.matmul(y_flat, components).T.contiguous().view(1, n_components, w, h, d)
    return x_reduced, y_reduced


def coembed(x, y, mode='dino', train=False, n_channels=None):
    if n_channels is None:
        n_channels = min(256, EMBED_DIM)
    emb_x_lr, crop_x = embed_oasis_lowres(x)
    emb_y_lr, crop_y = embed_oasis_lowres(y)

    if mode == 'dino':
        if train:
            indices = random.sample(range(EMBED_DIM), n_channels)
            indices = torch.tensor(indices, dtype=torch.long, device=emb_y_lr.device)
            emb_x_lr = emb_x_lr[:, indices, :, :, :]
            emb_y_lr = emb_y_lr[:, indices, :, :, :]
        else:
            emb_x_lr, emb_y_lr = _torch_pca_coembed(emb_x_lr, emb_y_lr, n_components=n_channels)

    # The expensive (8x, 8x) spatial upsample now runs on n_channels
    # instead of EMBED_DIM channels (the fix from before) -- but the finished
    # (1, n_channels, H, W, D) feature volumes are themselves multi-GB at
    # OASIS's resolution (e.g. ~1.8GB each at n_channels=64), so process x
    # and y one at a time and `del` each intermediate the moment it's
    # consumed, instead of keeping emb_x/emb_y (and their pre-crop upsample
    # buffers) alive for the rest of the function. Without this, a single
    # coembed() call was observed to accumulate ~7-8GB of dead tensors and
    # OOM even after the channel-count fix above.
    emb_x = _finish_embed(emb_x_lr, crop_x)
    del emb_x_lr
    x = torch.cat([x, emb_x], dim=1)
    del emb_x

    emb_y = _finish_embed(emb_y_lr, crop_y)
    del emb_y_lr
    y = torch.cat([y, emb_y], dim=1)
    del emb_y

    return x, y


def run(opt):
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    train_loader = getters.getDataLoader(opt, split='train')
    val_loader = getters.getDataLoader(opt, split='val')

    model, init_epoch = getters.getTrainModelWithCheckpoints(opt, model_type='last')
    model_saver = getters.getModelSaver(opt)

    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)

    if opt['sim_type'] == 'NCC953':
        criterion_sim = [NccLoss(win=[9, 9, 9]), NccLoss(win=[5, 5, 5]), NccLoss(win=[3, 3, 3]),
                          NccLoss(win=[3, 3, 3]), NccLoss(win=[3, 3, 3])]
    elif opt['sim_type'] == 'mse':
        criterion_sim = [nn.MSELoss()] * 5
    criterion_reg = Grad3d()
    criterion_dsc = BinaryDiceLoss()

    ss = opt['img_size']
    upscale = opt['upscale']
    num_classes = opt['num_classes']

    best_dsc, best_epoch = 0, 0

    for epoch in range(init_epoch, opt['epochs']):
        time_epoch = time.time()
        loss_all, loss_sim_all, loss_reg_all, loss_dsc_all = (AverageMeter() for _ in range(4))

        transformers = nn.ModuleList([layers.SpatialTransformer(
            (ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2] // upscale[2] ** i)).cuda() for i in range(opt['layer_num'])])
        integrates = nn.ModuleList([layers.VecInt(
            (ss[0] // 2 ** i, ss[1] // 2 ** i, ss[2] // upscale[2] ** i), 7).cuda() for i in range(opt['layer_num'])])

        for idx, data in enumerate(train_loader):
            model.train()
            data = [Variable(t.cuda()) for t in data[:4]]
            x, x_seg = data[0].float(), data[1].long()
            y, y_seg = data[2].float(), data[3].long()

            x_seg_oh = F.one_hot(x_seg, num_classes=num_classes).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float()
            y_seg_oh = F.one_hot(y_seg, num_classes=num_classes).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float()

            x_emb, y_emb = coembed(x, y, mode='dino', train=True, n_channels=opt['dino_channels'])

            xs = get_downsampled_images(x, opt['layer_num'], scale=0.5, mode='trilinear')
            ys = get_downsampled_images(y, opt['layer_num'], scale=0.5, mode='trilinear')
            x_seg_ohs = get_downsampled_images(x_seg_oh, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=num_classes)
            y_seg_ohs = get_downsampled_images(y_seg_oh, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=num_classes)

            int_flows, pos_flows = model(x_emb, y_emb, transformers, integrates, upscale)

            reg_loss = sum(criterion_reg(int_flows[i]) / (2 ** i) for i in range(5)) * opt['reg_w']
            sim_loss = sum(criterion_sim[i](transformers[i](xs[i], pos_flows[i]), ys[i]) / (2 ** i)
                            for i in range(5)) * opt['sim_w']

            if opt['dsc_w'] == 0:
                dsc_loss = reg_loss * 0
            else:
                dsc_loss = sum(criterion_dsc(transformers[i](x_seg_ohs[i], pos_flows[i]), y_seg_ohs[i]) / (2 ** i)
                                for i in range(5)) * opt['dsc_w']

            loss = sim_loss + reg_loss + dsc_loss

            loss_all.update(loss.item(), y.numel())
            loss_sim_all.update(sim_loss.item(), y.numel())
            loss_reg_all.update(reg_loss.item(), y.numel())
            loss_dsc_all.update(dsc_loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}'.format(
                idx, len(train_loader), loss.item(), sim_loss.item(), reg_loss.item(), dsc_loss.item()),
                end='\r', flush=True)

        print('Epoch [{}/{}], Time {:.2f}, Loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}'.format(
            epoch, opt['epochs'], time.time() - time_epoch, loss_all.avg, loss_sim_all.avg, loss_reg_all.avg, loss_dsc_all.avg))

        eval_dsc, init_dsc = AverageMeter(), AverageMeter()
        reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()
        with torch.no_grad():
            for data in val_loader:
                model.eval()
                data = [Variable(t.cuda()) for t in data[:4]]
                x, x_seg = data[0].float(), data[1].long()
                y, y_seg = data[2].float(), data[3].long()

                dsc = dice_eval(x_seg.long(), y_seg.long(), num_classes)
                init_dsc.update(dsc.item(), x.size(0))

                x_emb, y_emb = coembed(x, y, mode='dino', train=False, n_channels=opt['dino_channels'])
                pos_flow = model(x_emb, y_emb, transformers, integrates, upscale, registration=True)
                def_out = reg_model(x_seg.float(), pos_flow)

                dsc = dice_eval(def_out.long(), y_seg.long(), num_classes)
                eval_dsc.update(dsc.item(), x.size(0))

        if eval_dsc.avg > best_dsc:
            best_dsc, best_epoch = eval_dsc.avg, epoch

        print('Epoch [{}/{}], Time {:.4f}, init DSC {:.4f}, eval DSC {:.4f}, best DSC {:.4f} at epoch {}'.format(
            epoch, opt['epochs'], time.time() - time_epoch, init_dsc.avg, eval_dsc.avg, best_dsc, best_epoch))

        model_saver.saveModel(model, epoch, eval_dsc.avg)


if __name__ == '__main__':

    opt = {
        'logs_path': './logs',
        'save_freq': 5,
        'n_checkpoints': 10,
        'power': 0.9,
    }

    parser = argparse.ArgumentParser(description="oasis")
    parser.add_argument("-m", "--model", type=str, default='regdino_mlp')
    parser.add_argument("-bs", "--batch_size", type=int, default=1)
    parser.add_argument("-d", "--dataset", type=str, default='oasisreg')
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("-dp", "--datasets_path", type=str, default=".")
    parser.add_argument("--epochs", type=int, default=301)
    parser.add_argument("--sim_w", type=float, default=1.)
    parser.add_argument("--reg_w", type=float, default=1)
    parser.add_argument("--dsc_w", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--layer_num", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=str, default='(160,224,192)')
    parser.add_argument("--upscale", type=str, default='(2,2,2)')
    parser.add_argument("--sim_type", type=str, default='NCC953')
    parser.add_argument("--num_classes", type=int, default=36)  # OASIS: labels 0-35
    # NEW: DINO feature channels kept after PCA/random-select, was hardcoded
    # to 256 (=257 total simple_encoder input channels). Measured peak GPU
    # memory for ONE coembed() call (both x and y) at full OASIS resolution
    # (160,224,192), embedding step only, does NOT include the registration
    # model's own forward/backward:
    #   256 channels: OOMs outright (tries to allocate ~13-19GB in one shot)
    #    64 channels: ~7.0 GB peak
    #    32 channels: ~3.6 GB peak
    #    16 channels: ~2.4 GB peak
    # Defaulted to 32 here since this is a shared GPU with limited free
    # memory -- the registration model's own forward/backward pass still
    # needs headroom on top of whatever coembed() uses. Raise it (up to 256,
    # matching train_registration_all.py's original) if you have a GPU with
    # a lot more free VRAM (24GB+ dedicated) and want closer fidelity to the
    # original design.
    parser.add_argument("--dino_channels", type=int, default=32)

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    opt['nkwargs'] = {s.split('=')[0]: s.split('=')[1] for s in unknowns if '=' in s}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['nkwargs']['dino_channels'] = str(opt['dino_channels'])
    opt['img_size'] = eval(opt['img_size'])
    opt['upscale'] = eval(opt['upscale'])
    opt['in_shape'] = opt['img_size']

    print('sim_w: %.4f, reg_w: %.4f, dsc_w: %.4f, img_size: %s, sim_type: %s, num_classes: %d, dino_channels: %d' % (
        opt['sim_w'], opt['reg_w'], opt['dsc_w'], opt['img_size'], opt['sim_type'], opt['num_classes'], opt['dino_channels']))

    run(opt)

'''
python train_registration_oasis.py -m regdino_mlp -d oasisreg -bs 1 --num_classes 36 \
    --gpu_id 0 --epochs 301 start_channel=32 --img_size '(160,224,192)' --upscale '(2,2,2)' \
    --datasets_path . --dino_channels 32
'''
