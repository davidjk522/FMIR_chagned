import time
import torch
import argparse
import numpy as np
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F

from utils.functions import AverageMeter, registerSTModel, adjust_learning_rate, get_downsampled_images_2D, dice_eval, get_downsampled_images_2D_acdc,get_downsampled_images
from utils.loss import Grad3d, BinaryDiceLoss, NccLoss
from utils import getters, setters
from torch.utils.data import ConcatDataset, DataLoader

import random
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
from transformers import AutoImageProcessor, AutoModel, Dinov2Config, AutoConfig
from einops import repeat, rearrange
from torchvision import transforms
transform_image = transforms.Compose([
    #transforms.Resize(image_size),
    #transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def constancy_rg_loss(Im, If, u):

    Imx, Imy, Imz = torch.gradient(Im, axis=(2,3,4))
    Ifx, Ify, Ifz = torch.gradient(If, axis=(2,3,4))
    Ix = (Imx + Ifx) / 2
    Iy = (Imy + Ify) / 2
    Iz = (Imz + Ifz) / 2
    It = (If - Im) / 2

    Id = Ix*u[:,0:1] + Iy*u[:,1:2] + Iz*u[:,2:3]
    ans = (Id+It)**2

    # grad_mag = torch.sqrt(Ix**2 + Iy**2 + Iz**2 + 1e-5)

    return ans.mean()


###########dino
#backbone = torch.hub.load('./weight/dinov3-vitb16-pretrain-lvd1689m','dinov3_vitb16', source='local', weights='model.safetensors')
pretrained_model_name = "./weight/dinov3-vits16-pretrain-lvd1689m"
'''
config = Dinov3Config.from_pretrained(pretrained_model_name)
config.image_size = 256  # 
config.patch_size = 16   #
'''
config = AutoConfig.from_pretrained(pretrained_model_name)
#print(config)
processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
backbone = AutoModel.from_pretrained(pretrained_model_name, device_map="auto").cuda()
intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11], 
        }
        
encoder_size = 'base'
backbone = backbone
#self.head = DPTHead(2, self.backbone.embed_dim, 128, False, out_channels=[96, 192, 384, 768]) #model.config.hidden_size
#head = DPTHead(2, self.backbone.config.hidden_size, 128, False, out_channels=[96, 192, 384, 768]) #model.config.hidden_size
for p in backbone.parameters():
    p.requires_grad = False
###########

def embed_acdc(x):
    B,C,H,W,D = x.shape
    x = torch.permute(x,(0,4,1,2,3)).view(-1,C,H,W)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    x = transform_image(x)
    patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
    #print(x.shape)
    outputs = backbone(x, output_hidden_states=True)
    intermediate_features = outputs.hidden_states 
    #print(len(intermediate_features))
    #patch_features = outputs.last_hidden_state[:, 5:, :]
    patch_features = intermediate_features[2][:, 5:, :]
    
    feature_map = patch_features.reshape(-1, patch_h, patch_w, patch_features.shape[-1])
    feature_map = feature_map.permute(0, 3, 1, 2)  # [batch, dim, H, W]
    
    feature_map = torch.permute(feature_map,(1,2,3,0)).unsqueeze(0)
    feature_map = F.interpolate(feature_map, scale_factor=(8, 8, 1), mode='trilinear', align_corners=True)
    return feature_map


def embed_abdomen(x):
    B,C,H,W,D = x.shape
    target_size = 128  
    pad_h = (target_size - H) // 2
    pad_h_after = (target_size - H) // 2
    pad_w = (target_size - W) // 2
    pad_w_after = (target_size - W) // 2
    
    x = F.pad(x,
        #pad=(0, 0, pad_w, pad_w_after, pad_h, pad_h_after),  # W, H, D
        pad=(0,0, pad_w, pad_w_after,pad_h, pad_h_after),
        mode='constant',
        value=0).cuda()
    #print(x.shape)#96, 80, 128
    x = torch.permute(x,(0,4,1,2,3)).squeeze(0)#view(-1,1,128,128)
    x = repeat(x, 'b c h w -> b (repeat c) h w', repeat=3)
    x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
    mask = (x!=0)
    x = transform_image(x)*mask
    patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
    #print(x.shape)
    batch_size = 32
    res = []
    for i in range(0, x.shape[0], batch_size): 
        batch = x[i:i+batch_size]  # [B, 3, H, W]
        out = backbone(batch, output_hidden_states=True) #sam.image_encoder(batch)
        feature_map = out.last_hidden_state[:, 5:, :]
        feature_map = feature_map.view(-1, patch_h, patch_w, feature_map.shape[-1])  # 
        res.append(feature_map)
    patch_features = torch.cat(res, dim=0)
    
    patch_features = patch_features.permute(0, 3, 1, 2)  # [batch, dim, H, W]
    
    patch_features = torch.permute(patch_features,(1,2,3,0)).unsqueeze(0)
    
    patch_features = F.interpolate(patch_features,scale_factor=(8, 8,1),  mode='trilinear',align_corners=True)
    #feature_map = feature_map[:, :, pad_h//2: pad_h//2 + H//2, pad_w//2: pad_w//2 + W//2, :]
    patch_features = patch_features[:, :, pad_h: pad_h + H, pad_w: pad_w + W, :]
    #print(feature_map.shape)
    return patch_features



import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib
import os

def apply_pca_to_3d_features(x, y, n_components=36, model_path="pca_model36.pkl", standardize=False, random_state=42):
    """
    Apply PCA dimensionality reduction to 3D DINOv3 features for both x and y using the same transformation
    
    Args:
        x: torch.Tensor of shape (1, 768, w, h, d) - DINOv3 features
        y: torch.Tensor of shape (1, 768, w, h, d) - another feature tensor
        n_components: target dimension, default 256
        model_path: path to saved model file, default "pca_model.pkl"
        standardize: whether to standardize data, default True
        random_state: random seed for PCA, default 42
    
    Returns:
        x_reduced: reduced x features, shape (1, n_components, w, h, d)
        y_reduced: reduced y features, shape (1, n_components, w, h, d)
    """
    # Check if model exists
    if os.path.exists(model_path):
        # Load existing models
        models = joblib.load(model_path)
        pca_model = models['pca_model']
        scaler = models['scaler']
        #print(f"Loaded PCA model from {model_path}")
        fit_new_models = False
    else:
        # Create new models
        pca_model = None
        scaler = None
        fit_new_models = True
        #print("Fitting new PCA model")
    
    # Input validation
    assert x.dim() == 5 and x.shape[1] == 768, f"x should have shape (1, 768, w, h, d), got {x.shape}"
    assert y.dim() == 5 and y.shape[1] == 768, f"y should have shape (1, 768, w, h, d), got {y.shape}"
    assert x.shape[2:] == y.shape[2:], "x and y should have same spatial dimensions"
    
    # Save original shape
    w, h, d = x.shape[2:]
    num_voxels = w * h * d
    
    # Reshape tensors to (num_voxels, 768) using contiguous reshape
    x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, 768)  # shape: (num_voxels, 768)
    y_flat = y.permute(0, 2, 3, 4, 1).reshape(-1, 768)  # shape: (num_voxels, 768)
    
    # Convert to numpy for PCA processing
    x_np = x_flat.detach().cpu().numpy()
    y_np = y_flat.detach().cpu().numpy()
    
    # Standardization
    if standardize:
        if fit_new_models:
            scaler = StandardScaler()
            x_processed = scaler.fit_transform(x_np)
        else:
            x_processed = scaler.transform(x_np)
        y_processed = scaler.transform(y_np)
    else:
        x_processed = x_np
        y_processed = y_np
        scaler = None
    
    # PCA transformation
    if fit_new_models:
        pca_model = PCA(n_components=n_components, random_state=random_state)
        x_reduced_np = pca_model.fit_transform(x_processed)
        explained_variance = np.sum(pca_model.explained_variance_ratio_)
        #print(f"Fitted new PCA model: 768 -> {n_components}, explained variance: {explained_variance:.4f}")
        
        # Save the new models
        models = {
            'pca_model': pca_model,
            'scaler': scaler
        }
        joblib.dump(models, model_path)
        #print(f"Saved PCA model to {model_path}")
    else:
        x_reduced_np = pca_model.transform(x_processed)
        #print(f"Using existing PCA model: 768 -> {n_components}")
    
    y_reduced_np = pca_model.transform(y_processed)
    
    # Convert back to PyTorch tensors and reshape to original spatial dimensions
    x_reduced_flat = torch.from_numpy(x_reduced_np)  # shape: (num_voxels, n_components)
    y_reduced_flat = torch.from_numpy(y_reduced_np)  # shape: (num_voxels, n_components)
    
    # Reshape back to (1, n_components, w, h, d)
    x_reduced = x_reduced_flat.reshape(1, w, h, d, n_components).permute(0, 4, 1, 2, 3)
    y_reduced = y_reduced_flat.reshape(1, w, h, d, n_components).permute(0, 4, 1, 2, 3)
    
    # Ensure same dtype and device as input
    x_reduced = x_reduced.to(dtype=x.dtype, device=x.device)
    y_reduced = y_reduced.to(dtype=y.dtype, device=y.device)
    
    #print(f"Input shapes: x{x.shape}, y{y.shape}")
    #print(f"Output shapes: x{x_reduced.shape}, y{y_reduced.shape}")
    
    return x_reduced, y_reduced


def torch_pca(x, n_components=256):
    """
    PCA implementation using PyTorch (avoids CPU-GPU transfer)
    """
    # Center the data
    x_centered = x - x.mean(dim=0)
    
    # SVD
    U, S, V = torch.svd_lowrank(x_centered, q=n_components)
    
    # Project to lower dimension
    x_reduced = torch.matmul(x_centered, V)
    
    return x_reduced, V

def apply_pca_to_3d_features_torch(x, y, n_components=256, model_path="pca_model.pkl", standardize=True):
    """
    Ultra-fast PCA using PyTorch (no CPU-GPU transfer)
    """
    if os.path.exists(model_path):
        models = torch.load(model_path)
        pca_components = models['pca_components']
        mean = models['mean']
        scaler_mean = models.get('scaler_mean', None)
        scaler_std = models.get('scaler_std', None)
        #print(f"Loaded PCA model from {model_path}")
        fit_new_models = False
    else:
        pca_components = None
        mean = None
        scaler_mean = None
        scaler_std = None
        fit_new_models = True
        #print("Fitting new PCA model")
    
    # Original shape
    w, h, d = x.shape[2:]
    device = x.device
    
    # Reshape
    x_flat = x.contiguous().view(768, -1).T  # (num_voxels, 768)
    y_flat = y.contiguous().view(768, -1).T  # (num_voxels, 768)
    '''
    # Standardization
    if standardize:
        if fit_new_models:
            scaler_mean = x_flat.mean(dim=0)
            scaler_std = x_flat.std(dim=0)
            scaler_std[scaler_std == 0] = 1.0  # avoid division by zero
            x_processed = (x_flat - scaler_mean) / scaler_std
        else:
            x_processed = (x_flat - scaler_mean) / scaler_std
        y_processed = (y_flat - scaler_mean) / scaler_std
    else:
        x_processed = x_flat
        y_processed = y_flat
    '''
    x_processed = x_flat
    y_processed = y_flat
    # PCA
    if fit_new_models:
        # Center data
        #mean = x_processed.mean(dim=0)
        #x_centered = x_processed - mean
        
        # SVD
        U, S, V = torch.svd_lowrank(x_processed, q=n_components)
        pca_components = V[:, :n_components]
        
        # Project
        x_reduced = torch.matmul(x_processed, pca_components)
        
        # Save model
        models = {
            'pca_components': pca_components,
            'mean': mean,
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std
        }
        torch.save(models, model_path)
        #print(f"Saved PCA model to {model_path}")
    else:
        #x_centered = x_processed - mean
        x_reduced = torch.matmul(x_processed, pca_components)
    
    #y_centered = y_processed - mean
    y_reduced = torch.matmul(y_processed, pca_components)
    
    # Reshape back
    x_reduced = x_reduced.T.contiguous().view(1, n_components, w, h, d)
    y_reduced = y_reduced.T.contiguous().view(1, n_components, w, h, d)
    
    #print(f"Input shapes: x{x.shape}, y{y.shape}")
    #print(f"Output shapes: x{x_reduced.shape}, y{y_reduced.shape}")
    
    return x_reduced, y_reduced

def coembed(x, y,flag):
    if flag == 'acdcreg':
        emb_x = embed_acdc(x)
        emb_y = embed_acdc(y)
    else:
        emb_x = embed_abdomen(x)
        emb_y = embed_abdomen(y)
    emb_x, emb_y = apply_pca_to_3d_features_torch(emb_x, emb_y, n_components=256) #apply_pca_to_3d_features_torch  apply_pca_to_3d_features
    x = torch.cat([x,emb_x], dim=1)
    y = torch.cat([y,emb_y], dim=1)
    return x,y

def coembed(x, y,flag='acdcreg',mode='dino', train = False):
    print(mode)
    if flag == 'acdcreg':
        emb_x = embed_acdc(x)
        emb_y = embed_acdc(y)
    else:
        emb_x = embed_abdomen(x)
        emb_y = embed_abdomen(y)
    
    if mode == 'dino':
        if train == True:
            indices = random.sample(range(768), 256)
            indices = torch.tensor(indices, dtype=torch.long, device=emb_y.device)
            #emb_x = emb_x.index_select(1, indices)
            #emb_y = emb_y.index_select(1, indices)
            emb_x = emb_x[:, indices, :, :, :].cuda()
            emb_y = emb_y[:, indices, :, :, :].cuda()
        else:
            emb_x, emb_y = apply_pca_to_3d_features_torch(emb_x, emb_y, n_components=256) #apply_pca_to_3d_features_torch
    x = torch.cat([x,emb_x], dim=1)
    y = torch.cat([y,emb_y], dim=1)
    return x,y


def run(opt):
    # Setting up
    setters.setSeed(0)
    setters.setFoldersLoggers(opt)
    setters.setGPU(opt)

    # Getting model-related components
    
    acdc_train_loader = getters.getDataLoader(opt, split='train')
    acdc_val_loader = getters.getDataLoader(opt, split='val')
    opt['dataset'] = 'abdomenreg'
    #opt['root_dir'] = './../../../data/abdomenreg/'
    setters.setFoldersLoggers(opt)
    abdomen_train_loader = getters.getDataLoader(opt, split='train')
    abdomen_val_loader = getters.getDataLoader(opt, split='val')
    
    acdc_dataset = acdc_train_loader.dataset
    abdomen_dataset = abdomen_train_loader.dataset
    print(len(acdc_dataset),len(abdomen_dataset),opt['dataset'])
    combined_dataset = ConcatDataset([acdc_dataset, abdomen_dataset])
    '''
    opt['dataset'] = 'acdcreg'
    setters.setFoldersLoggers(opt)
    '''
    
    combined_train_loader = DataLoader(
    combined_dataset,
    batch_size=acdc_train_loader.batch_size,
    shuffle=True,
    num_workers=acdc_train_loader.num_workers
)
    
    acdc_dataset = acdc_val_loader.dataset
    abdomen_dataset = abdomen_val_loader.dataset
    combined_dataset = ConcatDataset([acdc_dataset, abdomen_dataset])
    
    combined_val_loader = DataLoader(
    combined_dataset,
    batch_size=acdc_val_loader.batch_size,
    shuffle=True,
    num_workers=acdc_val_loader.num_workers
)
    ####

    combined_train_loader = abdomen_train_loader
    combined_val_loader = abdomen_val_loader
    '''
    combined_train_loader = acdc_train_loader
    combined_val_loader = acdc_val_loader
    '''
    
    #####
    
    model, init_epoch = getters.getTrainModelWithCheckpoints(opt,model_type='last')
    model_saver = getters.getModelSaver(opt)

    
    transformer_seg = layers.SpatialTransformer(opt['img_size'],'nearest').cuda()
    transformer_img = layers.SpatialTransformer(opt['img_size'],'trilinear').cuda()
    optimizer = optim.Adam(model.parameters(), lr=opt['lr'], weight_decay=0, amsgrad=True)
    
    if opt['sim_type'] == 'NCC953':
        criterion_sim_0 = NccLoss(win=[9,9,9])
        criterion_sim_1 = NccLoss(win=[5,5,5])
        criterion_sim_2 = NccLoss(win=[3,3,3])
    elif opt['sim_type'] == 'mse':
        criterion_sim_0 = nn.MSELoss()
        criterion_sim_1 = nn.MSELoss()
        criterion_sim_2 = nn.MSELoss()
    criterion_reg = Grad3d()
    criterion_mse = nn.MSELoss()
    criterion_dsc = BinaryDiceLoss() # binary dice loss does not require class labels, as all replaced by one-hot encoding already
    best_dsc = 0
    best_epoch = 0

    for epoch in range(init_epoch, opt['epochs']):
        '''
        Training
        '''
        time_epoch = time.time()
        loss_all = AverageMeter()
        loss_sim_all = AverageMeter()
        loss_reg_all = AverageMeter()
        loss_dsc_all = AverageMeter()
        loss_hsr_all = AverageMeter()
        gradient_norms = []
        for idx, data in enumerate(combined_train_loader):  
            model.train()
            ####sp
            for m in model.modules():
                if hasattr(m, 'reset'):
                    m.reset()
            #####
            # adjust_learning_rate(optimizer, epoch, opt['epochs'], opt['lr'], opt['power'])
            data = [Variable(t.cuda()) for t in data[:4]]
            x, x_seg = data[0].float(), data[1].long()
            y, y_seg = data[2].float(), data[3].long()
            
            
            if x.shape[-1]==16:
                opt['num_classes'] = 4
                opt['upscale'] = (2,2,1)
                opt['img_size'] = (128, 128, 16)
                opt['dataset'] = 'acdcreg'
            else:
                opt['num_classes'] = 14
                opt['upscale'] = (2,2,2)
                opt['img_size'] = (192//2,160//2,256//2)
                opt['dataset'] = 'abdomenreg'
            #print(x.shape,x_seg.shape)
            x_seg_oh = F.one_hot(x_seg,num_classes=opt['num_classes']).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float()
            y_seg_oh = F.one_hot(y_seg,num_classes=opt['num_classes']).squeeze(1).permute(0, 4, 1, 2, 3).contiguous().float()
            upscale = opt['upscale']

            if ('encoderOnly' in opt['model']) or ('SAMIR' in opt['model']) or ('FMIR' in opt['model']) or ('regdino' in opt['model']):
                
                x_emb, y_emb = coembed(x,y,opt['dataset'], mode='dino', train = True)
                if opt['dataset'] == 'acdcreg':
                    xs = get_downsampled_images_2D_acdc(x, opt['layer_num'], scale=(1/upscale[0],1/upscale[1],1/upscale[2]), mode='trilinear')
                    ys = get_downsampled_images_2D_acdc(y, opt['layer_num'], scale=(1/upscale[0],1/upscale[1],1/upscale[2]), mode='trilinear')

                    x_seg_ohs = get_downsampled_images_2D_acdc(x_seg_oh, opt['layer_num'], scale=(1/upscale[0],1/upscale[1],1/upscale[2]), mode='trilinear', n_cs=opt['num_classes'])
                    y_seg_ohs = get_downsampled_images_2D_acdc(y_seg_oh, opt['layer_num'], scale=(1/upscale[0],1/upscale[1],1/upscale[2]), mode='trilinear', n_cs=opt['num_classes'])
                    
                    fea_xs = get_downsampled_images_2D_acdc(x_emb, opt['layer_num'], scale=(1/upscale[0],1/upscale[1],1/upscale[2]), mode='trilinear', n_cs=x_emb.shape[1])
                    fea_ys = get_downsampled_images_2D_acdc(y_emb, opt['layer_num'], scale=(1/upscale[0],1/upscale[1],1/upscale[2]), mode='trilinear', n_cs=x_emb.shape[1])
                else:
                    #print(x.shape)
                    xs = get_downsampled_images(x, opt['layer_num'], scale=0.5, mode='trilinear')
                    ys = get_downsampled_images(y, opt['layer_num'], scale=0.5, mode='trilinear')

                    x_seg_ohs = get_downsampled_images(x_seg_oh, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=opt['num_classes'])
                    y_seg_ohs = get_downsampled_images(y_seg_oh, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=opt['num_classes'])
                    fea_xs = get_downsampled_images(x_emb, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=x_emb.shape[1])
                    fea_ys = get_downsampled_images(y_emb, opt['layer_num'], scale=0.5, mode='trilinear', n_cs=x_emb.shape[1])
                    
                    
                ss = opt['img_size']
                #print(ss)
                transformers = nn.ModuleList([layers.SpatialTransformer((ss[0]//2**i,ss[1]//2**i,ss[2]//upscale[2]**i)).cuda() for i in range(opt['layer_num'])])
                integrates = nn.ModuleList([layers.VecInt((ss[0]//2**i,ss[1]//2**i,ss[2]//upscale[2]**i), 7).cuda() for i in range(opt['layer_num'])])

                
                int_flows, pos_flows = model(x_emb, y_emb, transformers, integrates, upscale)

                reg_loss_0 = criterion_reg(int_flows[0])
                reg_loss_1 = criterion_reg(int_flows[1]) / 2
                reg_loss_2 = criterion_reg(int_flows[2]) / 4
                reg_loss_3 = criterion_reg(int_flows[3]) / 8
                reg_loss_4 = criterion_reg(int_flows[4]) / 16
                reg_loss = (reg_loss_0 + reg_loss_1 + reg_loss_2 + reg_loss_3 + reg_loss_4) * opt['reg_w']
                loss_reg_all.update(reg_loss.item(), y.numel())
                #print(xs[1].shape,pos_flows[1].shape)

                sim_loss_0 = criterion_sim_0(transformers[0](xs[0], pos_flows[0]), ys[0])
                sim_loss_1 = criterion_sim_1(transformers[1](xs[1], pos_flows[1]), ys[1]) / 2
                sim_loss_2 = criterion_sim_2(transformers[2](xs[2], pos_flows[2]), ys[2]) / 4
                sim_loss_3 = criterion_sim_2(transformers[3](xs[3], pos_flows[3]), ys[3]) / 8
                sim_loss_4 = criterion_sim_2(transformers[4](xs[4], pos_flows[4]), ys[4]) / 16
                sim_loss = (sim_loss_0 + sim_loss_1 + sim_loss_2 + sim_loss_3 + sim_loss_4) * opt['sim_w']
                loss_sim_all.update(sim_loss.item(), y.numel())

                if opt['dsc_w'] == 0:
                    dsc_loss = reg_loss * 0
                else:
                    dsc_loss_0 = criterion_dsc(transformers[0](x_seg_ohs[0], pos_flows[0]), y_seg_ohs[0])
                    dsc_loss_1 = criterion_dsc(transformers[1](x_seg_ohs[1], pos_flows[1]), y_seg_ohs[1]) / 2
                    dsc_loss_2 = criterion_dsc(transformers[2](x_seg_ohs[2], pos_flows[2]), y_seg_ohs[2]) / 4
                    dsc_loss_3 = criterion_dsc(transformers[3](x_seg_ohs[3], pos_flows[3]), y_seg_ohs[3]) / 8
                    dsc_loss_4 = criterion_dsc(transformers[4](x_seg_ohs[4], pos_flows[4]), y_seg_ohs[4]) / 16
                    dsc_loss = (dsc_loss_0 + dsc_loss_1 + dsc_loss_2 + dsc_loss_3 + dsc_loss_4) * opt['dsc_w']
                loss_dsc_all.update(dsc_loss.item(), y.numel())
            else:
                int_flows, pos_flows = model(x, y)
                #print(int_flows.shape)
                #input()
                reg_loss = criterion_reg(int_flows)

                X_Y = transformer_img(x, pos_flows)
                warped_x_seg= transformer_img(x_seg_oh.cuda(), pos_flows)
                
                sim_loss = criterion_sim_0(y, X_Y)
                dsc_loss = criterion_dsc(warped_x_seg, y_seg_oh.cuda())

                loss_reg_all.update(reg_loss.item(), y.numel())
                loss_sim_all.update(sim_loss.item(), y.numel())
                loss_dsc_all.update(dsc_loss.item(), y.numel())
            
            if opt['hsr_w'] == 0:
                hsr_loss = reg_loss * 0
            else:
                '''
                hsr_loss_0 = constancy_rg_loss(fea_xs[0], fea_ys[0], pos_flows[0])
                hsr_loss_1 = constancy_rg_loss(fea_xs[1], fea_ys[1], pos_flows[1]) / 2
                hsr_loss_2 = constancy_rg_loss(fea_xs[2], fea_ys[2], pos_flows[2]) / 4
                hsr_loss_3 = constancy_rg_loss(fea_xs[3], fea_ys[3], pos_flows[3]) / 8
                hsr_loss_4 = constancy_rg_loss(fea_xs[4], fea_ys[4], pos_flows[4]) / 16
                '''
                hsr_loss_0 = criterion_mse(transformers[0](fea_xs[0], pos_flows[0]), fea_ys[0])
                hsr_loss_1 = criterion_mse(transformers[1](fea_xs[1], pos_flows[1]), fea_ys[1]) / 2
                hsr_loss_2 = criterion_mse(transformers[2](fea_xs[2], pos_flows[2]), fea_ys[2]) / 4
                hsr_loss_3 = criterion_mse(transformers[3](fea_xs[3], pos_flows[3]), fea_ys[3]) / 8
                hsr_loss_4 = criterion_mse(transformers[4](fea_xs[4], pos_flows[4]), fea_ys[4]) / 16
                hsr_loss = (hsr_loss_0 + hsr_loss_1 + hsr_loss_2 + hsr_loss_3 + hsr_loss_4) * opt['hsr_w']
            loss_hsr_all.update(hsr_loss.item(), y.numel())

            loss = sim_loss * opt['sim_w'] + reg_loss * opt['reg_w'] + dsc_loss * opt['dsc_w']+ hsr_loss * opt['hsr_w']


            loss_all.update(loss.item(), y.numel())
            optimizer.zero_grad()
            loss.backward()
            
            #######
            # 
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            gradient_norms.append(total_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #######
            
            optimizer.step()
            print('Iter {} of {} loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}, HSR {:.4f}'.format(idx, len(combined_train_loader), loss.item(), sim_loss.item(), reg_loss.item(), dsc_loss.item(),hsr_loss.item()), end='\r', flush=True)

        print('Epoch [{}/{}], Time {:.2f}, Loss {:.4f}, Sim {:.4f}, Reg {:.4f}, DSC {:.4f}, HSR {:.4f}'.format(epoch, opt['epochs'], time.time()-time_epoch, loss_all.avg, loss_sim_all.avg, loss_reg_all.avg, loss_dsc_all.avg, loss_hsr_all.avg))
        '''
        Validation
        '''
        eval_dsc = AverageMeter()
        init_dsc = AverageMeter()
        with torch.no_grad():
            #for data in val_loader:
            for data in (combined_val_loader):
                model.eval()
                data = [Variable(t.cuda())  for t in data[:4]]
                x, x_seg = data[0].float(), data[1].long()
                y, y_seg = data[2].float(), data[3].long()
                
                if x.shape[-1]==16:
                    opt['num_classes'] = 4
                    opt['upscale'] = (2,2,1)
                    opt['img_size'] = (128, 128, 16)
                    opt['dataset'] = 'acdcreg'
                else:
                    opt['num_classes'] = 14
                    opt['upscale'] = (2,2,2)
                    opt['img_size'] = (192//2,160//2,256//2)
                    opt['dataset'] = 'abdomenreg'

                
                ss = opt['img_size']
                upscale = opt['upscale']
                #print(ss,upscale)
                transformers = nn.ModuleList([layers.SpatialTransformer((ss[0]//2**i,ss[1]//2**i,ss[2]//upscale[2]**i)).cuda() for i in range(opt['layer_num'])])
                integrates = nn.ModuleList([layers.VecInt((ss[0]//2**i,ss[1]//2**i,ss[2]//upscale[2]**i), 7).cuda() for i in range(opt['layer_num'])])
                reg_model = registerSTModel(opt['img_size'], 'nearest').cuda()

                
                dsc = dice_eval(x_seg.long(), y_seg.long(), opt['num_classes'])
                init_dsc.update(dsc.item(), x.size(0))

                # x -> y
                
                

                x_emb, y_emb = coembed(x,y,opt['dataset'], mode='dino', train = False)
                pos_flow = model(x_emb, y_emb, transformers, integrates,upscale,registration=True)
                def_out = reg_model(x_seg.float(), pos_flow)
                
                dsc = dice_eval(def_out.long(), y_seg.long(), opt['num_classes'])
                eval_dsc.update(dsc.item(), x.size(0))

                if opt['is_bidir'] == 1:
                    # y -> x
                    pos_flow = model(y_emb, x_emb, transformers, integrates,upscale,registration=True)
                    def_out = reg_model(y_seg.cuda().float(), pos_flow)
                    dsc = dice_eval(def_out.long(), x_seg.long(), opt['num_classes'])
                    eval_dsc.update(dsc.item(), x.size(0))

        if eval_dsc.avg > best_dsc:
            best_dsc = eval_dsc.avg
            best_epoch = epoch

        print('Epoch [{}/{}], Time {:.4f}, init DSC {:.4f}, eval DSC {:.4f}, best DSC {:.4f} at epoch {}'.format(epoch, opt['epochs'], time.time()-time_epoch, init_dsc.avg, eval_dsc.avg, best_dsc, best_epoch))

        model_saver.saveModel(model, epoch, eval_dsc.avg)

if __name__ == '__main__':

    opt = {
        #'img_size': (128, 128, 16),  # input image size
        #'in_shape': (128, 128, 16),  # input image size
        'logs_path': './logs',       # path to save logs
        'save_freq': 5,              # save model every save_freq epochs
        'n_checkpoints': 10,          # number of checkpoints to keep
        'power': 0.9,                # decay power
    }

    parser = argparse.ArgumentParser(description = "cardiac")
    parser.add_argument("-m", "--model", type = str, default = 'someWarpComplex')
    parser.add_argument("-bs", "--batch_size", type = int, default = 1)
    parser.add_argument("-d", "--dataset", type = str, default = 'acdcreg')
    parser.add_argument("--gpu_id", type = str, default = '0')
    parser.add_argument("-dp", "--datasets_path", type = str, default = "/home/admin123/data/")
    parser.add_argument("--epochs", type = int, default = 301)
    parser.add_argument("--sim_w", type = float, default = 1.)
    parser.add_argument("--reg_w", type = float, default = 1)#0.01
    parser.add_argument("--dsc_w", type = float, default = 0)#0.1
    parser.add_argument("--hsr_w", type = float, default = 0.)
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--layer_num", type = int, default = 5)
    parser.add_argument("--num_workers", type = int, default = 4) # best, last or epoch
    parser.add_argument("--img_size", type = str, default = '(128, 128, 16)')
    parser.add_argument("--upscale", type = str, default = '(2,2,2)')
    parser.add_argument("--is_int", type = int, default = 0)
    # parser.add_argument("--largePool", type = int, default = 0)
    parser.add_argument("--sim_type", type = str, default = 'NCC953') # mse or ncc999, ncc995, ncc993 or others
    parser.add_argument("--num_classes", type = int, default = 4) #  number of anatomical classes, 4 for cardiac, 14 for abdominal
    parser.add_argument("--is_bidir", type = int, default = 0) #  only intra-subject cardiac needs bidirectional

    args, unknowns = parser.parse_known_args()
    opt = {**opt, **vars(args)}
    print(unknowns)
    opt['nkwargs'] = {s.split('=')[0]:s.split('=')[1] for s in unknowns}
    opt['nkwargs']['img_size'] = opt['img_size']
    opt['img_size'] = eval(opt['img_size'])
    opt['upscale'] = eval(opt['upscale'])
    opt['in_shape'] = opt['img_size']

    print('sim_w: %.4f, reg_w: %.4f, dsc_w: %.4f, img_size: %s, sim_type: %s, num_classes: %d' % (opt['sim_w'], opt['reg_w'], opt['dsc_w'], opt['img_size'], opt['sim_type'], opt['num_classes']))

    run(opt)

'''
python train_registration_ACDC.py -m encoderOnlyACDCComplex -d acdcreg -bs 1 --num_classes 4 start_channel=32 --gpu_id 5
CUDA_VISIBLE_DEVICES=5 python train_registration_all.py -m regdino_mlp -d acdcreg -bs 1 --num_classes 4 --gpu_id 4 --epochs 301  start_channel=32
CUDA_VISIBLE_DEVICES=5 python train_registration_ACDC.py -m SAMIR -d acdcreg -bs 1 --num_classes 4 --gpu_id 5 --epochs 301 start_channel=32 
python train_registration_ACDC.py -m SP_EOIR_ACDC -d acdcreg -bs 1 --num_classes 4 start_channel=32 --gpu_id 1
CUDA_VISIBLE_DEVICES=1 python train_registration_all.py -m regdino_mlp -d abdomenreg -bs 1 --num_classes 14 --gpu_id 0 --epochs 301  start_channel=32 --img_size '(192//2,160//2,256//2)'
CUDA_VISIBLE_DEVICES=5 python train_registration_all257_1022.py -m regdino_mlp -d acdcreg -bs 1 --num_classes 4 --gpu_id 0 --epochs 301  start_channel=32 --img_size '(128,128,16)' --upscale '(2,2,1)'
'''
