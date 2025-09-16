import os
import json
import random
import math
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from timm import create_model

# === Модель ===
class PuzzleRelationModel(nn.Module):
    def __init__(self,
                 backbone_name='swin_tiny_patch4_window7_224',
                 pretrained=False,
                 feature_dim=768,
                 shape_dim=32):
        super().__init__()
        self.backbone = create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=4,
            num_classes=0
        )
        concat_dim = feature_dim * 4 + shape_dim * 2

        self.adj_head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.dir_head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        self.rot_head = nn.Sequential(
            nn.Linear(concat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, frag_a, frag_b, shape_a, shape_b):
        fa = self.backbone(frag_a)
        fb = self.backbone(frag_b)
        feats_img = torch.cat([
            fa, fb, torch.abs(fa - fb), fa * fb
        ], dim=1)
        feats = torch.cat([feats_img, shape_a, shape_b], dim=1)

        logit_adj = self.adj_head(feats).squeeze(1)
        logits_dir = self.dir_head(feats)
        rot = self.rot_head(feats)
        return logit_adj, logits_dir, rot

# === Маскированный multi-task loss ===
class PuzzleLoss(nn.Module):
    def __init__(self, lambda_adj=1.0, lambda_dir=1.0, lambda_rot=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.lambda_adj = lambda_adj
        self.lambda_dir = lambda_dir
        self.lambda_rot = lambda_rot

    def forward(self, logit_adj, logits_dir, pred_rot,
                true_adj, true_dir, true_rot):
        loss_adj = self.bce(logit_adj, true_adj.float())
        mask = true_adj.bool()
        if mask.any():
            loss_dir = self.ce(logits_dir[mask], true_dir[mask])
            loss_rot = self.mse(pred_rot[mask], true_rot[mask])
        else:
            device = logit_adj.device
            loss_dir = torch.tensor(0.0, device=device)
            loss_rot = torch.tensor(0.0, device=device)
        loss = (self.lambda_adj * loss_adj +
                self.lambda_dir * loss_dir +
                self.lambda_rot * loss_rot)
        return loss, {'adj': loss_adj.item(), 'dir': loss_dir.item(), 'rot': loss_rot.item()}

# === Dataset ===
class PuzzlePairDataset(Dataset):
    def __init__(self, image_pairs, root_dir='', transform=None):
        self.pairs = image_pairs
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        path_a = os.path.join(self.root_dir, p['frag_a'])
        path_b = os.path.join(self.root_dir, p['frag_b'])
        # RGB+Mask
        a_rgba = Image.open(path_a).convert('RGBA')
        b_rgba = Image.open(path_b).convert('RGBA')
        rgb_a = a_rgba.convert('RGB'); mask_a = a_rgba.split()[-1]
        rgb_b = b_rgba.convert('RGB'); mask_b = b_rgba.split()[-1]
        rgb_a_t = self.transform(rgb_a); rgb_b_t = self.transform(rgb_b)
        mask_a_t = self.transform(mask_a); mask_b_t = self.transform(mask_b)
        img_a = torch.cat([rgb_a_t, mask_a_t], dim=0)
        img_b = torch.cat([rgb_b_t, mask_b_t], dim=0)
        fd_a = torch.tensor(p['fd_a'], dtype=torch.float32)
        fd_b = torch.tensor(p['fd_b'], dtype=torch.float32)
        true_adj = torch.tensor(p['is_adjacent'], dtype=torch.float32)
        true_dir = torch.tensor(p['direction'], dtype=torch.long)
        rad = math.radians(p['angle_diff'])
        true_rot = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        return img_a, img_b, fd_a, fd_b, true_adj, true_dir, true_rot

# === Train/Val функции ===
def train_epoch(model, loss_fn, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    metrics = {'adj':0,'dir':0,'rot':0}
    for a,b,fd_a,fd_b,adj,dir_,rot in tqdm(loader, desc='Train'):
        a,b,fd_a,fd_b,adj,dir_,rot = [x.to(device) for x in (a,b,fd_a,fd_b,adj,dir_,rot)]
        optimizer.zero_grad()
        la, ld, lr = model(a, b, fd_a, fd_b)
        loss, parts = loss_fn(la, ld, lr, adj, dir_, rot)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        for k in metrics: metrics[k] += parts[k]
    n = len(loader)
    return total_loss/n, {k: metrics[k]/n for k in metrics}


def validate(model, loss_fn, loader, device):
    model.eval()
    total_loss = 0.0
    metrics = {'adj':0,'dir':0,'rot':0}
    with torch.no_grad():
        for a,b,fd_a,fd_b,adj,dir_,rot in tqdm(loader, desc='Val'):
            a,b,fd_a,fd_b,adj,dir_,rot = [x.to(device) for x in (a,b,fd_a,fd_b,adj,dir_,rot)]
            la, ld, lr = model(a, b, fd_a, fd_b)
            loss, parts = loss_fn(la, ld, lr, adj, dir_, rot)
            total_loss += loss.item()
            for k in metrics: metrics[k] += parts[k]
    n = len(loader)
    return total_loss/n, {k: metrics[k]/n for k in metrics}

# === Main ===
if __name__ == '__main__':
    from torch.optim import AdamW
    batch_size = 32
    epochs     = 20
    lr_backbone= 5e-5
    lr_heads   = 1e-4

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def load_jsonl(fp):
        lst = []
        with open(fp,'r') as f:
            for ln in f:
                if ln.strip(): lst.append(json.loads(ln))
        return lst

    train_pairs = load_jsonl('train_fragments_final/train_pairs.jsonl')
    val_pairs   = load_jsonl('train_fragments_final2/val_pairs.jsonl')
    # root directories
    train_ds = PuzzlePairDataset(train_pairs, root_dir='train_fragments_final')
    val_ds   = PuzzlePairDataset(val_pairs,   root_dir='train_fragments_final2')
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4)

    shape_dim = len(train_pairs[0]['fd_a'])
    model = PuzzleRelationModel(shape_dim=shape_dim).to(device)

    optimizer = AdamW([
        {'params': model.backbone.parameters(), 'lr': lr_backbone},
        {'params': list(model.adj_head.parameters())
                    + list(model.dir_head.parameters())
                    + list(model.rot_head.parameters()),
         'lr': lr_heads}
    ], weight_decay=1e-2)
    loss_fn = PuzzleLoss(lambda_adj=1.0, lambda_dir=1.0, lambda_rot=1.0)

    best_val = float('inf')
    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        tr_loss, tr_met = train_epoch(model, loss_fn, train_loader, optimizer, device)
        v_loss, v_met   = validate(model, loss_fn, val_loader, device)
        print(f"Train loss {tr_loss:.4f} adj {tr_met['adj']:.4f} "
              f"dir {tr_met['dir']:.4f} rot {tr_met['rot']:.4f}")
        print(f"Val   loss {v_loss:.4f} adj {v_met['adj']:.4f} "
              f"dir {v_met['dir']:.4f} rot {v_met['rot']:.4f}")
        if v_loss < best_val:
            best_val = v_loss
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': v_loss},
                       'checkpoint_best.pth')
            print("[Saved best checkpoint]")
    torch.save({'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': v_loss},
               'final.pth')
