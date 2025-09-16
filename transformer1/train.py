import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm

# ===== Dataset =====
class PuzzlePairDataset(Dataset):
    DIR_MAPPING = {
        'non-neighbor': 0, 'noneighbor': 0,
        'left': 1, 'right': 2, 'up': 3, 'down': 4
    }

    def __init__(self, jsonl_path, img_size=224):
        self.entries = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.entries.append(json.loads(line))

        self.img_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.contour_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img1 = Image.open(entry['image1']).convert('RGBA')
        img2 = Image.open(entry['image2']).convert('RGBA')

        rgb1, mask1 = img1.convert('RGB'), img1.getchannel('A')
        rgb2, mask2 = img2.convert('RGB'), img2.getchannel('A')

        t_rgb1  = self.img_transform(rgb1)
        t_mask1 = self.contour_transform(mask1)
        t_rgb2  = self.img_transform(rgb2)
        t_mask2 = self.contour_transform(mask2)

        raw_dir = entry['direction']
        if isinstance(raw_dir, str):
            neighbor_label = int(raw_dir) if raw_dir.isdigit() else self.DIR_MAPPING.get(raw_dir.lower(), 0)
        else:
            neighbor_label = int(raw_dir)

        rot1 = int(entry['rotation1']) // 90
        rot2 = int(entry['rotation2']) // 90

        return {
            'rgb1': t_rgb1, 'cont1': t_mask1,
            'rgb2': t_rgb2, 'cont2': t_mask2,
            'neighbor': torch.tensor(neighbor_label, dtype=torch.long),
            'rot1': torch.tensor(rot1, dtype=torch.long),
            'rot2': torch.tensor(rot2, dtype=torch.long)
        }

# ===== Model =====
class FragmentEncoder(nn.Module):
    def __init__(self,
                 model_name='swin_base_patch4_window7_224',
                 pretrained_rgb=True,
                 pretrained_contour=False,
                 embed_dim=1024):
        super().__init__()
        self.rgb_encoder = timm.create_model(model_name, pretrained=pretrained_rgb, num_classes=0)
        self.cont_encoder = timm.create_model(model_name, pretrained=pretrained_contour, in_chans=1, num_classes=0)

        # Получаем размерность фичей
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            ch = self.rgb_encoder(dummy).shape[1]

        self.embed_dim = embed_dim

        # Проекционные слои
        self.proj_rgb = nn.Linear(ch, self.embed_dim)
        self.proj_cont = nn.Linear(ch, self.embed_dim)

        # Cross-attention
        self.cross_attn_img2cont = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)
        self.cross_attn_cont2img = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8, batch_first=True)

        # Fusion MLP
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * self.embed_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(self, x_rgb, x_cont):
        # Получаем фичи
        f_rgb = self.rgb_encoder(x_rgb)  # [B, C]
        f_cont = self.cont_encoder(x_cont)  # [B, C]

        # Добавляем размерность последовательности (N=1)
        f_rgb = f_rgb.unsqueeze(1)  # [B, 1, C]
        f_cont = f_cont.unsqueeze(1)  # [B, 1, C]

        # Проекция
        t_rgb_proj = self.proj_rgb(f_rgb)
        t_cont_proj = self.proj_cont(f_cont)

        # Cross-attention
        attn1, _ = self.cross_attn_img2cont(query=t_rgb_proj, key=t_cont_proj, value=t_cont_proj)
        attn2, _ = self.cross_attn_cont2img(query=t_cont_proj, key=t_rgb_proj, value=t_rgb_proj)

        # Среднее по последовательности и конкатенация
        v1 = attn1.mean(dim=1)
        v2 = attn2.mean(dim=1)
        v = torch.cat([v1, v2], dim=1)

        return self.fuse_mlp(v)

class PuzzleNet(nn.Module):
    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.fragment_enc = FragmentEncoder(**encoder_kwargs)
        D = self.fragment_enc.embed_dim
        self.pair_mlp = nn.Sequential(nn.Linear(4*D, D), nn.ReLU(inplace=True))
        self.neighbor_head = nn.Linear(D, 5)
        self.rot_diff_head = nn.Linear(D, 4)

    def forward(self, rgb1, cont1, rgb2, cont2):
        f1 = self.fragment_enc(rgb1, cont1)
        f2 = self.fragment_enc(rgb2, cont2)
        diff = torch.abs(f1 - f2)
        prod = f1 * f2
        x = torch.cat([f1, f2, diff, prod], dim=1)
        h = self.pair_mlp(x)
        return self.neighbor_head(h), self.rot_diff_head(h)

# ===== Train & Validation =====
def train_one_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    pbar = tqdm(loader, desc='Train', leave=False)
    for batch in pbar:
        rgb1, cont1 = batch['rgb1'].to(device), batch['cont1'].to(device)
        rgb2, cont2 = batch['rgb2'].to(device), batch['cont2'].to(device)
        y_nb = batch['neighbor'].to(device)
        y_r1, y_r2 = batch['rot1'].to(device), batch['rot2'].to(device)
        opt.zero_grad()
        nb_logits, rot_logits = model(rgb1, cont1, rgb2, cont2)
        rot_diff = (y_r2 - y_r1) % 4
        loss = F.cross_entropy(nb_logits, y_nb) + F.cross_entropy(rot_logits, rot_diff)
        loss.backward()
        opt.step()
        total += loss.item() * rgb1.size(0)
        pbar.set_postfix(loss=loss.item())
    return total / len(loader.dataset)

def validate(model, loader, device):
    model.eval()
    total_loss = correct_nb = correct_rt = cnt = 0
    pbar = tqdm(loader, desc='Val  ', leave=False)
    with torch.no_grad():
        for batch in pbar:
            rgb1, cont1 = batch['rgb1'].to(device), batch['cont1'].to(device)
            rgb2, cont2 = batch['rgb2'].to(device), batch['cont2'].to(device)
            y_nb = batch['neighbor'].to(device)
            y_r1, y_r2 = batch['rot1'].to(device), batch['rot2'].to(device)
            nb_logits, rot_logits = model(rgb1, cont1, rgb2, cont2)
            rot_diff = (y_r2 - y_r1) % 4
            loss = F.cross_entropy(nb_logits, y_nb) + F.cross_entropy(rot_logits, rot_diff)
            total_loss += loss.item() * rgb1.size(0)
            pred_nb = nb_logits.argmax(dim=1)
            pred_rt = rot_logits.argmax(dim=1)
            correct_nb += (pred_nb == y_nb).sum().item()
            correct_rt += (pred_rt == rot_diff).sum().item()
            cnt += rgb1.size(0)
    return total_loss/cnt, correct_nb/cnt, correct_rt/cnt

# ===== Main =====
if __name__ == '__main__':
    DATA_DIR   = Path(r'D:\jigsaw\last_att')
    IMG_SIZE   = 224
    BATCH_SIZE = 16
    LR         = 1e-4
    EPOCHS     = 10
    MODEL_NAME = 'swin_tiny_patch4_window7_224'

    train_loader = DataLoader(
        PuzzlePairDataset(DATA_DIR/'pairs_train'/'pair_info.jsonl', IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(
        PuzzlePairDataset(DATA_DIR/'pairs_val'/'pair_info.jsonl', IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PuzzleNet(model_name=MODEL_NAME, pretrained_rgb=True, pretrained_contour=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_loss = float('inf')
    best_path = 'best_puzzle_swin.pth'
    for epoch in range(1, EPOCHS+1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, acc_nb, acc_rt = validate(model, val_loader, device)
        print(f"Epoch {epoch}/{EPOCHS} | Train: {tr_loss:.4f} | Val: {val_loss:.4f} | Neigh Acc: {acc_nb:.4f} | Rot Acc: {acc_rt:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model at epoch {epoch}, val_loss={best_loss:.4f}")
    torch.save(model.state_dict(), 'last_puzzle_swin.pth')
    print("Done training. Last model saved as 'last_puzzle_swin.pth'.")
