import os
import glob
import random
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.models import vgg16, VGG16_Weights

# Fix for the torchmetrics and torch-fidelity installation error
try:
    from torchmetrics.image.fid import FrechetInceptionDistance
except ImportError:
    os.system('pip install -q torchmetrics[image] torch-fidelity')
    from torchmetrics.image.fid import FrechetInceptionDistance

# ==========================================
# HYPERPARAMETERS & DEVICE SETUP
# ==========================================
IMG_SIZE = 256
BATCH_SIZE = 16 
EPOCHS = 50
LR = 0.0002
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {DEVICE} | GPU Count: {torch.cuda.device_count()}")

# ==========================================
# 1. DATA PREPARATION (80/20 Split)
# ==========================================
BASE_DIR = "/kaggle/input"
all_files = glob.glob(f"{BASE_DIR}/**/*.*", recursive=True)

ct_paths = [f for f in all_files if "ct" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
mri_paths = [f for f in all_files if "mri" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if len(ct_paths) == 0 or len(mri_paths) == 0:
    raise ValueError("Dataset paths empty! Ensure the Kaggle dataset is attached.")

random.seed(42)
random.shuffle(ct_paths)
random.shuffle(mri_paths)

split_ct, split_mri = int(0.8 * len(ct_paths)), int(0.8 * len(mri_paths))
train_ct, val_ct = ct_paths[:split_ct], ct_paths[split_ct:]
train_mri, val_mri = mri_paths[:split_mri], mri_paths[split_mri:]

class CTMRIDataset(Dataset):
    def __init__(self, ct_paths, mri_paths, transform=None):
        self.ct_paths, self.mri_paths = ct_paths, mri_paths
        self.transform = transform
        self.length = max(len(self.ct_paths), len(self.mri_paths))

    def __len__(self): return self.length

    def __getitem__(self, idx):
        ct_img = Image.open(self.ct_paths[idx % len(self.ct_paths)]).convert("RGB")
        mri_img = Image.open(random.choice(self.mri_paths)).convert("RGB")
        if self.transform:
            ct_img, mri_img = self.transform(ct_img), self.transform(mri_img)
        return ct_img, mri_img

transform_train = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_val = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_loader = DataLoader(CTMRIDataset(train_ct, train_mri, transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(CTMRIDataset(val_ct, val_mri, transform_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

# ==========================================
# 2. ADVANCED MODELS & REPLAY BUFFER
# ==========================================
class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), nn.InstanceNorm2d(channels)
        )
    def forward(self, x): return x + self.block(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        model = [
            nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.ReLU(inplace=True)
        ]
        for _ in range(9): model += [ResidualBlock(256)]
        model += [
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.InstanceNorm2d(64), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
    def forward(self, x): return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )
    def forward(self, x): return self.model(x)

G_CT2MRI = nn.DataParallel(Generator()).to(DEVICE)
G_MRI2CT = nn.DataParallel(Generator()).to(DEVICE)
D_MRI    = nn.DataParallel(Discriminator()).to(DEVICE)
D_CT     = nn.DataParallel(Discriminator()).to(DEVICE)

# ==========================================
# 3. METRICS & LOSSES
# ==========================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval().to(DEVICE)
        for param in self.parameters(): param.requires_grad = False
    def forward(self, x, y):
        return F.l1_loss(self.slice((x + 1)/2), self.slice((y + 1)/2))

vgg_loss = VGGPerceptualLoss()

def sobel_filter(img):
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1,1,3,3).repeat(3,1,1,1).to(DEVICE)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1,1,3,3).repeat(3,1,1,1).to(DEVICE)
    return torch.sqrt(F.conv2d(img, kx, padding=1, groups=3)**2 + F.conv2d(img, ky, padding=1, groups=3)**2 + 1e-6)

opt_G = optim.Adam(list(G_CT2MRI.parameters()) + list(G_MRI2CT.parameters()), lr=LR, betas=(0.5, 0.999))
opt_D = optim.Adam(list(D_MRI.parameters()) + list(D_CT.parameters()), lr=LR, betas=(0.5, 0.999))

# Updated to non-deprecated AMP syntax
scaler = torch.amp.GradScaler('cuda')

criterion_GAN = nn.MSELoss()
criterion_Cycle = nn.L1Loss()
criterion_Identity = nn.L1Loss()

buffer_mri = ReplayBuffer()
buffer_ct = ReplayBuffer()

# ==========================================
# 4. OSATTA DOMAIN ADAPTATION (Fixed State Drift)
# ==========================================
def osatta_test_time_adaptation(generator, img, steps=1):
    original_state = copy.deepcopy(generator.state_dict())
    
    generator.train()
    norm_params = [p for m in generator.modules() if isinstance(m, nn.InstanceNorm2d) for p in m.parameters() if p.requires_grad]
    
    if len(norm_params) > 0:
        opt_tta = optim.Adam(norm_params, lr=1e-4)
        img_var = img.clone().requires_grad_(True)
        
        for _ in range(steps):
            opt_tta.zero_grad()
            output = generator(img_var)
            
            p = torch.softmax(output, dim=1)
            entropy_loss = -(p * torch.log(p + 1e-6)).sum(dim=1).mean()
            
            entropy_loss.backward()
            opt_tta.step()

    generator.eval()
    with torch.no_grad():
        adapted_output = generator(img)
        
    # Restore state to prevent model drift during validation
    generator.load_state_dict(original_state)
    return adapted_output

# ==========================================
# 5. TRAINING & VALIDATION
# ==========================================
print("Beginning Training Loop...")
os.makedirs("outputs", exist_ok=True)

# Bidirectional Evaluation setup
fid_metric_mri = FrechetInceptionDistance(feature=64).to(DEVICE)
fid_metric_ct = FrechetInceptionDistance(feature=64).to(DEVICE)

for epoch in range(EPOCHS):
    G_CT2MRI.train()
    G_MRI2CT.train()
    
    for i, (real_ct, real_mri) in enumerate(train_loader):
        real_ct, real_mri = real_ct.to(DEVICE), real_mri.to(DEVICE)
        
        # --- TRAIN GENERATORS ---
        opt_G.zero_grad()
        with torch.amp.autocast('cuda'):
            fake_mri = G_CT2MRI(real_ct)
            fake_ct = G_MRI2CT(real_mri)
            
            loss_id = (criterion_Identity(G_CT2MRI(real_mri), real_mri) + criterion_Identity(G_MRI2CT(real_ct), real_ct)) * 5.0
            valid = torch.ones_like(D_MRI(fake_mri), device=DEVICE)
            loss_gan = criterion_GAN(D_MRI(fake_mri), valid) + criterion_GAN(D_CT(fake_ct), valid)
            
            loss_cycle = (criterion_Cycle(G_MRI2CT(fake_mri), real_ct) + criterion_Cycle(G_CT2MRI(fake_ct), real_mri)) * 10.0
            
            loss_struct = (criterion_Cycle(sobel_filter(real_ct), sobel_filter(fake_mri)) + 
                           criterion_Cycle(sobel_filter(real_mri), sobel_filter(fake_ct))) * 2.0
            
            loss_perceptual = (vgg_loss(real_ct, fake_mri) + vgg_loss(real_mri, fake_ct)) * 0.5 
            
            loss_G = loss_gan + loss_cycle + loss_id + loss_struct + loss_perceptual
            
        scaler.scale(loss_G).backward()
        scaler.step(opt_G)
        
        # --- TRAIN DISCRIMINATORS ---
        opt_D.zero_grad()
        with torch.amp.autocast('cuda'):
            fake = torch.zeros_like(valid, device=DEVICE)
            
            fake_mri_buf = buffer_mri.push_and_pop(fake_mri.detach())
            fake_ct_buf = buffer_ct.push_and_pop(fake_ct.detach())
            
            loss_D_MRI = (criterion_GAN(D_MRI(real_mri), valid) + criterion_GAN(D_MRI(fake_mri_buf), fake)) * 0.5
            loss_D_CT = (criterion_GAN(D_CT(real_ct), valid) + criterion_GAN(D_CT(fake_ct_buf), fake)) * 0.5
            loss_D = loss_D_MRI + loss_D_CT
            
        scaler.scale(loss_D).backward()
        scaler.step(opt_D)
        scaler.update()

        if i % 50 == 0:
            print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(train_loader)}] "
                  f"Loss D: {loss_D.item():.4f} | Loss G: {loss_G.item():.4f} | "
                  f"Struct Loss: {loss_struct.item():.4f}")

    # ==========================================
    # EVALUATION PHASE
    # ==========================================
    print(f"Running Validation (Epoch {epoch})...")
    fid_metric_mri.reset()
    fid_metric_ct.reset()
    struct_loss_mri_accum = 0.0
    struct_loss_ct_accum = 0.0
    batches_evaluated = 0
    
    with torch.no_grad():
        for b, (val_ct, val_mri) in enumerate(val_loader):
            val_ct, val_mri = val_ct.to(DEVICE), val_mri.to(DEVICE)
            
            # Bidirectional OSATTA Inference
            with torch.enable_grad():
                fake_mri_val = osatta_test_time_adaptation(G_CT2MRI, val_ct)
                fake_ct_val = osatta_test_time_adaptation(G_MRI2CT, val_mri)
            
            # 1. Structural Edge Evaluation (CT -> MRI and MRI -> CT)
            struct_loss_mri_accum += F.l1_loss(sobel_filter(fake_mri_val), sobel_filter(val_ct)).item()
            struct_loss_ct_accum += F.l1_loss(sobel_filter(fake_ct_val), sobel_filter(val_mri)).item()
            
            # 2. FID Score Updates
            real_mri_uint8 = ((val_mri + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            fake_mri_uint8 = ((fake_mri_val + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            
            real_ct_uint8 = ((val_ct + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            fake_ct_uint8 = ((fake_ct_val + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
            
            fid_metric_mri.update(real_mri_uint8, real=True)
            fid_metric_mri.update(fake_mri_uint8, real=False)
            
            fid_metric_ct.update(real_ct_uint8, real=True)
            fid_metric_ct.update(fake_ct_uint8, real=False)
            
            batches_evaluated += 1
            if b == 0: 
                # Save sample images on first batch for visual inspection
                comp_tensor_mri = torch.cat([val_ct[:4], fake_mri_val[:4]], dim=0)
                comp_tensor_ct = torch.cat([val_mri[:4], fake_ct_val[:4]], dim=0)
                vutils.save_image(comp_tensor_mri, f"outputs/epoch_{epoch}_CT_to_MRI.png", nrow=4, normalize=True)
                vutils.save_image(comp_tensor_ct, f"outputs/epoch_{epoch}_MRI_to_CT.png", nrow=4, normalize=True)
                
            if b == 10: 
                break

    # Compute final metrics
    fid_score_mri = fid_metric_mri.compute()
    fid_score_ct = fid_metric_ct.compute()
    avg_struct_mri = struct_loss_mri_accum / batches_evaluated
    avg_struct_ct = struct_loss_ct_accum / batches_evaluated
    
    print(f"EPOCH {epoch} EVALUATION:")
    print(f"  CT->MRI | FID: {fid_score_mri.item():.2f} | Struct Loss: {avg_struct_mri:.4f}")
    print(f"  MRI->CT | FID: {fid_score_ct.item():.2f}  | Struct Loss: {avg_struct_ct:.4f}")

torch.save(G_CT2MRI.state_dict(), "G_CT_to_MRI_final.pth")
torch.save(G_MRI2CT.state_dict(), "G_MRI_to_CT_final.pth")
print("Model Training Complete. Checkpoints Saved.")
