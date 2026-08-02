import argparse
import os
import random
import sys
import shutil
import urllib.request
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.201)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TINY_IMAGENET_URL = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
PIXEL_ACC_THRESHOLD = 0.1
_ACTIVE_MEAN = CIFAR10_MEAN
_ACTIVE_STD = CIFAR10_STD
TRIM_METRIC_DEFINITIONS = 'Metric definitions (pixel MSE, lower is better; x = the INPUT image):\n  Internal dict keys use val_recon_full / val_recon_trim*; TensorBoard scalar stems use path names.\n  val_full = MSE(x, x_full)  : fc_dec(z) -> dec3_up -> dec2_up -> dec1_up  [TB: val_recon_3layer_latent]\n  trim1    = MSE(x, x_trim1) : fc_dec removed; e3 -> dec3_up -> dec2_up -> dec1_up  [TB: val_recon_3layer]\n  trim2    = MSE(x, x_trim2) : fc_dec + dec3_up removed; e2 -> dec2_up -> dec1_up  [TB: val_recon_2layer]\n  trim3    = MSE(x, x_trim3) : fc_dec + dec3_up + dec2_up removed; e1 -> dec1_up  [TB: val_recon_1layer]\n  Accuracy TB stems mirror recon: val_acc_3layer_latent, val_acc_3layer, val_acc_2layer, val_acc_1layer.\nae_recon trains on MSE(x, x_full) only (trim metrics are reported for comparison).\nae_layer trains feature matching: a^0 MSE(x,x_full) + a^1 MSE(e3,d3)\n                                + a^2 MSE(e2,d2) + a^3 MSE(e1,d1)\n  where d3=relu(fc_dec(z)), d2=relu(dec3_up(d3)), d1=relu(dec2_up(d2)).\n'
ADAPT_TRIM_METRIC_DEFINITIONS = 'ae_layer_adapt metric definitions (pixel MSE, lower is better; x = INPUT):\n  Internal dict keys use val_recon_full / val_recon_trim*; TensorBoard stems: 3layer_latent, 2layer_latent, 1layer_latent.\n  val_full  = MSE(x, x_full)  : fc_enc(e3) -> z -> fc_dec -> dec3_up -> dec2_up -> dec1_up  [TB: val_recon_3layer_latent]\n  val_trim1 = MSE(x, x_trim1) : fc_enc_trim1(e2) -> z -> fc_dec_trim1 -> dec2_up -> dec1_up  [TB: val_recon_2layer_latent]\n  val_trim2 = MSE(x, x_trim2) : fc_enc_trim2(e1) -> z -> fc_dec_trim2 -> dec1_up  [TB: val_recon_1layer_latent]\n  (3 paths only; no e3-splice trim level.)\n'
ADAPT2_LOSS_DEFINITIONS = 'ae_layer_adapt2 training loss (per sampled path; z=fc_enc(e3), z1=fc_enc_trim1(e2), z2=fc_enc_trim2(e1)):\n  full  = MSE(x,x_full) + alpha*(MSE(z,z1)+MSE(d2,d_trim1(z))) + alpha^2*(MSE(z,z2)+MSE(d1,d_trim2(z)))\n  trim1 = MSE(x,x_trim1) + alpha*(MSE(z1,z2)+MSE(dec2_up(d_trim1(z1)),d_trim2(z1)))\n  trim2 = MSE(x,x_trim2)\n'
ADAPT2_COMPONENT_KEYS = ('pixel', 'bridge_lat_a', 'bridge_spat_a', 'bridge_lat_a2', 'bridge_spat_a2')

class EncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)

class ConvAutoencoder(nn.Module):
    """Conv autoencoder with no skip_conv layers.

    Trim-eval splices an encoder feature directly into the decoder after removing
    the bottom decoder stage(s):
      val_full : fc_dec(z) -> dec3_up -> dec2_up -> dec1_up
      trim1    : e3        -> dec3_up -> dec2_up -> dec1_up  (fc_dec removed)
      trim2    : e2        -> dec2_up -> dec1_up             (fc_dec + dec3_up removed)
      trim3    : e1        -> dec1_up                        (fc_dec + dec3_up + dec2_up removed)
    """

    def __init__(self, latent_dim=128, channels=(16, 32, 64), bottleneck_size=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_size = bottleneck_size
        c1, c2, c3 = channels
        self._c3 = c3
        self.enc1 = EncoderBlock(3, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        feat = c3 * bottleneck_size * bottleneck_size
        self.fc_enc = nn.Linear(feat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feat)
        self.dec3_up = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec2_up = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1_up = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.fc_enc(e3.flatten(1))
        return (e1, e2, e3, bottleneck)

    def decode_full(self, bottleneck):
        x = F.relu(self.fc_dec(bottleneck).view(-1, self._c3, self.bottleneck_size, self.bottleneck_size))
        x = F.relu(self.dec3_up(x))
        x = F.relu(self.dec2_up(x))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim1(self, e3):
        x = F.relu(self.dec3_up(e3))
        x = F.relu(self.dec2_up(x))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim2(self, e2):
        x = F.relu(self.dec2_up(e2))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim3(self, e1):
        x = F.relu(self.dec1_up(e1))
        return self.out_conv(x)

    def decode_with_intermediates(self, bottleneck):
        """Full decode path returning the intermediate activations used by the
        layer loss. d3/d2/d1 are the inputs to dec3_up/dec2_up/dec1_up and match
        the encoder features e3/e2/e1 in shape."""
        d3 = F.relu(self.fc_dec(bottleneck).view(-1, self._c3, self.bottleneck_size, self.bottleneck_size))
        d2 = F.relu(self.dec3_up(d3))
        d1 = F.relu(self.dec2_up(d2))
        x_full = self.out_conv(F.relu(self.dec1_up(d1)))
        return (x_full, d3, d2, d1)

    def forward(self, x):
        _, _, _, bottleneck = self.encode(x)
        return self.decode_full(bottleneck)

class LayerAdaptAutoencoder(ConvAutoencoder):
    """Layer AE with shared-latent bridges at e2 (trim1) and e1 (trim2).

    Paths (3 only):
      full  : fc_enc(e3) -> z -> fc_dec -> full decoder
      trim1 : fc_enc_trim1(e2) -> z -> fc_dec_trim1 -> dec2_up -> dec1_up
      trim2 : fc_enc_trim2(e1) -> z -> fc_dec_trim2 -> dec1_up
    """

    def __init__(self, latent_dim=128, channels=(16, 32, 64), bottleneck_size=8):
        super().__init__(latent_dim=latent_dim, channels=channels, bottleneck_size=bottleneck_size)
        c1, c2, _ = channels
        self._c1 = c1
        self._c2 = c2
        e1_spatial = bottleneck_size * 4
        e2_spatial = bottleneck_size * 2
        e1_dim = c1 * e1_spatial * e1_spatial
        e2_dim = c2 * e2_spatial * e2_spatial
        self._e1_shape = (c1, e1_spatial, e1_spatial)
        self._e2_shape = (c2, e2_spatial, e2_spatial)
        self.fc_enc_trim1 = nn.Linear(e2_dim, latent_dim)
        self.fc_dec_trim1 = nn.Linear(latent_dim, e2_dim)
        self.fc_enc_trim2 = nn.Linear(e1_dim, latent_dim)
        self.fc_dec_trim2 = nn.Linear(latent_dim, e1_dim)

    def encode_trim1(self, e2):
        """e2 -> z via fc_enc_trim1."""
        return self.fc_enc_trim1(e2.flatten(1))

    def encode_trim2(self, e1):
        """e1 -> z via fc_enc_trim2."""
        return self.fc_enc_trim2(e1.flatten(1))

    def _dec_entry_trim1(self, z):
        """z -> d2 (dec2_up input) via fc_dec_trim1."""
        return F.relu(self.fc_dec_trim1(z).view(-1, *self._e2_shape))

    def _dec_entry_trim2(self, z):
        """z -> d1 (dec1_up input) via fc_dec_trim2."""
        return F.relu(self.fc_dec_trim2(z).view(-1, *self._e1_shape))

    def decode_trim1(self, e2):
        """trim1 path: e2 -> recon image."""
        x = self._dec_entry_trim1(self.encode_trim1(e2))
        x = F.relu(self.dec2_up(x))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim2(self, e1):
        """trim2 path: e1 -> recon image."""
        x = self._dec_entry_trim2(self.encode_trim2(e1))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def forward_trim1(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        return self.decode_trim1(e2)

    def forward_trim2(self, x):
        e1 = self.enc1(x)
        return self.decode_trim2(e1)

class FirstLayerAutoencoder(nn.Module):
    """Standalone first encoder stage + LayerAdapt trim2 latent bridges.

    Path: enc1 -> e1 -> fc_enc_trim2 -> z -> fc_dec_trim2 -> dec1_up -> out_conv
    """

    def __init__(self, latent_dim=128, channels=(16, 32, 64), bottleneck_size=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_size = bottleneck_size
        c1, _, _ = channels
        self._c1 = c1
        e1_spatial = bottleneck_size * 4
        e1_dim = c1 * e1_spatial * e1_spatial
        self._e1_shape = (c1, e1_spatial, e1_spatial)
        self.enc1 = EncoderBlock(3, c1)
        self.fc_enc_trim2 = nn.Linear(e1_dim, latent_dim)
        self.fc_dec_trim2 = nn.Linear(latent_dim, e1_dim)
        self.dec1_up = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def encode_trim2(self, e1):
        return self.fc_enc_trim2(e1.flatten(1))

    def _dec_entry_trim2(self, z):
        return F.relu(self.fc_dec_trim2(z).view(-1, *self._e1_shape))

    def forward(self, x):
        e1 = self.enc1(x)
        x = self._dec_entry_trim2(self.encode_trim2(e1))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

class GluedAutoencoder(nn.Module):
    """Full autoencoder copy for flowers fine-tuning (same layout as ConvAutoencoder)."""

    def __init__(self, latent_dim=128, channels=(16, 32, 64), bottleneck_size=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_size = bottleneck_size
        c1, c2, c3 = channels
        self._c3 = c3
        self.enc1 = EncoderBlock(3, c1)
        self.enc2 = EncoderBlock(c1, c2)
        self.enc3 = EncoderBlock(c2, c3)
        feat = c3 * bottleneck_size * bottleneck_size
        self.fc_enc = nn.Linear(feat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, feat)
        self.dec3_up = nn.ConvTranspose2d(c3, c2, kernel_size=4, stride=2, padding=1)
        self.dec2_up = nn.ConvTranspose2d(c2, c1, kernel_size=4, stride=2, padding=1)
        self.dec1_up = nn.ConvTranspose2d(c1, c1, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(c1, 3, kernel_size=3, padding=1)

    def encode(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        bottleneck = self.fc_enc(e3.flatten(1))
        return (e1, e2, e3, bottleneck)

    def decode_full(self, bottleneck):
        x = F.relu(self.fc_dec(bottleneck).view(-1, self._c3, self.bottleneck_size, self.bottleneck_size))
        x = F.relu(self.dec3_up(x))
        x = F.relu(self.dec2_up(x))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim1(self, e3):
        x = F.relu(self.dec3_up(e3))
        x = F.relu(self.dec2_up(x))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim2(self, e2):
        x = F.relu(self.dec2_up(e2))
        x = F.relu(self.dec1_up(x))
        return self.out_conv(x)

    def decode_trim3(self, e1):
        x = F.relu(self.dec1_up(e1))
        return self.out_conv(x)

    def forward(self, x):
        _, _, _, bottleneck = self.encode(x)
        return self.decode_full(bottleneck)

    def forward_bookends(self, x):
        """trim3: enc1 -> dec1_up -> out_conv."""
        return self.forward_trim3(x)

    def forward_trim3(self, x):
        """trim3: enc1 -> dec1_up -> out_conv."""
        e1 = self.enc1(x)
        return self.out_conv(F.relu(self.dec1_up(e1)))

    def forward_trim2(self, x):
        """trim2: enc1 -> enc2 -> dec2_up -> dec1_up -> out_conv."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d1 = F.relu(self.dec2_up(e2))
        return self.out_conv(F.relu(self.dec1_up(d1)))

    def forward_trim1(self, x):
        """trim1: enc1 -> enc2 -> enc3 -> dec3_up -> dec2_up -> dec1_up -> out_conv."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        d2 = F.relu(self.dec3_up(e3))
        d1 = F.relu(self.dec2_up(d2))
        return self.out_conv(F.relu(self.dec1_up(d1)))

def build_glue_model(pretrained_ckpt, device, latent_dim=128, bottleneck_size=8):
    """Build GluedAutoencoder initialized from a full ConvAutoencoder checkpoint."""
    model = GluedAutoencoder(latent_dim=latent_dim, bottleneck_size=bottleneck_size).to(device)
    checkpoint = torch.load(pretrained_ckpt, map_location=device, weights_only=False)
    load_model_weights(model, checkpoint['model_state_dict'])
    return model
GLUE_LAYER_LATENT_RUN_NAME = 'flowers102_glue_layer_latent'
GLUE_RECON_LATENT_RUN_NAME = 'flowers102_glue_recon_latent'
GLUE_LAYER_LATENT_RECON_TAG = 'glue_layer_latent'
GLUE_RECON_LATENT_RECON_TAG = 'glue_recon_latent'
LAYER_ADAPT_RUN_NAME = 'tiny_imagenet_ae_layer_adapt_500'
LAYER_ADAPT_RECON_TAG = 'ae_layer_adapt_500'
LAYER_ADAPT_300_RUN_NAME = 'tiny_imagenet_ae_layer_adapt_300'
LAYER_ADAPT_300_RECON_TAG = 'ae_layer_adapt_300'
LAYER_ADAPT2_RUN_NAME = 'tiny_imagenet_ae_layer_adapt2'
LAYER_ADAPT2_RECON_TAG = 'ae_layer_adapt2'
LAYER1_RUN_NAME = 'tiny_imagenet_ae_recon_l1'
LAYER1_RECON_TAG = 'ae_recon_l1'
LAYER1_LOSS_DEFINITION = 'ae_recon_l1: MSE(x, x_hat) on enc1 -> fc_enc_trim2 -> z -> fc_dec_trim2 -> dec1_up -> out_conv only.'
LAYER_ADAPT_PATH_NAMES = ('full', 'trim1', 'trim2')
GLUE_LATENT_MODULES = ('fc_enc', 'fc_dec')
GLUE_NON_LATENT_MODULES = ('enc1', 'enc2', 'enc3', 'dec3_up', 'dec2_up', 'dec1_up', 'out_conv')
DEFAULT_RECON_IMAGES = 16
RECON_IMAGE_COUNT = DEFAULT_RECON_IMAGES
RECON_RANDOM_SEED = 0
ACTIVE_RUNS = ('tiny_imagenet_ae_recon', 'tiny_imagenet_ae_layer_500', LAYER1_RUN_NAME, LAYER_ADAPT_RUN_NAME, LAYER_ADAPT_300_RUN_NAME, LAYER_ADAPT2_RUN_NAME, GLUE_LAYER_LATENT_RUN_NAME, GLUE_RECON_LATENT_RUN_NAME)
CANONICAL_RECON_RUNS = ACTIVE_RUNS
GLUE_RECON_RUNS = (GLUE_LAYER_LATENT_RUN_NAME, GLUE_RECON_LATENT_RUN_NAME)
RECON_TAG_BY_RUN = {'tiny_imagenet_ae_recon': 'ae_recon', 'tiny_imagenet_ae_layer_500': 'ae_layer_500', LAYER1_RUN_NAME: LAYER1_RECON_TAG, LAYER_ADAPT_RUN_NAME: LAYER_ADAPT_RECON_TAG, LAYER_ADAPT_300_RUN_NAME: LAYER_ADAPT_300_RECON_TAG, LAYER_ADAPT2_RUN_NAME: LAYER_ADAPT2_RECON_TAG, GLUE_LAYER_LATENT_RUN_NAME: GLUE_LAYER_LATENT_RECON_TAG, GLUE_RECON_LATENT_RUN_NAME: GLUE_RECON_LATENT_RECON_TAG}

def reconstruction_tag_for_run(run_name):
    """One unique TensorBoard reconstruction tag per model run."""
    if run_name not in RECON_TAG_BY_RUN:
        raise KeyError(f'No reconstruction tag configured for run: {run_name}')
    return RECON_TAG_BY_RUN[run_name]

def adapt_path_weights(alpha):
    return (1.0, alpha, alpha ** 2)

def sample_adapt_path_name(alpha):
    return random.choices(LAYER_ADAPT_PATH_NAMES, weights=adapt_path_weights(alpha), k=1)[0]

def adapt_stoch_forward(model, path_name):
    return {'full': model.forward, 'trim1': model.forward_trim1, 'trim2': model.forward_trim2}[path_name]

def freeze_glue_non_latent(model):
    """Freeze all modules except fc_enc/fc_dec (latent-only fine-tune)."""
    for name in GLUE_NON_LATENT_MODULES:
        for p in getattr(model, name).parameters():
            p.requires_grad = False

def set_glue_non_latent_eval(model):
    """Keep frozen conv/BN modules in eval during latent-only training."""
    for name in GLUE_NON_LATENT_MODULES:
        getattr(model, name).eval()

def glue_latent_parameters(model):
    params = []
    for name in GLUE_LATENT_MODULES:
        params.extend(getattr(model, name).parameters())
    return params

def resolve_middle_checkpoint(run_name, explicit_path=None):
    """Prefer explicit path, else epoch_500.pt, else latest.pt."""
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path
    for name in ('epoch_500.pt', 'latest.pt'):
        path = os.path.join('checkpoints', run_name, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f'Missing middle checkpoint for {run_name}')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def set_active_normalization(dataset):
    global _ACTIVE_MEAN, _ACTIVE_STD
    if dataset in ('tiny_imagenet', 'flowers102'):
        _ACTIVE_MEAN, _ACTIVE_STD = (IMAGENET_MEAN, IMAGENET_STD)
    else:
        _ACTIVE_MEAN, _ACTIVE_STD = (CIFAR10_MEAN, CIFAR10_STD)

def get_cifar10_loaders(batch_size, num_workers=4):
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    trainloader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    testloader = DataLoader(testset, shuffle=False, **loader_kwargs)
    return (trainloader, testloader)

def download_tiny_imagenet(root='./data'):
    dest_dir = os.path.join(root, 'tiny-imagenet-200')
    if os.path.isdir(dest_dir):
        return dest_dir
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, 'tiny-imagenet-200.zip')
    if not os.path.exists(zip_path):
        print(f'Downloading Tiny ImageNet from {TINY_IMAGENET_URL} ...')
        urllib.request.urlretrieve(TINY_IMAGENET_URL, zip_path)
    print('Extracting Tiny ImageNet ...')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(root)
    return dest_dir

def prepare_tiny_imagenet_val(dest_dir):
    """Reorganize val/images + val_annotations.txt into val/<wnid>/*.JPEG so
    ImageFolder can read class subfolders."""
    val_dir = os.path.join(dest_dir, 'val')
    images_dir = os.path.join(val_dir, 'images')
    annotations = os.path.join(val_dir, 'val_annotations.txt')
    if not os.path.isdir(images_dir):
        return
    mapping = {}
    with open(annotations, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.split('\t')
            if len(parts) >= 2:
                mapping[parts[0]] = parts[1]
    for fname, wnid in mapping.items():
        cls_dir = os.path.join(val_dir, wnid)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, fname)
        if os.path.exists(src):
            shutil.move(src, os.path.join(cls_dir, fname))
    shutil.rmtree(images_dir, ignore_errors=True)

def get_tiny_imagenet_loaders(batch_size, num_workers=4):
    dest_dir = download_tiny_imagenet('./data')
    prepare_tiny_imagenet_val(dest_dir)
    transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    trainset = torchvision.datasets.ImageFolder(os.path.join(dest_dir, 'train'), transform=transform_train)
    testset = torchvision.datasets.ImageFolder(os.path.join(dest_dir, 'val'), transform=transform_test)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    trainloader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    testloader = DataLoader(testset, shuffle=False, **loader_kwargs)
    return (trainloader, testloader)

def get_flowers102_loaders(batch_size, num_workers=4):
    transform_train = transforms.Compose([transforms.Resize((64, 64)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    transform_test = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    trainset = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform_train)
    testset = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform_test)
    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = True
    trainloader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    testloader = DataLoader(testset, shuffle=False, **loader_kwargs)
    return (trainloader, testloader)

def get_loaders(dataset, batch_size, num_workers=4):
    if dataset == 'tiny_imagenet':
        return get_tiny_imagenet_loaders(batch_size, num_workers)
    if dataset == 'cifar10':
        return get_cifar10_loaders(batch_size, num_workers)
    if dataset == 'flowers102':
        return get_flowers102_loaders(batch_size, num_workers)
    raise ValueError(f'Unknown dataset: {dataset}')

def denormalize(tensor):
    mean = torch.tensor(_ACTIVE_MEAN, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(_ACTIVE_STD, device=tensor.device).view(1, 3, 1, 1)
    return torch.clamp(tensor * std + mean, 0.0, 1.0)

def _dataset_indices_by_class(dataset):
    """Map class label -> dataset indices (CIFAR10, ImageFolder, Flowers102)."""
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        if hasattr(targets, 'tolist'):
            targets = targets.tolist()
    elif hasattr(dataset, 'samples'):
        targets = [label for _path, label in dataset.samples]
    else:
        return {}
    by_class = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(int(label), []).append(idx)
    return by_class

def sample_stratified_reconstruction_batch(loader, n_images=None, seed=None):
    """Pick inputs from different class groups; fixed seed = one stratified draw for comparison."""
    if n_images is None:
        n_images = RECON_IMAGE_COUNT
    dataset = loader.dataset
    rng = random.Random(seed)
    by_class = _dataset_indices_by_class(dataset)
    if not by_class:
        indices = list(range(len(dataset)))
        rng.shuffle(indices)
        chosen = indices[:n_images]
    else:
        classes = list(by_class.keys())
        rng.shuffle(classes)
        chosen = []
        for cls in classes:
            if len(chosen) >= n_images:
                break
            chosen.append(rng.choice(by_class[cls]))
        ptr = 0
        while len(chosen) < n_images:
            cls = classes[ptr % len(classes)]
            chosen.append(rng.choice(by_class[cls]))
            ptr += 1
        chosen = chosen[:n_images]
    collate_fn = getattr(loader, 'collate_fn', None)
    if collate_fn is None:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    inputs, _labels = collate_fn([dataset[i] for i in chosen])
    return inputs

def sample_fixed_reconstruction_batch(loader, n_images=None):
    """Fixed first-batch inputs for comparable 3_Reconstruction boards across epochs."""
    if n_images is None:
        n_images = RECON_IMAGE_COUNT
    inputs, _ = next(iter(loader))
    return inputs[:n_images]

def reconstruction_loss(pred, target):
    return F.mse_loss(pred, target)

def get_alpha(epoch, total_epochs, schedule='increasing', alpha_value=1.0, start_epoch=0, alpha_start=0.0, alpha_end=1.0):
    """Return base alpha for the layer loss.

    increasing (default): alpha ramps alpha_start -> alpha_end over [start_epoch, total_epochs).
    constant: fixed alpha_value for all epochs.
    """
    if schedule == 'constant':
        return alpha_value
    span = total_epochs - 1 - start_epoch
    if span <= 0:
        t = 1.0
    else:
        t = (epoch - start_epoch) / span
    if schedule == 'increasing':
        return alpha_start + (alpha_end - alpha_start) * t
    return alpha_start + (alpha_end - alpha_start) * t

def describe_alpha_schedule(args):
    """Human-readable summary of the layer alpha schedule at epoch 1 and final epoch."""
    if args.alpha_schedule == 'constant':
        return f'Alpha schedule: constant, alpha={args.alpha_value:g} (all {args.epochs} epochs)'
    a_first = get_alpha(args.start_epoch, args.epochs, args.alpha_schedule, args.alpha_value, args.start_epoch, args.alpha_start, args.alpha_end)
    a_last = get_alpha(args.epochs - 1, args.epochs, args.alpha_schedule, args.alpha_value, args.start_epoch, args.alpha_start, args.alpha_end)
    epoch_range = f'epochs {args.start_epoch + 1}-{args.epochs}' if args.start_epoch > 0 else f'{args.epochs} epochs'
    return f'Alpha schedule: increasing, alpha {args.alpha_start:g} -> {args.alpha_end:g} over {epoch_range} | epoch {args.start_epoch + 1}: alpha={a_first:.4f} alpha^3={a_first ** 3:.4f} | epoch {args.epochs}: alpha={a_last:.4f}'

def resolve_alpha(epoch, args):
    return get_alpha(epoch, args.epochs, args.alpha_schedule, args.alpha_value, args.start_epoch, args.alpha_start, args.alpha_end)

def layer_encoder_loss(model, inputs, alpha):
    """Feature-matching loss: align encoder features e_i with decoder
    intermediate activations d_i (same resolution).

      L = a^0 * MSE(x, x_full)
        + a^1 * MSE(e3, d3)
        + a^2 * MSE(e2, d2)
        + a^3 * MSE(e1, d1)
    """
    e1, e2, e3, bottleneck = model.encode(inputs)
    x_full, d3, d2, d1 = model.decode_with_intermediates(bottleneck)
    loss_pixel = F.mse_loss(x_full, inputs)
    loss_e3 = F.mse_loss(d3, e3)
    loss_e2 = F.mse_loss(d2, e2)
    loss_e1 = F.mse_loss(d1, e1)
    loss = alpha ** 0 * loss_pixel + alpha ** 1 * loss_e3 + alpha ** 2 * loss_e2 + alpha ** 3 * loss_e1
    components = {'pixel': loss_pixel.item(), 'feat_e3': loss_e3.item(), 'feat_e2': loss_e2.item(), 'feat_e1': loss_e1.item()}
    return (loss, x_full, components)

def adapt2_zero_components():
    return {key: 0.0 for key in ADAPT2_COMPONENT_KEYS}

def adapt2_path_loss(model, inputs, path_name, alpha):
    """Cross-bridge cascade loss per sampled path."""
    components = adapt2_zero_components()
    if path_name == 'full':
        e1, e2, e3, z = model.encode(inputs)
        x_full, _, d2, d1 = model.decode_with_intermediates(z)
        z1 = model.encode_trim1(e2)
        z2 = model.encode_trim2(e1)
        d_trim1 = model._dec_entry_trim1(z)
        d_trim2 = model._dec_entry_trim2(z)
        loss_pixel = F.mse_loss(x_full, inputs)
        bridge_lat_a = F.mse_loss(z, z1)
        bridge_spat_a = F.mse_loss(d2, d_trim1)
        bridge_lat_a2 = F.mse_loss(z, z2)
        bridge_spat_a2 = F.mse_loss(d1, d_trim2)
        loss = loss_pixel + alpha * (bridge_lat_a + bridge_spat_a) + alpha ** 2 * (bridge_lat_a2 + bridge_spat_a2)
        components = {'pixel': loss_pixel.item(), 'bridge_lat_a': bridge_lat_a.item(), 'bridge_spat_a': bridge_spat_a.item(), 'bridge_lat_a2': bridge_lat_a2.item(), 'bridge_spat_a2': bridge_spat_a2.item()}
    elif path_name == 'trim1':
        e1 = model.enc1(inputs)
        e2 = model.enc2(e1)
        x_trim1 = model.decode_trim1(e2)
        z1 = model.encode_trim1(e2)
        z2 = model.encode_trim2(e1)
        d_trim1 = model._dec_entry_trim1(z1)
        d_trim2 = model._dec_entry_trim2(z1)
        d2_trim = F.relu(model.dec2_up(d_trim1))
        loss_pixel = F.mse_loss(x_trim1, inputs)
        bridge_lat_a = F.mse_loss(z1, z2)
        bridge_spat_a = F.mse_loss(d2_trim, d_trim2)
        loss = loss_pixel + alpha * (bridge_lat_a + bridge_spat_a)
        components = {'pixel': loss_pixel.item(), 'bridge_lat_a': bridge_lat_a.item(), 'bridge_spat_a': bridge_spat_a.item(), 'bridge_lat_a2': 0.0, 'bridge_spat_a2': 0.0}
    elif path_name == 'trim2':
        e1 = model.enc1(inputs)
        x_trim2 = model.decode_trim2(e1)
        loss_pixel = F.mse_loss(x_trim2, inputs)
        loss = loss_pixel
        components = {'pixel': loss_pixel.item(), 'bridge_lat_a': 0.0, 'bridge_spat_a': 0.0, 'bridge_lat_a2': 0.0, 'bridge_spat_a2': 0.0}
    else:
        raise ValueError(f'Unknown adapt2 path: {path_name}')
    return (loss, components)

def evaluate_all_recon_metrics(model, loader, device, threshold=PIXEL_ACC_THRESHOLD):
    """Encode once per batch, decode each trim path. Returns per-element pixel
    MSE and pixel accuracy (all RGB within threshold) for each path."""
    model.eval()
    totals = {'val_recon_full': 0.0, 'val_recon_trim1': 0.0, 'val_recon_trim2': 0.0, 'val_recon_trim3': 0.0}
    acc_correct = {'val_acc_full': 0, 'val_acc_trim1': 0, 'val_acc_trim2': 0, 'val_acc_trim3': 0}
    total_elements = 0
    total_pixels = 0
    skip_levels = (
        ('val_recon_full', 'val_acc_full', lambda m, z, e1, e2, e3: m.decode_full(z)),
        ('val_recon_trim1', 'val_acc_trim1', lambda m, z, e1, e2, e3: m.decode_trim1(e3)),
        ('val_recon_trim2', 'val_acc_trim2', lambda m, z, e1, e2, e3: m.decode_trim2(e2)),
        ('val_recon_trim3', 'val_acc_trim3', lambda m, z, e1, e2, e3: m.decode_trim3(e1)),
    )
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            batch_pixels = inputs.size(0) * inputs.size(2) * inputs.size(3)
            e1, e2, e3, bottleneck = model.encode(inputs)
            target_dn = denormalize(inputs)
            for mse_key, acc_key, decode_fn in skip_levels:
                recon = decode_fn(model, bottleneck, e1, e2, e3)
                totals[mse_key] += F.mse_loss(recon, inputs, reduction='sum').item()
                recon_dn = denormalize(recon)
                correct = (recon_dn - target_dn).abs() <= threshold
                acc_correct[acc_key] += correct.all(dim=1).sum().item()
            total_elements += inputs.numel()
            total_pixels += batch_pixels
    metrics = {key: totals[key] / total_elements for key in totals}
    for acc_key in acc_correct:
        metrics[acc_key] = 100.0 * acc_correct[acc_key] / total_pixels
    return metrics

def evaluate_adapt_recon_metrics(model, loader, device, threshold=PIXEL_ACC_THRESHOLD):
    """3-path eval for LayerAdaptAutoencoder: full, trim1, trim2."""
    model.eval()
    totals = {'val_recon_full': 0.0, 'val_recon_trim1': 0.0, 'val_recon_trim2': 0.0}
    acc_correct = {'val_acc_full': 0, 'val_acc_trim1': 0, 'val_acc_trim2': 0}
    total_elements = 0
    total_pixels = 0
    skip_levels = (
        ('val_recon_full', 'val_acc_full', lambda m, z, e1, e2, e3: m.decode_full(z)),
        ('val_recon_trim1', 'val_acc_trim1', lambda m, z, e1, e2, e3: m.decode_trim1(e2)),
        ('val_recon_trim2', 'val_acc_trim2', lambda m, z, e1, e2, e3: m.decode_trim2(e1)),
    )
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            batch_pixels = inputs.size(0) * inputs.size(2) * inputs.size(3)
            e1, e2, e3, bottleneck = model.encode(inputs)
            target_dn = denormalize(inputs)
            for mse_key, acc_key, decode_fn in skip_levels:
                recon = decode_fn(model, bottleneck, e1, e2, e3)
                totals[mse_key] += F.mse_loss(recon, inputs, reduction='sum').item()
                recon_dn = denormalize(recon)
                correct = (recon_dn - target_dn).abs() <= threshold
                acc_correct[acc_key] += correct.all(dim=1).sum().item()
            total_elements += inputs.numel()
            total_pixels += batch_pixels
    metrics = {key: totals[key] / total_elements for key in totals}
    for acc_key in acc_correct:
        metrics[acc_key] = 100.0 * acc_correct[acc_key] / total_pixels
    return metrics

def evaluate_reconstruction(forward_fn, loader, device, threshold=PIXEL_ACC_THRESHOLD):
    """Evaluate a forward function on a loader; returns pixel MSE and accuracy."""
    total_mse = 0.0
    acc_correct = 0
    total_elements = 0
    total_pixels = 0
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(device)
            recon = forward_fn(inputs)
            total_mse += F.mse_loss(recon, inputs, reduction='sum').item()
            batch_pixels = inputs.size(0) * inputs.size(2) * inputs.size(3)
            recon_dn = denormalize(recon)
            target_dn = denormalize(inputs)
            correct = (recon_dn - target_dn).abs() <= threshold
            acc_correct += correct.all(dim=1).sum().item()
            total_elements += inputs.numel()
            total_pixels += batch_pixels
    return {'val_recon': total_mse / total_elements, 'val_acc': 100.0 * acc_correct / total_pixels}

def log_glue_three_row_reconstruction(model, loader, writer, device, epoch, tag, bookends_model=None, trim_forward=None, n_images=None):
    """Log input, bookends (1layer), and full (3layer_latent) recon as three rows.

    Row order in the image grid: input | bookends (enc1→dec1_up) | full model.
    Writes fixed boards to 3_Reconstruction and stratified boards to 4_Reconstruction_Random.
    """
    bookends_source = bookends_model if bookends_model is not None else model
    batches = ((TB_GROUP_RECONSTRUCTION, sample_fixed_reconstruction_batch(loader, n_images)), (TB_GROUP_RECONSTRUCTION_RANDOM, sample_stratified_reconstruction_batch(loader, n_images=n_images, seed=RECON_RANDOM_SEED)))
    with torch.no_grad():
        for group, inputs in batches:
            inputs = inputs.to(device)
            if trim_forward is not None:
                bookends_recon = trim_forward(inputs)
            else:
                bookends_recon = bookends_source.forward_bookends(inputs)
            full_recon = model(inputs)
            combined = torch.cat([denormalize(inputs), denormalize(bookends_recon), denormalize(full_recon)], dim=0)
            writer.add_images(f'{group}/{tag}', combined, epoch, dataformats='NCHW')

def train_one_epoch_glue_latent(model, loader, optimizer, device):
    """Full forward MSE; only fc_enc/fc_dec trainable; conv/BN frozen in eval."""
    model.train()
    set_glue_non_latent_eval(model)
    running_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc='Train (glue latent)', unit='batch', ncols=100)
    for inputs, _ in pbar:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon = model(inputs)
        loss = reconstruction_loss(recon, inputs)
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({'Loss': f'{running_loss / total_samples:.6f}'})
    return running_loss / total_samples

def train_one_epoch_pixel(model, loader, optimizer, device, desc='Train'):
    model.train()
    running_loss = 0.0
    total_samples = 0
    pbar = tqdm(loader, desc=desc, unit='batch', ncols=100)
    for inputs, _ in pbar:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        recon = model(inputs)
        loss = reconstruction_loss(recon, inputs)
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        pbar.set_postfix({'Loss': f'{running_loss / total_samples:.6f}'})
    return running_loss / total_samples

def make_glue_latent_stack(model, args):
    """Optimizer/scheduler over fc_enc and fc_dec only."""
    optimizer = optim.SGD(glue_latent_parameters(model), lr=args.lr, momentum=0.9, weight_decay=1e-05)
    milestones = [max(1, args.epochs // 2), max(2, 3 * args.epochs // 4)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return (optimizer, scheduler)

def train_one_epoch_layer_adapt_stoch(model, loader, optimizer, device, alpha):
    """Per-batch stochastic path pixel MSE; path weights are (1, alpha, alpha^2)."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    path_counts = {name: 0 for name in LAYER_ADAPT_PATH_NAMES}
    pbar = tqdm(loader, desc='Train (layer_adapt stoch)', unit='batch', ncols=100)
    for inputs, _ in pbar:
        inputs = inputs.to(device)
        path_name = sample_adapt_path_name(alpha)
        forward_fn = adapt_stoch_forward(model, path_name)
        optimizer.zero_grad()
        recon = forward_fn(inputs)
        loss = reconstruction_loss(recon, inputs)
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        path_counts[path_name] += 1
        pbar.set_postfix({'Loss': f'{running_loss / total_samples:.6f}'})
    return (running_loss / total_samples, path_counts)

def train_one_epoch_layer_adapt2_stoch(model, loader, optimizer, device, alpha):
    """Per-batch stochastic path cross-bridge cascade; path weights (1, alpha, alpha^2)."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    path_counts = {name: 0 for name in LAYER_ADAPT_PATH_NAMES}
    comp_totals = adapt2_zero_components()
    pbar = tqdm(loader, desc='Train (layer_adapt2 stoch)', unit='batch', ncols=100)
    for inputs, _ in pbar:
        inputs = inputs.to(device)
        path_name = sample_adapt_path_name(alpha)
        optimizer.zero_grad()
        loss, components = adapt2_path_loss(model, inputs, path_name, alpha)
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        path_counts[path_name] += 1
        for key in comp_totals:
            comp_totals[key] += components[key] * batch_size
        pbar.set_postfix({'Loss': f'{running_loss / total_samples:.6f}'})
    components_avg = {key: comp_totals[key] / total_samples for key in comp_totals}
    return (running_loss / total_samples, path_counts, components_avg)

def train_one_glue_latent_only(args, run_name, pretrained_ckpt, trainloader, testloader, device):
    """Fine-tune fc_enc/fc_dec only from a pretrained AE checkpoint."""
    log_dir = os.path.join('runs', run_name)
    checkpoint_dir = os.path.join('checkpoints', run_name)
    recon_tag = reconstruction_tag_for_run(run_name)
    print(f'\n--- Training latent-only glue model: {run_name} ---')
    print(f'Init checkpoint: {pretrained_ckpt}')
    print(f'Trainable modules: {GLUE_LATENT_MODULES}')
    model = build_glue_model(pretrained_ckpt, device, latent_dim=args.latent_dim, bottleneck_size=args.bottleneck_size)
    freeze_glue_non_latent(model)
    optimizer, scheduler = make_glue_latent_stack(model, args)
    start_epoch = 0
    latest_path = os.path.join(checkpoint_dir, 'latest.pt')
    if os.path.isfile(latest_path):
        checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        load_model_weights(model, checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        freeze_glue_non_latent(model)
        for group in optimizer.param_groups:
            group['lr'] = args.lr
        milestones = [max(1, args.epochs // 2), max(2, 3 * args.epochs // 4)]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
        fast_forward_scheduler(scheduler, start_epoch)
        print(f"Resumed from {latest_path} at epoch {start_epoch} (scheduler rebuilt for {args.epochs} epochs, milestones={milestones}, lr={optimizer.param_groups[0]['lr']})")
    else:
        fast_forward_scheduler(scheduler, start_epoch)
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch_glue_latent(model, trainloader, optimizer, device)
        model.eval()
        set_glue_non_latent_eval(model)
        val_metrics = evaluate_reconstruction(model, testloader, device)
        bookends_metrics = evaluate_reconstruction(model.forward_bookends, testloader, device)
        scheduler.step()
        step = epoch + 1
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_full', train_loss, step)
        writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', val_metrics['val_recon'], step)
        writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER_LATENT}', val_metrics['val_acc'], step)
        writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER}', bookends_metrics['val_recon'], step)
        writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER}', bookends_metrics['val_acc'], step)
        writer.add_scalar(f'{TB_GROUP_SCHEDULE}/LR', optimizer.param_groups[0]['lr'], step)
        if step % args.log_images_every == 0 or step == args.epochs:
            log_glue_three_row_reconstruction(model, testloader, writer, device, step, recon_tag)
        print(f"[{run_name}] Epoch {step}/{args.epochs} | train={train_loss:.6f} | val_mse={val_metrics['val_recon']:.6f} | val_acc={val_metrics['val_acc']:.2f}% | bookends_mse={bookends_metrics['val_recon']:.6f}")
        if step % args.checkpoint_every == 0 or step == args.epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{step}.pt')
            save_checkpoint(ckpt_path, step, 'glue_latent', model, optimizer, scheduler, train_loss, val_metrics)
            latest_path = os.path.join(checkpoint_dir, 'latest.pt')
            shutil.copy2(ckpt_path, latest_path)
            print(f'Saved checkpoint: {ckpt_path}')
        model.train()
        set_glue_non_latent_eval(model)
    writer.close()
    print(f'Latent-only glue training complete: {run_name}')
    return model

def train_glue_latent_only(args):
    """Independently fine-tune fc_enc/fc_dec of two GluedAutoencoders."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    layer_name = args.layer_run_name or 'tiny_imagenet_ae_layer_500'
    recon_name = args.recon_run_name or 'tiny_imagenet_ae_recon'
    layer_ckpt = resolve_middle_checkpoint(layer_name, args.resume_layer)
    recon_ckpt = resolve_middle_checkpoint(recon_name, args.resume_recon)
    print(f'Using device: {device}')
    print(f'Dataset: {args.dataset}')
    print(f'Epochs: {args.epochs}')
    print('Mode: train_glue (fc_enc/fc_dec only; full MSE)')
    print(f'Trainable modules: {GLUE_LATENT_MODULES}')
    print(f'glue_layer_latent init: {layer_ckpt}')
    print(f'glue_recon_latent init: {recon_ckpt}')
    set_seed(args.seed)
    set_active_normalization(args.dataset)
    trainloader, testloader = get_loaders(args.dataset, args.batch_size, args.num_workers)
    train_one_glue_latent_only(args, GLUE_LAYER_LATENT_RUN_NAME, layer_ckpt, trainloader, testloader, device)
    train_one_glue_latent_only(args, GLUE_RECON_LATENT_RUN_NAME, recon_ckpt, trainloader, testloader, device)
    print('\nLatent-only glue models trained.')

def train_glue(args):
    """Train glue_layer_latent + glue_recon_latent (fc_enc/fc_dec only)."""
    train_glue_latent_only(args)

def train_one_epoch(model, loader, optimizer, device, mode, alpha=None):
    model.train()
    running_loss = 0.0
    total_samples = 0
    comp_totals = {'pixel': 0.0, 'feat_e3': 0.0, 'feat_e2': 0.0, 'feat_e1': 0.0}
    pbar = tqdm(loader, desc=f'Train ({mode})', unit='batch', ncols=100)
    for inputs, _ in pbar:
        inputs = inputs.to(device)
        optimizer.zero_grad()
        components = None
        if mode == 'recon':
            recon = model(inputs)
            loss = reconstruction_loss(recon, inputs)
        elif mode == 'layer':
            loss, _, components = layer_encoder_loss(model, inputs, alpha)
        else:
            raise ValueError(f'Unknown mode: {mode}')
        loss.backward()
        optimizer.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        if components is not None:
            for key in comp_totals:
                comp_totals[key] += components[key] * batch_size
        pbar.set_postfix({'Loss': f'{running_loss / total_samples:.6f}'})
    avg_loss = running_loss / total_samples
    if mode == 'layer':
        return (avg_loss, {key: comp_totals[key] / total_samples for key in comp_totals})
    return (avg_loss, None)

def log_reconstruction_images(model, loader, writer, device, epoch, tag, n_images=None):
    """Log ConvAE recon rows to fixed and random TensorBoard groups.

    Row order: input | 3layer_latent (full) | 3layer (trim1) | 2layer (trim2) | 1layer (trim3).
    """
    model.eval()
    batches = ((TB_GROUP_RECONSTRUCTION, sample_fixed_reconstruction_batch(loader, n_images)), (TB_GROUP_RECONSTRUCTION_RANDOM, sample_stratified_reconstruction_batch(loader, n_images=n_images, seed=RECON_RANDOM_SEED)))
    with torch.no_grad():
        for group, inputs in batches:
            inputs = inputs.to(device)
            e1, e2, e3, bottleneck = model.encode(inputs)
            rows = [denormalize(inputs), denormalize(model.decode_full(bottleneck)), denormalize(model.decode_trim1(e3)), denormalize(model.decode_trim2(e2)), denormalize(model.decode_trim3(e1))]
            combined = torch.cat(rows, dim=0)
            writer.add_images(f'{group}/{tag}', combined, epoch, dataformats='NCHW')

def log_adapt_reconstruction_images(model, loader, writer, device, step, tag, n_images=None):
    """Log LayerAdapt recon rows to fixed and random TensorBoard groups.

    Row order: input | 3layer_latent (full) | 2layer_latent (trim1) | 1layer_latent (trim2).
    """
    model.eval()
    batches = ((TB_GROUP_RECONSTRUCTION, sample_fixed_reconstruction_batch(loader, n_images)), (TB_GROUP_RECONSTRUCTION_RANDOM, sample_stratified_reconstruction_batch(loader, n_images=n_images, seed=RECON_RANDOM_SEED)))
    with torch.no_grad():
        for group, inputs in batches:
            inputs = inputs.to(device)
            e1, e2, e3, bottleneck = model.encode(inputs)
            rows = [denormalize(inputs), denormalize(model.decode_full(bottleneck)), denormalize(model.decode_trim1(e2)), denormalize(model.decode_trim2(e1))]
            combined = torch.cat(rows, dim=0)
            writer.add_images(f'{group}/{tag}', combined, step, dataformats='NCHW')

def log_layer1_reconstruction_images(model, loader, writer, device, epoch, tag, n_images=None):
    """Log FirstLayerAutoencoder recon rows to fixed and random TensorBoard groups.

    Row order: input | 1layer_latent (single-path recon).
    """
    model.eval()
    batches = ((TB_GROUP_RECONSTRUCTION, sample_fixed_reconstruction_batch(loader, n_images)), (TB_GROUP_RECONSTRUCTION_RANDOM, sample_stratified_reconstruction_batch(loader, n_images=n_images, seed=RECON_RANDOM_SEED)))
    with torch.no_grad():
        for group, inputs in batches:
            inputs = inputs.to(device)
            recon = model(inputs)
            combined = torch.cat([denormalize(inputs), denormalize(recon)], dim=0)
            writer.add_images(f'{group}/{tag}', combined, epoch, dataformats='NCHW')

def save_checkpoint(path, epoch, mode, model, optimizer, scheduler, train_loss, val_metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {'epoch': epoch, 'mode': mode, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'train_loss': train_loss, 'val_metrics': val_metrics}
    torch.save(checkpoint, path)

def load_model_weights(model, state_dict, strict=False):
    """Load weights, ignoring removed layers (e.g. old skip_conv checkpoints)."""
    model.load_state_dict(state_dict, strict=strict)

def load_checkpoint(path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    load_model_weights(model, checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint

def default_run_name(mode, dataset='tiny_imagenet'):
    if mode == 'recon':
        return 'tiny_imagenet_ae_recon' if dataset == 'tiny_imagenet' else 'conv_ae_recon'
    if mode == 'layer':
        return 'tiny_imagenet_ae_layer_500' if dataset == 'tiny_imagenet' else 'conv_ae_layer'
    if mode == 'layer_adapt':
        return LAYER_ADAPT_RUN_NAME
    if mode == 'layer_adapt2':
        return LAYER_ADAPT2_RUN_NAME
    if mode == 'layer1':
        return LAYER1_RUN_NAME
    raise ValueError(f'Unknown mode: {mode}')

def make_training_stack(model, args, device):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-05)
    milestones = [max(1, args.epochs // 2), max(2, 3 * args.epochs // 4)]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    return (optimizer, scheduler)

def fast_forward_scheduler(scheduler, steps):
    for _ in range(steps):
        scheduler.step()

def load_model_from_checkpoint(path, model, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    load_model_weights(model, checkpoint['model_state_dict'])
    return checkpoint['epoch']

def _max_epoch_from_run(run_name):
    """Return the highest epoch step found in TB logs or checkpoints."""
    max_epoch = 0
    log_dir = os.path.join('runs', run_name)
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        ea = EventAccumulator(log_dir)
        ea.Reload()
        for tag in (f'{TB_GROUP_LOSS}/train', f'{TB_GROUP_LOSS}/train_pixel', f'{TB_GROUP_LOSS}/train_scale', f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER}', f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER}', f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER}', f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER_LATENT}', f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER_LATENT}', f'{TB_GROUP_LOSS}/val_recon_full', f'{TB_GROUP_LOSS}/val_recon', 'Loss/train', 'Loss/val_recon_full'):
            if tag in ea.Tags().get('scalars', []):
                max_epoch = max(max_epoch, max((e.step for e in ea.Scalars(tag))))
            if tag in ea.Tags().get('tensors', []):
                max_epoch = max(max_epoch, max((e.step for e in ea.Tensors(tag))))
    except Exception:
        pass
    ckpt_dir = os.path.join('checkpoints', run_name)
    if os.path.isdir(ckpt_dir):
        for name in os.listdir(ckpt_dir):
            if name.startswith('epoch_') and name.endswith('.pt'):
                max_epoch = max(max_epoch, int(name.replace('epoch_', '').replace('.pt', '')))
    return max_epoch
TB_GROUP_LOSS = '1_Loss'
TB_GROUP_ACCURACY = '2_Accuracy'
TB_GROUP_RECONSTRUCTION = '3_Reconstruction'
TB_GROUP_RECONSTRUCTION_RANDOM = '4_Reconstruction_Random'
TB_GROUP_SCHEDULE = 'Schedule'
TB_VAL_RECON_3LAYER_LATENT = 'val_recon_3layer_latent'
TB_VAL_RECON_3LAYER = 'val_recon_3layer'
TB_VAL_RECON_2LAYER = 'val_recon_2layer'
TB_VAL_RECON_1LAYER = 'val_recon_1layer'
TB_VAL_RECON_2LAYER_LATENT = 'val_recon_2layer_latent'
TB_VAL_RECON_1LAYER_LATENT = 'val_recon_1layer_latent'
TB_VAL_ACC_3LAYER_LATENT = 'val_acc_3layer_latent'
TB_VAL_ACC_3LAYER = 'val_acc_3layer'
TB_VAL_ACC_2LAYER = 'val_acc_2layer'
TB_VAL_ACC_1LAYER = 'val_acc_1layer'
TB_VAL_ACC_2LAYER_LATENT = 'val_acc_2layer_latent'
TB_VAL_ACC_1LAYER_LATENT = 'val_acc_1layer_latent'
RECON_IMAGE_TAG_GROUPS = (TB_GROUP_RECONSTRUCTION, TB_GROUP_RECONSTRUCTION_RANDOM)

def _convae_scalar_rename_map():
    return {f'{TB_GROUP_LOSS}/val_recon_full': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', f'{TB_GROUP_LOSS}/val_recon_trim1': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER}', f'{TB_GROUP_LOSS}/val_recon_trim2': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER}', f'{TB_GROUP_LOSS}/val_recon_trim3': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER}', f'{TB_GROUP_ACCURACY}/val_full': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER_LATENT}', f'{TB_GROUP_ACCURACY}/val_trim1': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER}', f'{TB_GROUP_ACCURACY}/val_trim2': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_2LAYER}', f'{TB_GROUP_ACCURACY}/val_trim3': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER}'}

def _layer_adapt_scalar_rename_map():
    return {f'{TB_GROUP_LOSS}/val_recon_full': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', f'{TB_GROUP_LOSS}/val_recon_trim1': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER_LATENT}', f'{TB_GROUP_LOSS}/val_recon_trim2': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER_LATENT}', f'{TB_GROUP_ACCURACY}/val_full': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER_LATENT}', f'{TB_GROUP_ACCURACY}/val_trim1': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_2LAYER_LATENT}', f'{TB_GROUP_ACCURACY}/val_trim2': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER_LATENT}'}

def _layer1_scalar_rename_map():
    return {f'{TB_GROUP_LOSS}/val_recon': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER_LATENT}', f'{TB_GROUP_ACCURACY}/val_recon': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER_LATENT}'}

def _glue_scalar_rename_map():
    return {f'{TB_GROUP_LOSS}/val_full': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', f'{TB_GROUP_LOSS}/val_bookends': f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER}', f'{TB_GROUP_ACCURACY}/val_full': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER_LATENT}', f'{TB_GROUP_ACCURACY}/val_bookends': f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER}'}

TB_SCALAR_RENAME_BY_RUN = {'tiny_imagenet_ae_recon': _convae_scalar_rename_map(), 'tiny_imagenet_ae_layer_500': _convae_scalar_rename_map(), LAYER1_RUN_NAME: _layer1_scalar_rename_map(), LAYER_ADAPT_RUN_NAME: _layer_adapt_scalar_rename_map(), LAYER_ADAPT_300_RUN_NAME: _layer_adapt_scalar_rename_map(), LAYER_ADAPT2_RUN_NAME: _layer_adapt_scalar_rename_map(), GLUE_LAYER_LATENT_RUN_NAME: _glue_scalar_rename_map(), GLUE_RECON_LATENT_RUN_NAME: _glue_scalar_rename_map()}

def _purge_summary_tags(log_dir, tags_to_drop, exclude_basename=None, exclude_paths=None):
    """Remove summary values with given tags from event files in log_dir."""
    import glob
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.compat.proto import event_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter
    if not os.path.isdir(log_dir):
        return 0
    skip_paths = set(exclude_paths or [])
    total_removed = 0
    for path in sorted(glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))):
        if path in skip_paths:
            continue
        if exclude_basename and os.path.basename(path) == exclude_basename:
            continue
        events_to_keep = []
        removed_in_file = 0
        for event in EventFileLoader(path).Load():
            if event.HasField('summary'):
                kept_values = [v for v in event.summary.value if v.tag not in tags_to_drop]
                removed_in_file += len(event.summary.value) - len(kept_values)
                if not kept_values:
                    continue
                if len(kept_values) < len(event.summary.value):
                    new_event = event_pb2.Event()
                    new_event.CopyFrom(event)
                    del new_event.summary.value[:]
                    new_event.summary.value.extend(kept_values)
                    events_to_keep.append(new_event)
                else:
                    events_to_keep.append(event)
            else:
                events_to_keep.append(event)
        if removed_in_file == 0:
            continue
        tmp_path = path + '.tmp'
        with open(tmp_path, 'wb') as f:
            writer = RecordWriter(f)
            for event in events_to_keep:
                writer.write(event.SerializeToString())
            writer.close()
        os.replace(tmp_path, path)
        total_removed += removed_in_file
    return total_removed

def _expected_summary_data_class(value):
    """Infer correct summary metadata data_class from plugin_name or tag prefix."""
    from tensorboard.compat.proto import summary_pb2
    plugin = value.metadata.plugin_data.plugin_name
    tag = value.tag
    if plugin == 'scalars' or tag.startswith(f'{TB_GROUP_LOSS}/') or tag.startswith(f'{TB_GROUP_ACCURACY}/') or tag.startswith(f'{TB_GROUP_SCHEDULE}/'):
        return summary_pb2.DataClass.DATA_CLASS_SCALAR
    if plugin == 'images' or tag.startswith(f'{TB_GROUP_RECONSTRUCTION}/') or tag.startswith(f'{TB_GROUP_RECONSTRUCTION_RANDOM}/'):
        return summary_pb2.DataClass.DATA_CLASS_BLOB_SEQUENCE
    return None

def _repair_summary_data_class(log_dir, exclude_basename=None, exclude_paths=None, dry_run=False):
    """Fix summary values with missing metadata.data_class (atomic replace per file)."""
    import glob
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.compat.proto import event_pb2, summary_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter
    if not os.path.isdir(log_dir):
        return 0
    skip_paths = set(exclude_paths or [])
    total_repaired = 0
    for path in sorted(glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))):
        if path in skip_paths:
            continue
        if exclude_basename and os.path.basename(path) == exclude_basename:
            continue
        events_to_write = []
        repaired_in_file = 0
        for event in EventFileLoader(path).Load():
            if event.HasField('summary'):
                repaired_any = False
                for value in event.summary.value:
                    if value.metadata.data_class in (0, summary_pb2.DataClass.DATA_CLASS_UNKNOWN):
                        expected = _expected_summary_data_class(value)
                        if expected is not None:
                            repaired_any = True
                if repaired_any:
                    new_event = event_pb2.Event()
                    new_event.CopyFrom(event)
                    for value in new_event.summary.value:
                        if value.metadata.data_class in (0, summary_pb2.DataClass.DATA_CLASS_UNKNOWN):
                            expected = _expected_summary_data_class(value)
                            if expected is not None:
                                value.metadata.data_class = expected
                                repaired_in_file += 1
                    events_to_write.append(new_event)
                else:
                    events_to_write.append(event)
            else:
                events_to_write.append(event)
        if repaired_in_file == 0:
            continue
        if dry_run:
            total_repaired += repaired_in_file
            continue
        tmp_path = path + '.tmp'
        try:
            with open(tmp_path, 'wb') as f:
                writer = RecordWriter(f)
                for event in events_to_write:
                    writer.write(event.SerializeToString())
                writer.close()
            os.replace(tmp_path, path)
            total_repaired += repaired_in_file
        except OSError as exc:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f'WARNING: skipped locked event file {path}: {exc}')
    return total_repaired

def _rename_summary_tags(log_dir, rename_map, exclude_basename=None, exclude_paths=None, dry_run=False):
    """Rename summary scalar tags in event files (atomic replace per file).

    Do not run while training is still writing to the same log_dir — an open
    SummaryWriter will append events with stale metadata after the rewrite.
    """
    import glob
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.compat.proto import event_pb2
    from tensorboard.summary.writer.record_writer import RecordWriter
    if not os.path.isdir(log_dir) or not rename_map:
        return 0
    skip_paths = set(exclude_paths or [])
    total_renamed = 0
    for path in sorted(glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))):
        if path in skip_paths:
            continue
        if exclude_basename and os.path.basename(path) == exclude_basename:
            continue
        events_to_write = []
        renamed_in_file = 0
        for event in EventFileLoader(path).Load():
            if event.HasField('summary'):
                renamed_any = False
                for value in event.summary.value:
                    if value.tag in rename_map:
                        renamed_any = True
                if renamed_any:
                    new_event = event_pb2.Event()
                    new_event.CopyFrom(event)
                    for value in new_event.summary.value:
                        new_tag = rename_map.get(value.tag)
                        if new_tag is not None:
                            value.tag = new_tag
                            renamed_in_file += 1
                    events_to_write.append(new_event)
                else:
                    events_to_write.append(event)
            else:
                events_to_write.append(event)
        if renamed_in_file == 0:
            continue
        if dry_run:
            total_renamed += renamed_in_file
            continue
        tmp_path = path + '.tmp'
        try:
            with open(tmp_path, 'wb') as f:
                writer = RecordWriter(f)
                for event in events_to_write:
                    writer.write(event.SerializeToString())
                writer.close()
            os.replace(tmp_path, path)
            total_renamed += renamed_in_file
        except OSError as exc:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            print(f'WARNING: skipped locked event file {path}: {exc}')
    return total_renamed

def _newest_event_file(log_dir):
    import glob
    paths = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not paths:
        return None

    def _event_timestamp(path):
        parts = os.path.basename(path).split('.')
        if len(parts) >= 4 and parts[3].isdigit():
            return int(parts[3])
        return 0
    return max(paths, key=_event_timestamp)

def _purge_all_reconstruction_tags(log_dir):
    """Remove every reconstruction image summary from a TensorBoard run."""
    import glob
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    if not os.path.isdir(log_dir):
        return 0
    tags_to_drop = set()
    for path in glob.glob(os.path.join(log_dir, 'events.out.tfevents.*')):
        for event in EventFileLoader(path).Load():
            if event.HasField('summary'):
                for value in event.summary.value:
                    if any((value.tag.startswith(f'{group}/') for group in RECON_IMAGE_TAG_GROUPS)):
                        tags_to_drop.add(value.tag)
    if not tags_to_drop:
        return 0
    return _purge_summary_tags(log_dir, tags_to_drop)

def _cleanup_empty_event_files(log_dir, max_bytes=200):
    """Delete header-only TensorBoard event files left after purges."""
    import glob
    removed = 0
    for path in glob.glob(os.path.join(log_dir, 'events.out.tfevents.*')):
        if os.path.getsize(path) <= max_bytes:
            os.remove(path)
            removed += 1
    return removed

def _purge_stale_reconstruction_tags(log_dir, keep_tag):
    """Remove legacy recon tags; keep only the canonical tag for this run."""
    import glob
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    if not os.path.isdir(log_dir):
        return 0
    canonical = f'{TB_GROUP_RECONSTRUCTION}/{keep_tag}'
    keep = {canonical, f'{TB_GROUP_RECONSTRUCTION_RANDOM}/{keep_tag}'}
    tags_to_drop = set()
    for path in glob.glob(os.path.join(log_dir, 'events.out.tfevents.*')):
        for event in EventFileLoader(path).Load():
            if event.HasField('summary'):
                for value in event.summary.value:
                    if any((value.tag.startswith(f'{group}/') for group in RECON_IMAGE_TAG_GROUPS)) and value.tag not in keep:
                        tags_to_drop.add(value.tag)
    if not tags_to_drop:
        return 0
    return _purge_summary_tags(log_dir, tags_to_drop)

def _image_epochs_for_run(run_name, log_images_every=5):
    """Epoch steps to backfill reconstruction images for a run."""
    max_epoch = _max_epoch_from_run(run_name)
    if max_epoch <= 0:
        return []
    ckpt_dir = os.path.join('checkpoints', run_name)
    epochs = []
    for step in range(log_images_every, max_epoch + 1, log_images_every):
        if os.path.isfile(os.path.join(ckpt_dir, f'epoch_{step}.pt')):
            epochs.append(step)
    return epochs

def prune_checkpoints(run_name, interval=20, dry_run=False):
    """Keep epoch checkpoints at every interval and the final epoch; always keep latest.pt."""
    ckpt_dir = os.path.join('checkpoints', run_name)
    if not os.path.isdir(ckpt_dir):
        print(f'[{run_name}] skipped (no checkpoint dir)')
        return {'kept': 0, 'removed': 0}
    max_epoch = 0
    epoch_files = []
    for name in os.listdir(ckpt_dir):
        if name.startswith('epoch_') and name.endswith('.pt'):
            epoch = int(name.replace('epoch_', '').replace('.pt', ''))
            epoch_files.append((epoch, name))
            max_epoch = max(max_epoch, epoch)
    kept = []
    removed = []
    for epoch, name in sorted(epoch_files):
        path = os.path.join(ckpt_dir, name)
        if epoch % interval == 0 or epoch == max_epoch:
            kept.append(name)
        else:
            removed.append(path)
    print(f'[{run_name}] max_epoch={max_epoch} keep={len(kept)} (+latest) remove={len(removed)}')
    if dry_run:
        for path in removed:
            print(f'  would remove: {path}')
    else:
        for path in removed:
            os.remove(path)
            print(f'  removed: {path}')
    return {'kept': len(kept), 'removed': len(removed)}

def run_prune_checkpoints(args):
    """Prune checkpoint files for ACTIVE_RUNS or a single --run-name."""
    runs = [args.run_name] if args.run_name else list(ACTIVE_RUNS)
    total_removed = 0
    for run_name in runs:
        result = prune_checkpoints(run_name, interval=args.checkpoint_interval, dry_run=args.dry_run)
        total_removed += result['removed']
    action = 'would remove' if args.dry_run else 'removed'
    print(f'Done: {action} {total_removed} checkpoint file(s) across {len(runs)} run(s)')

def rewrite_tensorboard_scalars(run_names=None, dry_run=False):
    """Rewrite legacy scalar tags in TensorBoard event files to path-stem names.

    Also repairs summary metadata.data_class after each run. Stop training
    (writer.close()) before running — concurrent SummaryWriter appends corrupt metadata.
    """
    if run_names is None:
        run_names = [run_name for run_name in ACTIVE_RUNS if os.path.isdir(os.path.join('runs', run_name))]
    total_renamed = 0
    total_repaired = 0
    for run_name in run_names:
        log_dir = os.path.join('runs', run_name)
        if not os.path.isdir(log_dir):
            print(f'[{run_name}] skipped (no runs dir)')
            continue
        prefix = 'DRY-RUN ' if dry_run else ''
        rename_map = TB_SCALAR_RENAME_BY_RUN.get(run_name) or {}
        renamed = 0
        if rename_map:
            renamed = _rename_summary_tags(log_dir, rename_map, dry_run=dry_run)
            print(f'[{prefix}{run_name}] renamed {renamed} scalar value(s)')
        else:
            print(f'[{run_name}] no scalar rename map (metadata repair only)')
        repaired = _repair_summary_data_class(log_dir, dry_run=dry_run)
        print(f'[{prefix}{run_name}] repaired {repaired} summary metadata value(s)')
        total_renamed += renamed
        total_repaired += repaired
    rename_action = 'would rename' if dry_run else 'renamed'
    repair_action = 'would repair' if dry_run else 'repaired'
    print(f'Done: {rename_action} {total_renamed} scalar value(s), {repair_action} {total_repaired} metadata value(s) across {len(run_names)} run(s)')

def repair_tensorboard_metadata(run_names=None, dry_run=False):
    """Repair summary metadata.data_class in TensorBoard event files."""
    if run_names is None:
        run_names = [run_name for run_name in ACTIVE_RUNS if os.path.isdir(os.path.join('runs', run_name))]
    total_repaired = 0
    for run_name in run_names:
        log_dir = os.path.join('runs', run_name)
        if not os.path.isdir(log_dir):
            print(f'[{run_name}] skipped (no runs dir)')
            continue
        prefix = 'DRY-RUN ' if dry_run else ''
        repaired = _repair_summary_data_class(log_dir, dry_run=dry_run)
        print(f'[{prefix}{run_name}] repaired {repaired} summary metadata value(s)')
        total_repaired += repaired
    action = 'would repair' if dry_run else 'repaired'
    print(f'Done: {action} {total_repaired} metadata value(s) across {len(run_names)} run(s)')

def run_rewrite_tensorboard_scalars(args):
    """Rewrite legacy TB scalar tags for ACTIVE_RUNS or a single --run-name."""
    runs = [args.run_name] if args.run_name else None
    rewrite_tensorboard_scalars(run_names=runs, dry_run=args.dry_run)

def run_repair_tensorboard_metadata(args):
    """Repair TB summary metadata for ACTIVE_RUNS or a single --run-name."""
    runs = [args.run_name] if args.run_name else None
    repair_tensorboard_metadata(run_names=runs, dry_run=args.dry_run)

def restore_tensorboard_reconstruction(device=None, log_images_every=5):
    """Unique 3_Reconstruction + 4_Reconstruction_Random tags per model."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for run_name in CANONICAL_RECON_RUNS:
        tag = reconstruction_tag_for_run(run_name)
        log_dir = os.path.join('runs', run_name)
        removed = _purge_stale_reconstruction_tags(log_dir, tag)
        print(f'[{run_name}] removed {removed} stale reconstruction events')
    set_seed(42)
    flowers_loader = None
    tiny_loader = None
    glue_latent_runs = (GLUE_LAYER_LATENT_RUN_NAME, GLUE_RECON_LATENT_RUN_NAME)
    for run_name in CANONICAL_RECON_RUNS:
        epochs = _image_epochs_for_run(run_name, log_images_every=log_images_every)
        if not epochs:
            print(f'[{run_name}] skipped (no epochs found)')
            continue
        tag = reconstruction_tag_for_run(run_name)
        log_dir = os.path.join('runs', run_name)
        writer = SummaryWriter(log_dir=log_dir)
        purge_tags = {f'{TB_GROUP_RECONSTRUCTION}/{tag}', f'{TB_GROUP_RECONSTRUCTION_RANDOM}/{tag}'}
        _purge_summary_tags(log_dir, purge_tags)
        print(f'[{run_name}] backfilling {tag} for epochs {epochs[0]}..{epochs[-1]} ({len(epochs)} steps)')
        if run_name in glue_latent_runs:
            if flowers_loader is None:
                set_active_normalization('flowers102')
                _, flowers_loader = get_loaders('flowers102', 64, 0)
            for epoch in epochs:
                ckpt_path = os.path.join('checkpoints', run_name, f'epoch_{epoch}.pt')
                if not os.path.isfile(ckpt_path):
                    continue
                model = GluedAutoencoder(latent_dim=128, bottleneck_size=8).to(device)
                load_model_from_checkpoint(ckpt_path, model, device)
                model.eval()
                log_glue_three_row_reconstruction(model, flowers_loader, writer, device, epoch, tag)
        elif run_name in (LAYER_ADAPT_RUN_NAME, LAYER_ADAPT_300_RUN_NAME, LAYER_ADAPT2_RUN_NAME):
            if tiny_loader is None:
                set_active_normalization('tiny_imagenet')
                _, tiny_loader = get_loaders('tiny_imagenet', 64, 0)
            for epoch in epochs:
                ckpt_path = os.path.join('checkpoints', run_name, f'epoch_{epoch}.pt')
                if not os.path.isfile(ckpt_path):
                    continue
                model = LayerAdaptAutoencoder(latent_dim=128, bottleneck_size=8).to(device)
                load_model_from_checkpoint(ckpt_path, model, device)
                model.eval()
                log_adapt_reconstruction_images(model, tiny_loader, writer, device, epoch, tag)
        else:
            if tiny_loader is None:
                set_active_normalization('tiny_imagenet')
                _, tiny_loader = get_loaders('tiny_imagenet', 64, 0)
            for epoch in epochs:
                ckpt_path = os.path.join('checkpoints', run_name, f'epoch_{epoch}.pt')
                if not os.path.isfile(ckpt_path):
                    continue
                model = ConvAutoencoder(latent_dim=128, bottleneck_size=8).to(device)
                load_model_from_checkpoint(ckpt_path, model, device)
                log_reconstruction_images(model, tiny_loader, writer, device, epoch, tag)
        writer.close()

def format_val_metrics(val_metrics):
    return f"val_full={val_metrics['val_recon_full']:.4f} | trim1={val_metrics['val_recon_trim1']:.4f} | trim2={val_metrics['val_recon_trim2']:.4f} | trim3={val_metrics['val_recon_trim3']:.4f} | acc_full={val_metrics['val_acc_full']:.2f}% | acc_trim1={val_metrics['val_acc_trim1']:.2f}% | acc_trim2={val_metrics['val_acc_trim2']:.2f}% | acc_trim3={val_metrics['val_acc_trim3']:.2f}%"

def format_adapt_val_metrics(val_metrics):
    return f"val_full={val_metrics['val_recon_full']:.4f} | trim1={val_metrics['val_recon_trim1']:.4f} | trim2={val_metrics['val_recon_trim2']:.4f} | acc_full={val_metrics['val_acc_full']:.2f}% | acc_trim1={val_metrics['val_acc_trim1']:.2f}% | acc_trim2={val_metrics['val_acc_trim2']:.2f}%"

def log_accuracy_to_tensorboard(writer, val_metrics, epoch):
    """Log validation pixel accuracy scalars at global epoch step."""
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER_LATENT}', val_metrics['val_acc_full'], epoch)
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER}', val_metrics['val_acc_trim1'], epoch)
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_2LAYER}', val_metrics['val_acc_trim2'], epoch)
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER}', val_metrics['val_acc_trim3'], epoch)

def log_adapt_accuracy_to_tensorboard(writer, val_metrics, epoch):
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_3LAYER_LATENT}', val_metrics['val_acc_full'], epoch)
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_2LAYER_LATENT}', val_metrics['val_acc_trim1'], epoch)
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER_LATENT}', val_metrics['val_acc_trim2'], epoch)

def log_adapt_metrics_to_tensorboard(writer, train_loss, val_metrics, epoch, lr, alpha=None, components=None):
    if components is not None:
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_scale', train_loss, epoch)
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', components['pixel'], epoch)
    else:
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', train_loss, epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', val_metrics['val_recon_full'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER_LATENT}', val_metrics['val_recon_trim1'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER_LATENT}', val_metrics['val_recon_trim2'], epoch)
    log_adapt_accuracy_to_tensorboard(writer, val_metrics, epoch)
    writer.add_scalar(f'{TB_GROUP_SCHEDULE}/LR', lr, epoch)
    if alpha is not None:
        writer.add_scalar(f'{TB_GROUP_SCHEDULE}/Alpha', alpha, epoch)

def log_adapt2_metrics_to_tensorboard(writer, train_loss, val_metrics, epoch, lr, alpha=None, components=None):
    writer.add_scalar(f'{TB_GROUP_LOSS}/train_scale', train_loss, epoch)
    if components is not None:
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', components['pixel'], epoch)
        for key in ADAPT2_COMPONENT_KEYS:
            if key != 'pixel':
                writer.add_scalar(f'{TB_GROUP_LOSS}/train_{key}', components[key], epoch)
    else:
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', train_loss, epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', val_metrics['val_recon_full'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER_LATENT}', val_metrics['val_recon_trim1'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER_LATENT}', val_metrics['val_recon_trim2'], epoch)
    log_adapt_accuracy_to_tensorboard(writer, val_metrics, epoch)
    writer.add_scalar(f'{TB_GROUP_SCHEDULE}/LR', lr, epoch)
    if alpha is not None:
        writer.add_scalar(f'{TB_GROUP_SCHEDULE}/Alpha', alpha, epoch)

def log_layer1_metrics_to_tensorboard(writer, train_loss, val_metrics, epoch, lr):
    writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', train_loss, epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER_LATENT}', val_metrics['val_recon'], epoch)
    writer.add_scalar(f'{TB_GROUP_ACCURACY}/{TB_VAL_ACC_1LAYER_LATENT}', val_metrics['val_acc'], epoch)
    writer.add_scalar(f'{TB_GROUP_SCHEDULE}/LR', lr, epoch)

def log_metrics_to_tensorboard(writer, train_loss, val_metrics, epoch, lr, alpha=None, components=None):
    if components is not None:
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_scale', train_loss, epoch)
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', components['pixel'], epoch)
    else:
        writer.add_scalar(f'{TB_GROUP_LOSS}/train_pixel', train_loss, epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER_LATENT}', val_metrics['val_recon_full'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_3LAYER}', val_metrics['val_recon_trim1'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_2LAYER}', val_metrics['val_recon_trim2'], epoch)
    writer.add_scalar(f'{TB_GROUP_LOSS}/{TB_VAL_RECON_1LAYER}', val_metrics['val_recon_trim3'], epoch)
    log_accuracy_to_tensorboard(writer, val_metrics, epoch)
    writer.add_scalar(f'{TB_GROUP_SCHEDULE}/LR', lr, epoch)
    if alpha is not None:
        writer.add_scalar(f'{TB_GROUP_SCHEDULE}/Alpha', alpha, epoch)

def train_layer_adapt(args):
    """Train LayerAdaptAutoencoder with per-batch stochastic path pixel MSE."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name or default_run_name('layer_adapt')
    log_dir = os.path.join('runs', run_name)
    checkpoint_dir = os.path.join('checkpoints', run_name)
    recon_tag = reconstruction_tag_for_run(run_name)
    print(f'Using device: {device}')
    print(f'Mode: layer_adapt (stochastic path sampling)')
    print(f'Dataset: {args.dataset} (bottleneck_size={args.bottleneck_size})')
    print(f'Run name: {run_name}')
    print(f'Log dir: {log_dir}')
    print(f'Checkpoint dir: {checkpoint_dir}')
    print(ADAPT_TRIM_METRIC_DEFINITIONS.strip())
    print('Path weights per epoch: full=1, trim1=alpha, trim2=alpha^2 (pixel MSE on sampled path)')
    print(describe_alpha_schedule(args))
    print(f'Logging scalars + reconstruction every {args.log_images_every} epochs; checkpoint every {args.checkpoint_every} epochs')
    set_seed(args.seed)
    set_active_normalization(args.dataset)
    trainloader, testloader = get_loaders(args.dataset, args.batch_size, args.num_workers)
    model = LayerAdaptAutoencoder(latent_dim=args.latent_dim, bottleneck_size=args.bottleneck_size).to(device)
    optimizer, scheduler = make_training_stack(model, args, device)
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        ckpt_epoch = checkpoint['epoch']
        if start_epoch == 0:
            start_epoch = ckpt_epoch
        elif start_epoch != ckpt_epoch:
            raise ValueError(f'--start-epoch ({start_epoch}) must match checkpoint epoch ({ckpt_epoch})')
        print(f'Resumed from {args.resume} at epoch {ckpt_epoch}')
    else:
        fast_forward_scheduler(scheduler, start_epoch)
    args.start_epoch = start_epoch
    if start_epoch >= args.epochs:
        raise ValueError(f'--start-epoch ({start_epoch}) must be < --epochs ({args.epochs})')
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, args.epochs):
        alpha = resolve_alpha(epoch, args)
        weights = adapt_path_weights(alpha)
        weight_total = sum(weights)
        train_loss, path_counts = train_one_epoch_layer_adapt_stoch(model, trainloader, optimizer, device, alpha=alpha)
        scheduler.step()
        step = epoch + 1
        path_summary = ' '.join((f'{name}={path_counts[name]}' for name in LAYER_ADAPT_PATH_NAMES))
        weight_summary = ' '.join((f'{name}={w / weight_total:.3f}' if weight_total > 0 else f'{name}=0.000' for name, w in zip(LAYER_ADAPT_PATH_NAMES, weights)))
        if step % args.log_images_every == 0 or step == args.epochs:
            model.eval()
            val_metrics = evaluate_adapt_recon_metrics(model, testloader, device)
            log_adapt_metrics_to_tensorboard(writer, train_loss, val_metrics, step, optimizer.param_groups[0]['lr'], alpha=alpha)
            log_adapt_reconstruction_images(model, testloader, writer, device, step, recon_tag)
            print(f'Epoch {step}/{args.epochs} | train={train_loss:.6f} | {format_adapt_val_metrics(val_metrics)} | alpha={alpha:.4f} | weights({weight_summary}) | paths({path_summary})')
            if step % args.checkpoint_every == 0 or step == args.epochs:
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{step}.pt')
                save_checkpoint(ckpt_path, step, 'layer_adapt', model, optimizer, scheduler, train_loss, val_metrics)
                latest_path = os.path.join(checkpoint_dir, 'latest.pt')
                shutil.copy2(ckpt_path, latest_path)
                print(f'Saved checkpoint: {ckpt_path}')
            model.train()
        else:
            print(f'Epoch {step}/{args.epochs} | train={train_loss:.6f} | alpha={alpha:.4f} | weights({weight_summary}) | paths({path_summary})')
    writer.close()
    print('Layer-adapt training complete.')

def train_layer_adapt2(args):
    """Train LayerAdaptAutoencoder with adapt-style path sampling and cross-bridge cascade loss."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name or default_run_name('layer_adapt2')
    log_dir = os.path.join('runs', run_name)
    checkpoint_dir = os.path.join('checkpoints', run_name)
    recon_tag = reconstruction_tag_for_run(run_name)
    print(f'Using device: {device}')
    print(f'Mode: layer_adapt2 (adapt alpha sampling + cross-bridge cascade)')
    print(f'Dataset: {args.dataset} (bottleneck_size={args.bottleneck_size})')
    print(f'Run name: {run_name}')
    print(f'Log dir: {log_dir}')
    print(f'Checkpoint dir: {checkpoint_dir}')
    print(ADAPT_TRIM_METRIC_DEFINITIONS.strip())
    print(ADAPT2_LOSS_DEFINITIONS.strip())
    print('Path weights per epoch: full=1, trim1=alpha, trim2=alpha^2')
    print(describe_alpha_schedule(args))
    print(f'Logging scalars + reconstruction every {args.log_images_every} epochs; checkpoint every {args.checkpoint_every} epochs')
    if not args.resume:
        for path in (checkpoint_dir, log_dir):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f'Removed existing run data: {path}')
    set_seed(args.seed)
    set_active_normalization(args.dataset)
    trainloader, testloader = get_loaders(args.dataset, args.batch_size, args.num_workers)
    model = LayerAdaptAutoencoder(latent_dim=args.latent_dim, bottleneck_size=args.bottleneck_size).to(device)
    optimizer, scheduler = make_training_stack(model, args, device)
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        ckpt_epoch = checkpoint['epoch']
        if start_epoch == 0:
            start_epoch = ckpt_epoch
        elif start_epoch != ckpt_epoch:
            raise ValueError(f'--start-epoch ({start_epoch}) must match checkpoint epoch ({ckpt_epoch})')
        print(f'Resumed from {args.resume} at epoch {ckpt_epoch}')
    else:
        fast_forward_scheduler(scheduler, start_epoch)
    args.start_epoch = start_epoch
    if start_epoch >= args.epochs:
        raise ValueError(f'--start-epoch ({start_epoch}) must be < --epochs ({args.epochs})')
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, args.epochs):
        alpha = resolve_alpha(epoch, args)
        weights = adapt_path_weights(alpha)
        weight_total = sum(weights)
        train_loss, path_counts, components = train_one_epoch_layer_adapt2_stoch(model, trainloader, optimizer, device, alpha=alpha)
        scheduler.step()
        step = epoch + 1
        path_summary = ' '.join((f'{name}={path_counts[name]}' for name in LAYER_ADAPT_PATH_NAMES))
        weight_summary = ' '.join((f'{name}={w / weight_total:.3f}' if weight_total > 0 else f'{name}=0.000' for name, w in zip(LAYER_ADAPT_PATH_NAMES, weights)))
        if step % args.log_images_every == 0 or step == args.epochs:
            model.eval()
            val_metrics = evaluate_adapt_recon_metrics(model, testloader, device)
            log_adapt2_metrics_to_tensorboard(writer, train_loss, val_metrics, step, optimizer.param_groups[0]['lr'], alpha=alpha, components=components)
            log_adapt_reconstruction_images(model, testloader, writer, device, step, recon_tag)
            print(f'Epoch {step}/{args.epochs} | train={train_loss:.6f} | {format_adapt_val_metrics(val_metrics)} | alpha={alpha:.4f} | weights({weight_summary}) | paths({path_summary})')
            if step % args.checkpoint_every == 0 or step == args.epochs:
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{step}.pt')
                save_checkpoint(ckpt_path, step, 'layer_adapt2', model, optimizer, scheduler, train_loss, val_metrics)
                latest_path = os.path.join(checkpoint_dir, 'latest.pt')
                shutil.copy2(ckpt_path, latest_path)
                print(f'Saved checkpoint: {ckpt_path}')
            model.train()
        else:
            print(f'Epoch {step}/{args.epochs} | train={train_loss:.6f} | alpha={alpha:.4f} | weights({weight_summary}) | paths({path_summary})')
    writer.close()
    print('Layer-adapt2 training complete.')

def train_layer1(args):
    """Train FirstLayerAutoencoder with pixel MSE on the trim2 bridge path."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name or default_run_name('layer1')
    log_dir = os.path.join('runs', run_name)
    checkpoint_dir = os.path.join('checkpoints', run_name)
    recon_tag = reconstruction_tag_for_run(run_name)
    print(f'Using device: {device}')
    print(f'Mode: layer1 (first-layer trim2 bridge)')
    print(f'Dataset: {args.dataset} (bottleneck_size={args.bottleneck_size})')
    print(f'Run name: {run_name}')
    print(f'Log dir: {log_dir}')
    print(f'Checkpoint dir: {checkpoint_dir}')
    print(LAYER1_LOSS_DEFINITION)
    print(f'Logging scalars + reconstruction every {args.log_images_every} epochs; checkpoint every {args.checkpoint_every} epochs')
    set_seed(args.seed)
    set_active_normalization(args.dataset)
    trainloader, testloader = get_loaders(args.dataset, args.batch_size, args.num_workers)
    model = FirstLayerAutoencoder(latent_dim=args.latent_dim, bottleneck_size=args.bottleneck_size).to(device)
    optimizer, scheduler = make_training_stack(model, args, device)
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        ckpt_epoch = checkpoint['epoch']
        if start_epoch == 0:
            start_epoch = ckpt_epoch
        elif start_epoch != ckpt_epoch:
            raise ValueError(f'--start-epoch ({start_epoch}) must match checkpoint epoch ({ckpt_epoch})')
        print(f'Resumed from {args.resume} at epoch {ckpt_epoch}')
    else:
        fast_forward_scheduler(scheduler, start_epoch)
    args.start_epoch = start_epoch
    if start_epoch >= args.epochs:
        raise ValueError(f'--start-epoch ({start_epoch}) must be < --epochs ({args.epochs})')
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, args.epochs):
        step = epoch + 1
        train_loss = train_one_epoch_pixel(model, trainloader, optimizer, device, desc='Train (layer1)')
        val_metrics = evaluate_reconstruction(model, testloader, device)
        scheduler.step()
        log_layer1_metrics_to_tensorboard(writer, train_loss, val_metrics, step, optimizer.param_groups[0]['lr'])
        if step % args.log_images_every == 0:
            log_layer1_reconstruction_images(model, testloader, writer, device, step, recon_tag)
        print(f'Epoch {step}/{args.epochs} | train={train_loss:.6f} | val_mse={val_metrics["val_recon"]:.4f} | val_acc={val_metrics["val_acc"]:.2f}%')
        if step % args.checkpoint_every == 0 or step == args.epochs:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{step}.pt')
            save_checkpoint(ckpt_path, step, 'layer1', model, optimizer, scheduler, train_loss, val_metrics)
            latest_path = os.path.join(checkpoint_dir, 'latest.pt')
            shutil.copy2(ckpt_path, latest_path)
            print(f'Saved checkpoint: {ckpt_path}')
        model.train()
    writer.close()
    print('Layer1 training complete.')

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run_name = args.run_name or default_run_name(args.mode, args.dataset)
    log_dir = os.path.join('runs', run_name)
    checkpoint_dir = os.path.join('checkpoints', run_name)
    print(f'Using device: {device}')
    print(f'Mode: {args.mode}')
    print(f'Dataset: {args.dataset} (bottleneck_size={args.bottleneck_size})')
    print(f'Run name: {run_name}')
    print(f'Log dir: {log_dir}')
    print(f'Checkpoint dir: {checkpoint_dir}')
    if args.mode == 'layer':
        print(describe_alpha_schedule(args))
    set_seed(args.seed)
    set_active_normalization(args.dataset)
    trainloader, testloader = get_loaders(args.dataset, args.batch_size, args.num_workers)
    model = ConvAutoencoder(latent_dim=args.latent_dim, bottleneck_size=args.bottleneck_size).to(device)
    optimizer, scheduler = make_training_stack(model, args, device)
    start_epoch = args.start_epoch
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        ckpt_epoch = checkpoint['epoch']
        if start_epoch == 0:
            start_epoch = ckpt_epoch
        elif start_epoch != ckpt_epoch:
            raise ValueError(f'--start-epoch ({start_epoch}) must match checkpoint epoch ({ckpt_epoch})')
        print(f'Resumed from {args.resume} at epoch {ckpt_epoch}')
    else:
        fast_forward_scheduler(scheduler, start_epoch)
    args.start_epoch = start_epoch
    if start_epoch >= args.epochs:
        raise ValueError(f'--start-epoch ({start_epoch}) must be < --epochs ({args.epochs})')
    writer = SummaryWriter(log_dir=log_dir)
    for epoch in range(start_epoch, args.epochs):
        alpha = resolve_alpha(epoch, args) if args.mode == 'layer' else None
        train_loss, components = train_one_epoch(model, trainloader, optimizer, device, args.mode, alpha=alpha)
        val_metrics = evaluate_all_recon_metrics(model, testloader, device)
        scheduler.step()
        log_metrics_to_tensorboard(writer, train_loss, val_metrics, epoch + 1, optimizer.param_groups[0]['lr'], alpha=alpha, components=components)
        if (epoch + 1) % args.log_images_every == 0:
            log_reconstruction_images(model, testloader, writer, device, epoch + 1, reconstruction_tag_for_run(run_name))
        print(f'Epoch {epoch + 1}/{args.epochs} | train={train_loss:.6f} | {format_val_metrics(val_metrics)}' + (f' | alpha={alpha:.4f}' if args.mode == 'layer' else ''))
        if (epoch + 1) % args.checkpoint_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1}.pt')
            save_checkpoint(ckpt_path, epoch + 1, args.mode, model, optimizer, scheduler, train_loss, val_metrics)
            latest_path = os.path.join(checkpoint_dir, 'latest.pt')
            shutil.copy2(ckpt_path, latest_path)
            print(f'Saved checkpoint: {ckpt_path}')
    writer.close()
    print('Training complete.')

def parse_args():
    parser = argparse.ArgumentParser(description='Train ConvAutoencoder (Tiny ImageNet / CIFAR-10 / Flowers102)')
    parser.add_argument('--mode', choices=['recon', 'layer', 'layer1', 'layer_adapt', 'layer_adapt2', 'train_glue', 'restore_tb_recon', 'prune_checkpoints', 'rewrite_tb_scalars', 'repair_tb_metadata'], default='recon')
    parser.add_argument('--dataset', choices=['cifar10', 'tiny_imagenet', 'flowers102'], default='tiny_imagenet')
    parser.add_argument('--bottleneck-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=0 if sys.platform == 'win32' else 4, help='DataLoader workers (default 0 on Windows for CUDA stability)')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--latent-dim', type=int, default=128)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--recon-run-name', type=str, default=None)
    parser.add_argument('--layer-run-name', type=str, default=None)
    parser.add_argument('--checkpoint-every', type=int, default=5)
    parser.add_argument('--log-images-every', type=int, default=5)
    parser.add_argument('--recon-images', type=int, default=DEFAULT_RECON_IMAGES, help='Images per reconstruction row in TensorBoard (stratified across classes)')
    parser.add_argument('--resume', type=str, default=None, help='Resume single-model training')
    parser.add_argument('--resume-recon', type=str, default=None, help='Resume recon model weights')
    parser.add_argument('--resume-layer', type=str, default=None, help='Resume layer model weights')
    parser.add_argument('--start-epoch', type=int, default=0, help='Epoch index to continue from (0-based)')
    parser.add_argument('--alpha-schedule', choices=['increasing', 'constant'], default='increasing', help='Layer loss alpha schedule (recon unaffected)')
    parser.add_argument('--alpha-value', type=float, default=1.0, help='Fixed alpha when --alpha-schedule constant')
    parser.add_argument('--alpha-start', type=float, default=0.0, help='Alpha at --start-epoch when --alpha-schedule increasing')
    parser.add_argument('--alpha-end', type=float, default=1.0, help='Alpha at final epoch when --alpha-schedule increasing')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dry-run', action='store_true', help='List actions without writing (prune_checkpoints / rewrite_tb_scalars / repair_tb_metadata modes)')
    parser.add_argument('--checkpoint-interval', type=int, default=20, help='Keep every Nth epoch checkpoint plus final (prune_checkpoints mode)')
    return parser.parse_args()
if __name__ == '__main__':
    args = parse_args()
    globals()['RECON_IMAGE_COUNT'] = args.recon_images
    if args.mode == 'restore_tb_recon':
        restore_tensorboard_reconstruction(log_images_every=args.log_images_every)
    elif args.mode == 'prune_checkpoints':
        run_prune_checkpoints(args)
    elif args.mode == 'rewrite_tb_scalars':
        run_rewrite_tensorboard_scalars(args)
    elif args.mode == 'repair_tb_metadata':
        run_repair_tensorboard_metadata(args)
    elif args.mode == 'train_glue':
        train_glue(args)
    elif args.mode == 'layer_adapt':
        train_layer_adapt(args)
    elif args.mode == 'layer_adapt2':
        train_layer_adapt2(args)
    elif args.mode == 'layer1':
        train_layer1(args)
    else:
        train(args)
