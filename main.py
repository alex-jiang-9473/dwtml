import argparse
import getpass
import imageio
import json
import os
import random
import torch
import util
from siren import Siren
from torchvision import transforms
from torchvision.utils import save_image
from training import Trainer
from resource_monitor import ResourceMonitor, reset_cuda_memory, get_memory_stats

# python main.py -iid 2 -lss 40 -nl 10 -ni 1000

parser = argparse.ArgumentParser()
parser.add_argument("-ld", "--logdir", help="Path to save logs", default="./results/siren_main")
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=2000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=2e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-fd", "--full_dataset", help="Whether to use full dataset", action='store_true')
parser.add_argument("-iid", "--image_id", help="Image ID to train on, if not the full dataset", type=int, default=8)
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=28)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=10)
parser.add_argument("-w0", "--w0", help="w0 parameter for SIREN model.", type=float, default=30.0)
parser.add_argument("-w0i", "--w0_initial", help="w0 parameter for first layer of SIREN model.", type=float, default=30.0)

args = parser.parse_args()

# Set up torch and cuda - mixed precision training (requires CUDA)
if not torch.cuda.is_available():
    raise RuntimeError("Mixed precision training requires CUDA. Please run on GPU.")

dtype = torch.float32
device = torch.device('cuda')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Initialize global resource monitor
global_monitor = ResourceMonitor(device='cuda')
global_monitor.start()

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

if args.full_dataset:
    min_id, max_id = 1, 24  # Kodak dataset runs from kodim01.png to kodim24.png
else:
    min_id, max_id = args.image_id, args.image_id

# Dictionary to register mixed precision results
results = {
    'bpp': [],
    'train_psnr': [],
    'img_psnr': [],
    'train_feature_count': [],
    'train_feature_dim': [],
    'total_feature_values': [],
    'train_memory_growth_mb': []
}

# Create directory to store experiments
if not os.path.exists(args.logdir):
    os.makedirs(args.logdir)

# Fit images
for i in range(min_id, max_id + 1):
    print(f'\n{"="*60}')
    print(f'Image {i}')
    print("="*60)
    
    # Reset memory for clean measurements per image
    reset_cuda_memory()

    # Load image
    img = imageio.imread(f"kodak-dataset/kodim{str(i).zfill(2)}.png")
    img = transforms.ToTensor()(img).float().to(device, dtype)

    # Setup model
    func_rep = Siren(
        dim_in=2,
        dim_hidden=args.layer_size,
        dim_out=3,
        num_layers=args.num_layers,
        final_activation=torch.nn.Identity(),
        w0_initial=args.w0_initial,
        w0=args.w0
    ).to(device)

    # Set up training
    trainer = Trainer(func_rep, lr=args.learning_rate)
    coordinates, features = util.to_coordinates_and_features(img)
    coordinates, features = coordinates.to(device, dtype), features.to(device, dtype)

    # Report training data size (same role as coeff count in sparse-band training)
    num_train_features = int(features.shape[0])
    feature_dim = int(features.shape[1])
    total_feature_values = num_train_features * feature_dim
    print(f'Training feature samples: {num_train_features:,} (pixels)')
    print(f'Feature dimension per sample: {feature_dim}')
    print(f'Total feature values: {total_feature_values:,}')

    # Calculate total parameters
    total_params = sum(p.numel() for p in func_rep.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    
    # Calculate model size for FP16. Divide by 8000 to go from bits to kB
    # Model weights stay in FP32, but will be saved as FP16
    model_size = (total_params * 16) / 8000.  # 16 bits per param
    print(f'Model size (FP16): {model_size:.1f}kB')
    bpp = (total_params * 16) / (img.shape[1] * img.shape[2])
    print(f'Mixed precision bpp: {bpp:.2f}')

# Memory usage around training (peak allocated during training)
    # Reset immediately before measurement for stable multi-run comparisons.
    reset_cuda_memory()
    torch.cuda.synchronize()

    # Train model with mixed precision (AMP)
    trainer.train(coordinates, features, num_iters=args.num_iters, report_memory_growth_only=True)
    torch.cuda.synchronize()
    train_mem = get_memory_stats(device='cuda')
    mem_growth = train_mem['peak_allocated_mb'] if train_mem else 0.0
    print(f'Training memory peak usage (allocated): {mem_growth:.2f} MB')

    print(f'Best training psnr: {trainer.best_vals["psnr"]:.2f}')

    # Log results
    results['bpp'].append(bpp)
    results['train_psnr'].append(trainer.best_vals['psnr'])
    results['train_feature_count'].append(num_train_features)
    results['train_feature_dim'].append(feature_dim)
    results['total_feature_values'].append(total_feature_values)
    results['train_memory_growth_mb'].append(mem_growth)

    # Save best model (convert to FP16 for storage)
    best_model_fp16 = {k: v.half() for k, v in trainer.best_model.items()}
    torch.save(best_model_fp16, args.logdir + f'/best_model_{i}.pt')

    # Update current model to be best model
    func_rep.load_state_dict(trainer.best_model)

    # Inference with FP16 for memory efficiency
    func_rep = func_rep.half()
    coordinates = coordinates.half()
    
    with torch.no_grad():
        img_recon = func_rep(coordinates).float().reshape(img.shape[1], img.shape[2], 3).permute(2, 0, 1)
        img_psnr = util.get_clamped_psnr(img_recon, img)
        results['img_psnr'].append(img_psnr)
        print(f'Image PSNR (FP16 inference): {img_psnr:.2f} dB')
        save_image(torch.clamp(img_recon, 0, 1).to('cpu'), args.logdir + f'/reconstruction_{i}.png')

    # Save logs for individual image
    with open(args.logdir + f'/logs{i}.json', 'w') as f:
        json.dump(trainer.logs, f)

    print('\n')

print('Results (Mixed Precision Training):')
print(results)
with open(args.logdir + f'/results.json', 'w') as f:
    json.dump(results, f)

# Compute and save aggregated results
results_mean = {key: util.mean(results[key]) for key in results}
with open(args.logdir + f'/results_mean.json', 'w') as f:
    json.dump(results_mean, f)

print('\nAggregate results:')
print(f'BPP: {results_mean["bpp"]:.2f}')
print(f'Training PSNR: {results_mean["train_psnr"]:.2f} dB')
print(f'Image PSNR (FP16): {results_mean["img_psnr"]:.2f} dB')

# Save global resource summary
global_summary_path = os.path.join(args.logdir, 'global_resource_summary.json')
global_monitor.save_summary(global_summary_path)
