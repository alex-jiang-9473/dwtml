import torch
import time
import tqdm
from collections import OrderedDict
from util import get_clamped_psnr
from resource_monitor import ResourceMonitor, print_memory_stats, get_memory_stats, reset_cuda_memory


class Trainer():
    def __init__(self, representation, lr=1e-3, print_freq=1):
        """Model to learn a representation of a single datapoint.

        Args:
            representation (siren.Siren): Neural net representation of image to
                be trained.
            lr (float): Learning rate to be used in Adam optimizer.
            print_freq (int): Frequency with which to print losses.
        """
        self.representation = representation
        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        self.best_vals = {'psnr': 0.0, 'loss': 1e8}
        self.logs = {'psnr': [], 'loss': []}
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())
        self._use_fourier = hasattr(self.representation, 'fourier')
        self._cached_encoding = None
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(device='cuda')

    def train(self, coordinates, features, num_iters, report_memory_growth_only=False):
        """Fit neural net to image with automatic mixed precision.

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
            report_memory_growth_only (bool): If True, print only peak memory usage during training.
        """
        # Start resource monitoring
        self.resource_monitor.start()
        
        if report_memory_growth_only:
            # Reset CUDA memory peaks before training for clean measurement
            reset_cuda_memory()
            torch.cuda.synchronize()
        else:
            print_memory_stats("Memory at training start", device='cuda')
        
        scaler = torch.amp.GradScaler('cuda')
        
        with tqdm.trange(num_iters, ncols=100, dynamic_ncols=True) as t:
            for i in t:
                iter_start_time = time.time()
                
                # Update model with mixed precision
                self.optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    predicted = self.representation(coordinates)
                    loss = self.loss_func(predicted, features)
                
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                # Calculate psnr in FP32 for stability
                with torch.no_grad():
                    psnr = get_clamped_psnr(predicted.float(), features.float())

                # Log iteration time
                iter_time = time.time() - iter_start_time
                self.resource_monitor.log_iteration_time(iter_time)

                # Print results and update logs
                log_dict = {'loss': loss.item(),
                            'psnr': psnr,
                            'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])

                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
        
        # Log final memory stats
        if report_memory_growth_only:
            torch.cuda.synchronize()
            train_mem = get_memory_stats(device='cuda')
            if train_mem is not None:
                print(f"Peak allocated during training: {train_mem['peak_allocated_mb']:.2f} MB")
        else:
            print_memory_stats("Memory at training end", device='cuda')
            self.resource_monitor.print_summary()


    def train_with_fourier(self, coordinates, features, num_iters):
        """Fit neural net to image use fourier encoding as z().

        Args:
            coordinates (torch.Tensor): Tensor of coordinates.
                Shape (num_points, coordinate_dim).
            features (torch.Tensor): Tensor of features. Shape (num_points, feature_dim).
            num_iters (int): Number of iterations to train for.
        """
        # Start resource monitoring
        self.resource_monitor.start()
        print_memory_stats("Memory at fourier training start", device='cuda')
        
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        scaler = torch.amp.GradScaler()

        # Cache Fourier encoding once at the beginning
        with torch.no_grad():
            self._cached_encoding = self.representation.fourier(coordinates)

        with tqdm.trange(num_iters, ncols=100) as t:
            for i in t:
                iter_start_time = time.time()
                
                self.optimizer.zero_grad()

                with torch.autocast(device_type=device_type):
                    predicted = self.representation.siren(self._cached_encoding)
                    loss = self.loss_func(predicted, features)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                
                # Calculate psnr
                psnr = get_clamped_psnr(predicted, features)

                # Log iteration time
                iter_time = time.time() - iter_start_time
                self.resource_monitor.log_iteration_time(iter_time)

                log_dict = {'loss': loss.item(), 'psnr': psnr, 'best_psnr': self.best_vals['psnr']}
                t.set_postfix(**log_dict)
                for key in ['loss', 'psnr']:
                    self.logs[key].append(log_dict[key])
                # Update best values
                if loss.item() < self.best_vals['loss']:
                    self.best_vals['loss'] = loss.item()
                if psnr > self.best_vals['psnr']:
                    self.best_vals['psnr'] = psnr
                    # If model achieves best PSNR seen during training, update model
                    if i > int(num_iters / 2.):
                        for k, v in self.representation.state_dict().items():
                            self.best_model[k].copy_(v)
        
        # Log final memory stats
        print_memory_stats("Memory at fourier training end", device='cuda')
        self.resource_monitor.print_summary()
