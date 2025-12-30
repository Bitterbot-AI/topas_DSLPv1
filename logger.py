"""Logging utilities for DSPL training with TensorBoard and file logging."""

import os
import sys
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np

import torch

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class DSPLLogger:
    """
    Comprehensive logger for DSPL training with support for:
    - Console logging
    - File logging
    - TensorBoard logging
    - Metrics history tracking
    - Research-focused granular metrics
    """

    def __init__(self, output_dir: str, experiment_name: Optional[str] = None,
                 use_tensorboard: bool = True, log_level: int = logging.INFO):
        """
        Initialize the logger.

        Args:
            output_dir: Directory to save logs and checkpoints
            experiment_name: Name for this experiment (defaults to timestamp)
            use_tensorboard: Whether to use TensorBoard logging
            log_level: Python logging level
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(output_dir, "logs", self.experiment_name)
        self.vis_dir = os.path.join(output_dir, "visualizations", self.experiment_name)

        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)

        # Setup Python logger
        self.logger = logging.getLogger(f"DSPL_{self.experiment_name}")
        self.logger.setLevel(log_level)
        self.logger.handlers = []  # Clear existing handlers

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, "training.log")
        )
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        # TensorBoard writer
        self.tb_writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = os.path.join(self.log_dir, "tensorboard")
            self.tb_writer = SummaryWriter(tb_dir)
            self.info(f"TensorBoard logs: {tb_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            # Silent fallback - TensorBoard is optional
            pass

        # Metrics history for plotting - enhanced with more granular metrics
        self.history = {
            'train_loss': {'total': [], 'primary': [], 'deep': [], 'logic': [], 'halt': [], 'comp': []},
            'eval': {
                'accuracy': [],      # Tasks solved (exact match)
                'pixel_acc': [],     # Overall pixel accuracy
                'nonbg_acc': [],     # Non-background pixel accuracy
                'mean_iou': [],      # Mean IoU
                'avg_steps': [],     # Average pondering steps
                'per_class_acc': [], # Per-class accuracy (list of 10 values per epoch)
            },
            'halting_probs': [],
            'learning_rate': [],
            'gradient_norm': [],     # Track gradient norms for stability
            'batch_time': [],        # Samples per second
            'gpu_memory': [],        # GPU memory usage
            'task_progress': {},     # Per-task accuracy over time
        }

        self.global_step = 0
        self.epoch_start_time = None
        self.batch_start_time = None

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def log_config(self, config: Dict[str, Any]):
        """Log and save configuration."""
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        self.info(f"Configuration saved to {config_path}")

        if self.tb_writer:
            self.tb_writer.add_text("config", json.dumps(config, indent=2), 0)

    def log_model_info(self, model: torch.nn.Module):
        """Log model architecture and parameter count."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Save model architecture
        arch_path = os.path.join(self.log_dir, "model_architecture.txt")
        with open(arch_path, 'w') as f:
            f.write(str(model))
            f.write(f"\n\nTotal parameters: {total_params:,}")
            f.write(f"\nTrainable parameters: {trainable_params:,}")

    def start_epoch(self, epoch: int):
        """Mark the start of an epoch for timing."""
        self.epoch_start_time = time.time()

    def start_batch(self):
        """Mark the start of a batch for timing."""
        self.batch_start_time = time.time()

    def log_batch(self, epoch: int, batch_idx: int, total_batches: int,
                  loss: float, loss_dict: Dict[str, float], lr: float,
                  batch_size: int = None):
        """Log training batch metrics."""
        self.global_step += 1

        # Calculate batch time and throughput
        if self.batch_start_time is not None and batch_size is not None:
            batch_time = time.time() - self.batch_start_time
            samples_per_sec = batch_size / batch_time if batch_time > 0 else 0
            self.history['batch_time'].append(samples_per_sec)

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar("train/loss_total", loss, self.global_step)
            for key, value in loss_dict.items():
                self.tb_writer.add_scalar(f"train/loss_{key}", value, self.global_step)
            self.tb_writer.add_scalar("train/learning_rate", lr, self.global_step)

    def log_gradient_norm(self, model: torch.nn.Module, epoch: int):
        """Log gradient norms for debugging training stability."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.history['gradient_norm'].append((epoch, total_norm))

        if self.tb_writer:
            self.tb_writer.add_scalar("train/gradient_norm", total_norm, epoch)

        return total_norm

    def log_gpu_memory(self, epoch: int):
        """Log GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3

            self.history['gpu_memory'].append({
                'epoch': epoch,
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            })

            if self.tb_writer:
                self.tb_writer.add_scalar("system/gpu_memory_allocated_gb", allocated, epoch)
                self.tb_writer.add_scalar("system/gpu_memory_reserved_gb", reserved, epoch)

    def log_epoch(self, epoch: int, train_loss: float, loss_components: Dict[str, float],
                  lr: float):
        """Log end-of-epoch training metrics."""
        self.history['train_loss']['total'].append(train_loss)
        for key in ['primary', 'deep', 'logic', 'halt', 'comp']:
            if key in loss_components:
                self.history['train_loss'][key].append(loss_components[key])
        self.history['learning_rate'].append(lr)

        # Calculate epoch time
        epoch_time = ""
        if self.epoch_start_time is not None:
            elapsed = time.time() - self.epoch_start_time
            epoch_time = f" [{elapsed:.1f}s]"

        self.info(
            f"Epoch {epoch} - Loss: {train_loss:.4f} "
            f"(P:{loss_components.get('primary', 0):.3f} "
            f"D:{loss_components.get('deep', 0):.3f} "
            f"L:{loss_components.get('logic', 0):.3f} "
            f"H:{loss_components.get('halt', 0):.3f} "
            f"C:{loss_components.get('comp', 0):.3f}) "
            f"LR: {lr:.6f}{epoch_time}"
        )

    def log_eval(self, epoch: int, accuracy: float, avg_steps: float,
                 solved: int, total: int):
        """Log basic evaluation metrics (backward compatible)."""
        self.history['eval']['accuracy'].append((epoch, accuracy))
        self.history['eval']['avg_steps'].append((epoch, avg_steps))

        if self.tb_writer:
            self.tb_writer.add_scalar("eval/accuracy", accuracy, epoch)
            self.tb_writer.add_scalar("eval/avg_steps", avg_steps, epoch)
            self.tb_writer.add_scalar("eval/solved", solved, epoch)

        self.info(
            f"[Epoch {epoch}] Eval: {solved}/{total} ({accuracy*100:.2f}%) solved, "
            f"avg steps: {avg_steps:.1f}"
        )

    def log_eval_detailed(self, epoch: int, eval_results: Dict[str, Any]):
        """Log detailed evaluation metrics for research."""
        accuracy = eval_results.get('accuracy', 0)
        pixel_acc = eval_results.get('pixel_acc', 0)
        nonbg_acc = eval_results.get('nonzero_acc', 0)
        mean_iou = eval_results.get('mean_iou', 0)
        avg_steps = eval_results.get('avg_steps', 0)
        solved = eval_results.get('solved', 0)
        total = eval_results.get('total', 0)
        per_class_acc = eval_results.get('per_class_acc', [])
        task_accs = eval_results.get('task_accuracies', {})

        # Store in history
        self.history['eval']['accuracy'].append((epoch, accuracy))
        self.history['eval']['pixel_acc'].append((epoch, pixel_acc))
        self.history['eval']['nonbg_acc'].append((epoch, nonbg_acc))
        self.history['eval']['mean_iou'].append((epoch, mean_iou))
        self.history['eval']['avg_steps'].append((epoch, avg_steps))
        self.history['eval']['per_class_acc'].append((epoch, per_class_acc))

        # Track per-task progress over time
        for task_id, task_acc in task_accs.items():
            if task_id not in self.history['task_progress']:
                self.history['task_progress'][task_id] = []
            self.history['task_progress'][task_id].append((epoch, task_acc))

        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar("eval/accuracy", accuracy, epoch)
            self.tb_writer.add_scalar("eval/pixel_acc", pixel_acc, epoch)
            self.tb_writer.add_scalar("eval/nonbg_acc", nonbg_acc, epoch)
            self.tb_writer.add_scalar("eval/mean_iou", mean_iou, epoch)
            self.tb_writer.add_scalar("eval/avg_steps", avg_steps, epoch)
            self.tb_writer.add_scalar("eval/solved", solved, epoch)

            # Per-class accuracy
            for i, acc in enumerate(per_class_acc):
                self.tb_writer.add_scalar(f"eval/class_{i}_acc", acc, epoch)

        # Console logging
        self.info(
            f"[Epoch {epoch}] Eval: {solved}/{total} solved ({accuracy*100:.2f}%) | "
            f"Pixel Acc: {pixel_acc*100:.2f}% | Non-BG Acc: {nonbg_acc*100:.2f}% | "
            f"mIoU: {mean_iou*100:.2f}%"
        )

    def log_partial_matches(self, epoch: int, partial_matches: Dict[str, int], avg_steps: float):
        """Log partial match statistics."""
        self.info(
            f"[Epoch {epoch}] Partial matches: >90%: {partial_matches.get('>90', 0)}, "
            f">80%: {partial_matches.get('>80', 0)}, >70%: {partial_matches.get('>70', 0)}, "
            f">50%: {partial_matches.get('>50', 0)} | Avg Steps: {avg_steps:.1f}"
        )

        if self.tb_writer:
            for thresh, count in partial_matches.items():
                self.tb_writer.add_scalar(f"eval/partial_{thresh.replace('>', 'gt')}", count, epoch)

    def log_per_class_accuracy(self, epoch: int, per_class_acc: List[float]):
        """Log per-class accuracy breakdown."""
        acc_str = " ".join([f"C{i}:{acc*100:.1f}%" for i, acc in enumerate(per_class_acc)])
        self.info(f"[Epoch {epoch}] Per-class Acc: {acc_str}")

    def log_top_tasks(self, epoch: int, task_accuracies: Dict[int, float], top_n: int = 10):
        """Log top performing tasks."""
        sorted_tasks = sorted(task_accuracies.items(), key=lambda x: x[1], reverse=True)
        top_tasks = sorted_tasks[:top_n]

        top_strs = [f"T{idx}:{acc*100:.1f}%" for idx, acc in top_tasks]
        self.info(f"[Epoch {epoch}] Top {top_n} tasks: {' '.join(top_strs)}")

        if top_tasks:
            best_idx, best_acc = top_tasks[0]
            self.info(f"[Epoch {epoch}] Best task: T{best_idx} at {best_acc*100:.2f}%")

    def log_worst_tasks(self, epoch: int, task_accuracies: Dict[int, float], bottom_n: int = 5):
        """Log worst performing tasks for debugging."""
        sorted_tasks = sorted(task_accuracies.items(), key=lambda x: x[1])
        worst_tasks = sorted_tasks[:bottom_n]

        worst_strs = [f"T{idx}:{acc*100:.1f}%" for idx, acc in worst_tasks]
        self.info(f"[Epoch {epoch}] Worst {bottom_n} tasks: {' '.join(worst_strs)}")

    def log_task_regression(self, epoch: int, task_accuracies: Dict[int, float], threshold: float = 0.1):
        """Log tasks that regressed significantly from their best performance."""
        regressions = []
        for task_id, current_acc in task_accuracies.items():
            if task_id in self.history['task_progress']:
                history = self.history['task_progress'][task_id]
                if history:
                    best_acc = max(acc for _, acc in history)
                    if best_acc - current_acc > threshold:
                        regressions.append((task_id, best_acc, current_acc))

        if regressions:
            reg_strs = [f"T{tid}:{best*100:.1f}%->{curr*100:.1f}%" for tid, best, curr in regressions[:5]]
            self.warning(f"[Epoch {epoch}] Regressions (>{threshold*100:.0f}%): {' '.join(reg_strs)}")

    def log_halting_probs(self, epoch: int, halting_probs: list):
        """Log halting probability distribution for an epoch."""
        probs = [h.mean().item() if isinstance(h, torch.Tensor) else h for h in halting_probs]
        self.history['halting_probs'].append((epoch, probs))

        if self.tb_writer:
            for step, prob in enumerate(probs):
                self.tb_writer.add_scalar(f"halting/step_{step+1}", prob, epoch)

            # Log cumulative halting (what % halted by step N)
            cumsum = np.cumsum(probs)
            for step, cum_prob in enumerate(cumsum):
                self.tb_writer.add_scalar(f"halting/cumulative_step_{step+1}", cum_prob, epoch)

    def log_weight_stats(self, model: torch.nn.Module, epoch: int, layer_names: List[str] = None):
        """Log weight statistics for key layers."""
        if not self.tb_writer:
            return

        for name, param in model.named_parameters():
            if layer_names is None or any(ln in name for ln in layer_names):
                self.tb_writer.add_histogram(f"weights/{name}", param.data, epoch)
                self.tb_writer.add_scalar(f"weights/{name}_mean", param.data.mean(), epoch)
                self.tb_writer.add_scalar(f"weights/{name}_std", param.data.std(), epoch)

    def log_canvas_evolution(self, epoch: int, task_idx: int,
                             intermediate_logits, target_out, test_input,
                             halting_probs):
        """Save visualization of canvas evolution."""
        try:
            from visualization import visualize_canvas_evolution
            save_path = os.path.join(
                self.vis_dir, f"canvas_evolution_epoch{epoch}_task{task_idx}.png"
            )
            visualize_canvas_evolution(
                intermediate_logits, target_out, test_input,
                halting_probs, save_path=save_path
            )
        except Exception as e:
            self.warning(f"Failed to save canvas evolution: {e}")

    def log_task_prediction(self, epoch: int, task_idx: int, model,
                            train_in, train_out, test_in, target_out,
                            demo_mask, device):
        """Save visualization of task prediction."""
        try:
            from visualization import visualize_task_prediction
            save_path = os.path.join(
                self.vis_dir, f"task_pred_epoch{epoch}_task{task_idx}.png"
            )
            visualize_task_prediction(
                model, train_in, train_out, test_in, target_out,
                demo_mask, device, save_path=save_path
            )
        except Exception as e:
            self.warning(f"Failed to save task prediction: {e}")

    def save_training_summary(self):
        """Save training summary plot and metrics."""
        try:
            from visualization import create_training_summary_plot
            save_path = os.path.join(self.vis_dir, "training_summary.png")
            create_training_summary_plot(
                self.history['train_loss'],
                self.history['eval']['accuracy'],
                self.history['eval']['avg_steps'],
                save_path=save_path
            )
            self.info(f"Training summary saved to {save_path}")
        except Exception as e:
            self.warning(f"Failed to save training summary: {e}")

        # Save metrics as JSON
        metrics_path = os.path.join(self.log_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            # Convert numpy types for JSON serialization
            json.dump(self._serialize_history(self.history), f, indent=2)
        self.info(f"Metrics saved to {metrics_path}")

    def _serialize_history(self, obj):
        """Recursively convert numpy types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._serialize_history(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_history(item) for item in obj]
        elif isinstance(obj, tuple):
            return [self._serialize_history(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                        epoch: int, best: bool = False, suffix: str = "",
                        scheduler=None, ema_model=None, extra_state: Dict = None):
        """Save model checkpoint with optional additional state."""
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        state = {
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': self.history
        }

        # Optional: Save scheduler state for proper resume
        if scheduler is not None:
            state['scheduler_state_dict'] = scheduler.state_dict()

        # Optional: Save EMA model weights
        if ema_model is not None:
            state['ema_state_dict'] = ema_model.state_dict() if hasattr(ema_model, 'state_dict') else None

        # Optional: Any extra state (e.g., best metrics)
        if extra_state is not None:
            state['extra'] = extra_state

        if best:
            path = os.path.join(ckpt_dir, f"dspl_model_best{suffix}.pth")
        else:
            path = os.path.join(ckpt_dir, f"dspl_model_epoch{epoch}.pth")

        torch.save(state, path)
        self.info(f"Checkpoint saved: {path}")

    def close(self):
        """Close logger and save final summaries."""
        self.save_training_summary()
        if self.tb_writer:
            self.tb_writer.close()
        self.info("Training complete. Logger closed.")
