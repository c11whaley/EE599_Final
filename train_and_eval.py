# final_full_script_resnet18_combined_ptq_FIXED.py
import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torch.ao.quantization as quant
from torchvision import datasets, transforms, models
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime
import shutil

# --- FX IMPORTS ---
import torch.fx as fx
from torch.ao.quantization import QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_fx, prepare_qat_fx, convert_fx 
# ----------------------

warnings.filterwarnings("ignore")

# Force non-interactive backend for plotting to prevent crashes/blanks
plt.switch_backend('agg')

# Force CPU for all operations to ensure FX compatibility
device = torch.device("cpu")
print(f"Using device for all operations: {device}")

# Directories - relative to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "experiment_results_cifar10_resnet18")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# Debug: print script directory on startup
print(f"Script directory: {SCRIPT_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"Checkpoint directory: {CHECKPOINT_DIR}")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
EPOCHS_BASELINE = 25
EPOCHS_PRUNE = 25
EPOCHS_PTQ = 25
EPOCHS_QAT = 25
LEARNING_RATE = 0.001
BATCH_SIZE = 128

# Pruning configurations: (name_suffix, sparsity_amount)
PRUNING_CONFIGS = [
    ('_25pct', 0.25),
    ('_50pct', 0.50),
    ('_75pct', 0.75),
]

# Default pruning amounts (kept for backward compatibility)
STRUCT_PRUNE_AMT = 0.25
GLOBAL_PRUNE_AMT = 0.25

# CIFAR-10 data loaders
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])

# CIFAR-10 data path relative to script
CIFAR_DATA_DIR = os.path.join(SCRIPT_DIR, "data", "cifar10")
os.makedirs(CIFAR_DATA_DIR, exist_ok=True)

trainset = datasets.CIFAR10(CIFAR_DATA_DIR, train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(CIFAR_DATA_DIR, train=False, download=True, transform=transform_test)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ResNet-18 builder
def create_resnet18(num_classes=10):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.fuse_model = lambda: None
    return model

# Checkpoint utilities
def save_checkpoint(model, optimizer, epoch, test_acc, tag, training_complete=False):
    model_cpu = model.to(torch.device("cpu"))
    state = {
        'epoch': epoch,
        'model_state_dict': model_cpu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': test_acc,
        'tag': tag,
        'training_complete': training_complete
    }
    filepath = os.path.join(CHECKPOINT_DIR, f'{tag}_best.pth.tar')
    torch.save(state, filepath)
    print(f"  [Checkpoint] Saved checkpoint for {tag} -> {filepath}")

def load_checkpoint(model, optimizer, tag):
    filepath = os.path.join(CHECKPOINT_DIR, f'{tag}_best.pth.tar')
    if not os.path.exists(filepath):
        print(f"  [Checkpoint] No checkpoint found for {tag}.")
        print(f"             Expected at: {filepath}")
        # List available checkpoints for debugging
        if os.path.exists(CHECKPOINT_DIR):
            files = os.listdir(CHECKPOINT_DIR)
            if files:
                print(f"             Available checkpoints: {files}")
            else:
                print(f"             Checkpoint directory is empty.")
        else:
            print(f"             Checkpoint directory does not exist: {CHECKPOINT_DIR}")
        return 0, 0.0
    try:
        checkpoint = torch.load(filepath, map_location=device)
    except Exception as e:
        print(f"  [Checkpoint] Error loading checkpoint file: {e}")
        return 0, 0.0
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    except RuntimeError as e:
        print(f"  [Checkpoint] Warning: Strict load failed ({e}). Trying strict=False.")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    start_epoch = checkpoint['epoch']
    best_test_acc = checkpoint['best_test_acc']
    print(f"  [Checkpoint] Loaded checkpoint for {tag} from epoch {start_epoch} with best test accuracy: {best_test_acc:.2f}%")
    return start_epoch, best_test_acc

def checkpoint_exists(tag):
    filepath = os.path.join(CHECKPOINT_DIR, f'{tag}_best.pth.tar')
    return os.path.exists(filepath)

def is_training_complete(tag, target_epochs):
    """
    Check if a model has already completed training for the target number of epochs.
    Returns True if training is complete, False otherwise.
    Uses a 'training_complete' flag in the checkpoint for reliable detection.
    """
    filepath = os.path.join(CHECKPOINT_DIR, f'{tag}_best.pth.tar')
    if not os.path.exists(filepath):
        return False
    
    try:
        checkpoint = torch.load(filepath, map_location=device)
        
        # Check for training_complete flag (new method)
        if checkpoint.get('training_complete', False):
            print(f"  [Training Status] {tag} marked as complete. Skipping.")
            return True
        
        # Fallback to epoch-based check for backwards compatibility
        completed_epoch = checkpoint.get('epoch', 0)
        if completed_epoch >= target_epochs:
            print(f"  [Training Status] {tag} already trained for {completed_epoch} epochs (target: {target_epochs}). Skipping.")
            return True
        else:
            print(f"  [Training Status] {tag} partially trained ({completed_epoch}/{target_epochs} epochs). Resuming from epoch {completed_epoch + 1}.")
            return False
    except Exception as e:
        print(f"  [Training Status] Error checking checkpoint for {tag}: {e}")
        return False

def get_checkpoint_tag(base_name, suffix=''):
    """Generate checkpoint tag with optional suffix."""
    return f"{base_name}{suffix}" if suffix else base_name

# Training / evaluation utilities
def train(model, loader, criterion, optimizer, epochs, tag="model", load_if_exists=True):
    model.to(device)
    history = {'epoch': [], 'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}
    start_epoch = 0
    best_test_acc = 0.0
    
    if load_if_exists:
        start_epoch, best_test_acc = load_checkpoint(model, optimizer, tag)
        
    print(f"[{tag}] Starting training from epoch {start_epoch + 1}/{epochs}.")
    
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        
        epoch_train_loss = train_loss / total
        epoch_train_acc = correct / total * 100.0
        
        test_loss, test_acc = evaluate(model, testloader, criterion)
        
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_checkpoint(model, optimizer, epoch + 1, best_test_acc, tag, training_complete=False)
        
        print(f"[{tag}] Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
    
    # Mark training as complete after all epochs finish
    if best_test_acc > 0.0:
        print(f"[{tag}] Training complete! Loading best saved model with accuracy {best_test_acc:.2f}% for final use...")
        dummy_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        load_checkpoint(model, dummy_optimizer, tag)
        # Save completion marker
        save_checkpoint(model, dummy_optimizer, epochs, best_test_acc, tag, training_complete=True)
        
    return model, history

def evaluate(model, loader, criterion):
    eval_device = torch.device("cpu")
    model = model.to(eval_device)
    model.eval()
    correct = 0
    total_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(eval_device), y.to(eval_device)
            out = model(x)
            total_loss += criterion(out, y).item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
            
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total * 100.0 if total > 0 else 0.0
    return avg_loss, accuracy

def model_size(model, model_name, debug=False):
    """
    Calculate model size in MB.
    For JIT scripted models, uses state_dict serialization.
    For regular models, counts parameters and buffers.
    """
    import io
    
    # Check if model is JIT scripted
    is_jit = isinstance(model, torch.jit.ScriptModule)
    is_quantized = getattr(model, 'is_quantized_model', False)
    
    if debug:
        print(f"    [MODEL_SIZE] Model type: {type(model).__name__}")
        print(f"    [MODEL_SIZE] Is JIT scripted: {is_jit}")
        print(f"    [MODEL_SIZE] Is quantized: {is_quantized}")
    
    # For JIT models or quantized models, always use state_dict method
    if is_jit or is_quantized:
        if debug:
            print(f"    [MODEL_SIZE] Using state_dict serialization method...")
        try:
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            total_bytes = buffer.tell()
            size_mb = total_bytes / 1e6
            if debug:
                print(f"    [MODEL_SIZE] State dict size: {total_bytes:,} bytes = {size_mb:.4f} MB")
            return size_mb
        except Exception as e:
            if debug:
                print(f"    [MODEL_SIZE] State dict method failed: {e}")
                print(f"    [MODEL_SIZE] Trying torch.jit.save method...")
            # Fallback: save entire JIT model
            try:
                buffer = io.BytesIO()
                torch.jit.save(model, buffer)
                total_bytes = buffer.tell()
                size_mb = total_bytes / 1e6
                if debug:
                    print(f"    [MODEL_SIZE] JIT model size: {total_bytes:,} bytes = {size_mb:.4f} MB")
                return size_mb
            except Exception as e2:
                print(f"    [MODEL_SIZE] Error: {e2}")
                return 0.0
    
    # For regular models, count parameters and buffers
    def count_bytes(mod, prefix=""):
        total_bytes = 0
        param_names = set(name for name, _ in mod.named_parameters(recurse=False))

        for name, param in mod.named_parameters(recurse=False):
            try:
                param_bytes = param.numel() * param.element_size()
            except Exception:
                param_bytes = param.numel() * 4
            total_bytes += param_bytes
            if debug:
                es = param.element_size() if hasattr(param, 'element_size') else 4
                print(f"      {prefix}{name}: {param.numel():,} params × {es} bytes = {param_bytes:,} bytes")

        for name, buf in mod.named_buffers(recurse=False):
            if 'mask' in name:
                continue
            if name in param_names:
                continue
            if isinstance(buf, torch.Tensor):
                try:
                    buf_bytes = buf.numel() * buf.element_size()
                except Exception:
                    dtype = getattr(buf, 'dtype', None)
                    if dtype in (torch.qint8, torch.quint8, torch.int8, torch.uint8):
                        elem_size = 1
                    elif dtype in (torch.qint32, torch.int32, torch.float32):
                        elem_size = 4
                    else:
                        elem_size = 4
                    buf_bytes = buf.numel() * elem_size
                total_bytes += buf_bytes
                if debug:
                    try:
                        es = buf.element_size()
                    except:
                        es = elem_size
                    print(f"      {prefix}{name} (buffer): {buf.numel():,} × {es} bytes = {buf_bytes:,} bytes")

        for subname, submod in mod.named_children():
            total_bytes += count_bytes(submod, prefix=prefix + subname + ".")

        return total_bytes
    
    try:
        total_bytes = count_bytes(model, prefix="")
        size_mb = total_bytes / 1e6
        if debug:
            print(f"    [MODEL_SIZE] Total: {total_bytes:,} bytes = {size_mb:.4f} MB")
        return size_mb
    except Exception as e:
        print(f"Error calculating model size: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def param_count(model, debug=False):
    """
    Count total number of parameters in the model.
    For JIT models, uses state_dict.
    For FX GraphModule, uses state_dict.
    """
    # Check if JIT scripted
    is_jit = isinstance(model, torch.jit.ScriptModule)
    
    # Check if FX GraphModule
    is_fx_graph = type(model).__name__ == 'GraphModule'
    
    if debug:
        print(f"    [PARAM_COUNT] Model type: {type(model).__name__}")
        print(f"    [PARAM_COUNT] Is JIT: {is_jit}, Is FX GraphModule: {is_fx_graph}")
    
    if is_jit or is_fx_graph:
        if debug:
            print(f"    [PARAM_COUNT] Using state_dict to count parameters")
        try:
            state_dict = model.state_dict()
            if debug:
                print(f"    [PARAM_COUNT] State dict has {len(state_dict)} entries")
            
            # Filter for only torch.Tensor objects (skip scales, zero_points, dtypes, etc.)
            tensor_params = {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            
            if debug:
                print(f"    [PARAM_COUNT] Found {len(tensor_params)} tensor entries")
                # Show first few entries for debugging
                for i, (name, tensor) in enumerate(list(tensor_params.items())[:5]):
                    print(f"      {name}: {tensor.numel():,} elements")
                if len(tensor_params) > 5:
                    print(f"      ... and {len(tensor_params)-5} more tensor entries")
            
            total = sum(p.numel() for p in tensor_params.values())
            if debug:
                print(f"    [PARAM_COUNT] Total: {total:,}")
            return total
        except Exception as e:
            if debug:
                print(f"    [PARAM_COUNT] Error with state_dict: {e}")
                import traceback
                traceback.print_exc()
            return 0
    
    # Regular model counting
    def count_params(mod, prefix=""):
        total = 0
        param_names = set(name for name, _ in mod.named_parameters(recurse=False))

        for name, param in mod.named_parameters(recurse=False):
            num_params = param.numel()
            total += num_params
            if debug:
                print(f"      {prefix}{name}: {num_params:,}")

        for name, buf in mod.named_buffers(recurse=False):
            if 'mask' in name:
                continue
            if name in param_names:
                continue
            if isinstance(buf, torch.Tensor):
                buf_params = buf.numel()
                total += buf_params
                if debug:
                    print(f"      {prefix}{name} (buffer): {buf_params:,}")

        for subname, submod in mod.named_children():
            sub_count = count_params(submod, prefix=prefix + subname + ".")
            total += sub_count

        return total
    
    total = count_params(model)
    if debug:
        print(f"    [PARAM_COUNT] Total: {total:,}")
    return total


def count_nonzero_params(model, debug=False):
    """
    Count non-zero parameters.
    For JIT models and FX GraphModule, assumes all parameters are non-zero (can't access masks).
    """
    is_jit = isinstance(model, torch.jit.ScriptModule)
    is_fx_graph = type(model).__name__ == 'GraphModule'
    
    if debug:
        print(f"    [NONZERO_COUNT] Model type: {type(model).__name__}")
        print(f"    [NONZERO_COUNT] Is JIT: {is_jit}, Is FX GraphModule: {is_fx_graph}")
    
    if is_jit or is_fx_graph:
        if debug:
            print(f"    [NONZERO_COUNT] JIT/FX model - counting all params as nonzero")
        total = param_count(model, debug=False)
        if debug:
            print(f"    [NONZERO_COUNT] Final: nonzero={total:,}, total={total:,}")
        return total, total
    
    # Regular model counting
    def count_nonzero(mod, prefix=""):
        nonzero = 0
        total = 0
        param_names = set(name for name, _ in mod.named_parameters(recurse=False))

        for name, param in mod.named_parameters(recurse=False):
            param_total = param.numel()
            total += param_total

            if name.endswith('_orig'):
                mask_name = name.replace('_orig', '_mask')
                if hasattr(mod, mask_name):
                    mask = getattr(mod, mask_name)
                    try:
                        masked = param * mask
                        param_nonzero = int(torch.count_nonzero(masked).item())
                    except Exception:
                        param_nonzero = int(torch.count_nonzero(param).item())
                else:
                    param_nonzero = int(torch.count_nonzero(param).item())
            else:
                param_nonzero = int(torch.count_nonzero(param).item())

            nonzero += param_nonzero
            if debug:
                print(f"      {prefix}{name}: {param_nonzero:,} / {param_total:,}")

        for name, buf in mod.named_buffers(recurse=False):
            if 'mask' in name:
                continue
            if name in param_names:
                continue
            if isinstance(buf, torch.Tensor):
                buf_total = buf.numel()
                total += buf_total
                try:
                    buf_nonzero = int(torch.count_nonzero(buf).item())
                except Exception:
                    try:
                        buf_nonzero = int(torch.count_nonzero(buf.int_repr()).item())
                    except Exception:
                        buf_nonzero = buf_total
                nonzero += buf_nonzero
                if debug:
                    print(f"      {prefix}{name} (buffer): {buf_nonzero:,} / {buf_total:,}")

        for subname, submod in mod.named_children():
            sub_nonzero, sub_total = count_nonzero(submod, prefix=prefix + subname + ".")
            nonzero += sub_nonzero
            total += sub_total

        return nonzero, total
    
    nonzero, total = count_nonzero(model)
    if debug:
        print(f"    [NONZERO_COUNT] Final: nonzero={nonzero:,}, total={total:,}")
    return nonzero, total

# ---------------------------------------------------------

def measure_latency(model, num_samples=1000):
    """
    Measure inference latency for single image predictions.
    
    NOTE: Pruned models with masks will likely be SLOWER than baseline because:
    - PyTorch pruning uses masking (weight = weight_orig * mask)
    - Full tensor is still stored and computed
    - Mask multiplication adds overhead
    
    To see actual speedup, you need:
    - Sparse tensor formats
    - Specialized sparse kernels
    - Hardware with sparse computation support
    """
    eval_device = torch.device("cpu")
    model_for_eval = model.to(eval_device)
    model_for_eval.eval()
    
    dummy_input = torch.randn(1, 3, 32, 32).to(eval_device)
    
    # Warmup runs
    warmup_runs = 100
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model_for_eval(dummy_input)
    
    # Actual measurement
    times = []
    with torch.no_grad():
        for i in range(num_samples):
            start = time.perf_counter()
            _ = model_for_eval(dummy_input)
            end = time.perf_counter()
            times.append(end - start)
    
    # Return average in milliseconds
    avg_time = sum(times) / len(times) * 1000
    return avg_time

# Pruning utilities
def apply_structured_pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            try:
                prune.ln_structured(module, 'weight', amount=amount, n=2, dim=0)
            except Exception: pass
    return model

def apply_global_pruning(model, amount=0.2):
    parameters_to_prune = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    try:
        prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=amount)
    except Exception: pass
    return model

# FX quant utilities
def get_fx_quant_config():
    qconfig = quant.get_default_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    return qconfig_mapping

def apply_static_quantization(model, trainloader):
    print("  DEBUG: Starting FX Static Quantization...")
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    model_on_cpu = model.to(torch.device("cpu"))
    try:
        graph_module = fx.symbolic_trace(model_on_cpu)
    except Exception as e:
        print(f"  ERROR: Symbolic trace failed. Quantization cannot proceed. {e}")
        return model_on_cpu
    qconfig_mapping = get_fx_quant_config()
    prepared_model = prepare_fx(graph_module, qconfig_mapping, dummy_input)
    print("  DEBUG: Running calibration...")
    with torch.no_grad():
        for i, (x, _) in enumerate(trainloader):
            if i >= 10: break
            prepared_model(x.to(torch.device("cpu")))
    print("  DEBUG: Converting to quantized model...")
    quantized_model = convert_fx(prepared_model)
    quantized_model.is_quantized_model = True
    print(f"  DEBUG: Quantized model created")
    return quantized_model

def apply_qat(model, trainloader, criterion, optimizer, epochs, load_if_exists=True):
    print("  DEBUG: Starting FX QAT...")
    dummy_input = torch.randn(1, 3, 32, 32)
    model_on_cpu = model.to(torch.device("cpu"))
    try:
        graph_module = fx.symbolic_trace(model_on_cpu)
    except Exception as e:
        print(f"  ERROR: Symbolic trace failed. {e}")
        return model_on_cpu, None
    qat_qconfig = quant.get_default_qat_qconfig("fbgemm")
    qconfig_mapping = QConfigMapping().set_global(qat_qconfig)
    print(f"  DEBUG: Preparing model for QAT...")
    prepared_model = prepare_qat_fx(graph_module, qconfig_mapping, dummy_input)
    print(f"  DEBUG: Training model with QAT on CPU...")
    prepared_model, history = train(prepared_model, trainloader, criterion, optimizer, epochs, tag="qat", load_if_exists=load_if_exists)
    prepared_model.eval()
    print(f"  DEBUG: Converting QAT model...")
    quantized_model = convert_fx(prepared_model)
    quantized_model.is_quantized_model = True
    print(f"  DEBUG: QAT model created")
    return quantized_model, history

# Plotting
def plot_training_history(history, model_name, save_dir):
    if history is None or not history.get('epoch'):
        print(f"Skipping plot for {model_name}: No training history.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training History - {model_name}', fontsize=16, fontweight='bold')
    epochs = history['epoch']
    ax1 = axes[0]
    ax1.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy')
    ax1.plot(epochs, history['test_acc'], 'r-s', label='Test Accuracy')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Accuracy (%)'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2 = axes[1]
    ax2.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax2.plot(epochs, history['test_loss'], 'r-s', label='Test Loss')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_history.png'))
    plt.close()

def plot_all_models_comparison(all_histories, save_dir):
    histories_to_plot = {k: v for k, v in all_histories.items() if v is not None and v.get('epoch')}
    if not histories_to_plot: return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = plt.cm.tab10(range(len(histories_to_plot)))
    ax1 = axes[0]; ax2 = axes[1]
    for i, (model_name, history) in enumerate(histories_to_plot.items()):
        ax1.plot(history['epoch'], history['test_acc'], marker='o', label=model_name, color=colors[i])
        ax2.plot(history['epoch'], history['test_loss'], marker='o', label=model_name, color=colors[i])
    ax1.set_title('Test Accuracy'); ax1.legend(); ax1.grid(True)
    ax2.set_title('Test Loss'); ax2.legend(); ax2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'all_models_training_comparison.png'))
    plt.close()

def plot_comparison(df, save_dir):
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    models_list = df['model'].tolist()
    colors = plt.cm.Set3(range(len(models_list)))
    
    def add_labels(ax, bars, fmt):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, fmt.format(height), ha='center', va='bottom', fontsize=9)

    ax1 = axes[0,0]; bars1 = ax1.bar(models_list, df['test_accuracy'], color=colors, edgecolor='k')
    ax1.set_title('Test Accuracy (%)'); ax1.set_ylim([0, 105]); ax1.tick_params(axis='x', rotation=45)
    add_labels(ax1, bars1, "{:.2f}")

    ax2 = axes[0,1]; bars2 = ax2.bar(models_list, df['size_MB'], color=colors, edgecolor='k')
    ax2.set_title('Model Size (MB)'); ax2.tick_params(axis='x', rotation=45)
    add_labels(ax2, bars2, "{:.3f}")

    ax3 = axes[1,0]; bars3 = ax3.bar(models_list, df['latency_ms_per_image'], color=colors, edgecolor='k')
    ax3.set_title('Latency (ms/image)'); ax3.tick_params(axis='x', rotation=45)
    add_labels(ax3, bars3, "{:.3f}")

    ax4 = axes[1,1]; bars4 = ax4.bar(models_list, df['params']/1e6, color=colors, edgecolor='k')
    ax4.set_title('Parameters (Millions)'); ax4.tick_params(axis='x', rotation=45)
    add_labels(ax4, bars4, "{:.3f}M")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'comparison_plots.png'))
    plt.close()

def plot_efficiency_metrics(df, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]; ax1.scatter(df['size_MB'], df['test_accuracy'], s=200, c=range(len(df)), cmap='viridis', edgecolor='k')
    for i, m in enumerate(df['model']):
        ax1.annotate(m, (df['size_MB'].iloc[i], df['test_accuracy'].iloc[i]), xytext=(5,5), textcoords='offset points')
    ax1.set_xlabel('Size (MB)'); ax1.set_ylabel('Accuracy (%)'); ax1.grid(True)

    ax2 = axes[1]; ax2.scatter(df['latency_ms_per_image'], df['test_accuracy'], s=200, c=range(len(df)), cmap='viridis', edgecolor='k')
    for i, m in enumerate(df['model']):
        ax2.annotate(m, (df['latency_ms_per_image'].iloc[i], df['test_accuracy'].iloc[i]), xytext=(5,5), textcoords='offset points')
    ax2.set_xlabel('Latency (ms)'); ax2.set_ylabel('Accuracy (%)'); ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'efficiency_plots.png'))
    plt.close()

# Main run
def run_experiments():
    criterion = nn.CrossEntropyLoss()
    all_histories = {}
    all_records = []

    # 1) Baseline
    print("\n" + "="*80 + "\n=== 1. Baseline (FP32) ===\n" + "="*80)
    if is_training_complete("baseline", EPOCHS_BASELINE):
        baseline = create_resnet18()
        load_checkpoint(baseline, optim.Adam(baseline.parameters(), lr=LEARNING_RATE), "baseline")
        try: baseline_history = pd.read_csv(os.path.join(METRICS_DIR, 'baseline_metrics.csv')).to_dict(orient='list')
        except: baseline_history = None
    else:
        baseline = create_resnet18()
        optimizer = optim.Adam(baseline.parameters(), lr=LEARNING_RATE)
        baseline, baseline_history = train(baseline, trainloader, criterion, optimizer, EPOCHS_BASELINE, tag="baseline", load_if_exists=True)
        if baseline_history:
            pd.DataFrame(baseline_history).to_csv(os.path.join(METRICS_DIR, 'baseline_metrics.csv'), index=False)
    all_histories['baseline'] = baseline_history
    if baseline_history:
        plot_training_history(baseline_history, 'baseline', PLOTS_DIR)

    # 2) Structured Pruning - Multiple Sparsity Levels
    print("\n" + "="*80 + "\n=== 2. Structured Pruning ===\n" + "="*80)
    structured_models = {}
    for suffix, sparsity in PRUNING_CONFIGS:
        tag = get_checkpoint_tag("structured_pruned", suffix)
        print(f"\n--- Structured Pruning {sparsity*100:.0f}% ---")
        
        if is_training_complete(tag, EPOCHS_PRUNE):
            s_model = create_resnet18()
            s_model = apply_structured_pruning(s_model, sparsity)
            load_checkpoint(s_model, optim.Adam(s_model.parameters(), lr=LEARNING_RATE), tag)
            try: s_history = pd.read_csv(os.path.join(METRICS_DIR, f'structured_pruned{suffix}_metrics.csv')).to_dict(orient='list')
            except: s_history = None
        else:
            s_model = create_resnet18()
            s_optimizer = optim.Adam(s_model.parameters(), lr=LEARNING_RATE)
            _, _ = load_checkpoint(s_model, s_optimizer, "baseline")
            s_model = apply_structured_pruning(s_model, sparsity)
            s_model, s_history = train(s_model, trainloader, criterion, s_optimizer, EPOCHS_PRUNE, tag=tag, load_if_exists=True)
            if s_history:
                pd.DataFrame(s_history).to_csv(os.path.join(METRICS_DIR, f'structured_pruned{suffix}_metrics.csv'), index=False)
        
        all_histories[tag] = s_history
        structured_models[tag] = (s_model, sparsity)
        if s_history:
            plot_training_history(s_history, tag, PLOTS_DIR)

    # 3) Global Pruning - Multiple Sparsity Levels
    print("\n" + "="*80 + "\n=== 3. Global Pruning ===\n" + "="*80)
    global_models = {}
    for suffix, sparsity in PRUNING_CONFIGS:
        tag = get_checkpoint_tag("global_pruned", suffix)
        print(f"\n--- Global Pruning {sparsity*100:.0f}% ---")
        
        if is_training_complete(tag, EPOCHS_PRUNE):
            g_model = create_resnet18()
            g_model = apply_global_pruning(g_model, sparsity)
            load_checkpoint(g_model, optim.Adam(g_model.parameters(), lr=LEARNING_RATE), tag)
            try: g_history = pd.read_csv(os.path.join(METRICS_DIR, f'global_pruned{suffix}_metrics.csv')).to_dict(orient='list')
            except: g_history = None
        else:
            g_model = create_resnet18()
            g_optimizer = optim.Adam(g_model.parameters(), lr=LEARNING_RATE)
            _, _ = load_checkpoint(g_model, g_optimizer, "baseline")
            g_model = apply_global_pruning(g_model, sparsity)
            g_model, g_history = train(g_model, trainloader, criterion, g_optimizer, EPOCHS_PRUNE, tag=tag, load_if_exists=True)
            if g_history:
                pd.DataFrame(g_history).to_csv(os.path.join(METRICS_DIR, f'global_pruned{suffix}_metrics.csv'), index=False)
        
        all_histories[tag] = g_history
        global_models[tag] = (g_model, sparsity)
        if g_history:
            plot_training_history(g_history, tag, PLOTS_DIR)

    # 4) PTQ
    print("\n" + "="*80 + "\n=== 4. PTQ (FX Graph Mode) ===\n" + "="*80)
    ptq_scripted_path = os.path.join(MODELS_DIR, 'ptq_scripted.pt')
    
    if os.path.exists(ptq_scripted_path):
        print("  [PTQ] Loading pre-quantized PTQ model.")
        # For metrics, we need the pre-JIT model, so recreate it
        ptq_model = create_resnet18()
        ptq_opt = optim.Adam(ptq_model.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(ptq_model, ptq_opt, "baseline")
        ptq_model = apply_static_quantization(ptq_model, trainloader)
    else:
        print("  [PTQ] Creating and quantizing baseline model...")
        ptq_model = create_resnet18()
        ptq_opt = optim.Adam(ptq_model.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(ptq_model, ptq_opt, "baseline")
        ptq_model = apply_static_quantization(ptq_model, trainloader)
        torch.jit.save(torch.jit.script(ptq_model), ptq_scripted_path)
    all_histories['ptq'] = None

    # 5) QAT
    print("\n" + "="*80 + "\n=== 5. QAT (FX Graph Mode) ===\n" + "="*80)
    qat_scripted_path = os.path.join(MODELS_DIR, 'qat_scripted.pt')
    
    if is_training_complete("qat", EPOCHS_QAT):
        print("  [QAT] Loading pre-trained QAT model...")
        qat_model = create_resnet18()
        qat_opt = optim.Adam(qat_model.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(qat_model, qat_opt, "baseline")
        qat_model, qat_history = apply_qat(qat_model, trainloader, criterion, qat_opt, EPOCHS_QAT, load_if_exists=True)
        try: qat_history = pd.read_csv(os.path.join(METRICS_DIR, 'qat_metrics.csv')).to_dict(orient='list')
        except: qat_history = None
    else:
        print("  [QAT] Training QAT from checkpoint...")
        qat_model = create_resnet18()
        qat_opt = optim.Adam(qat_model.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(qat_model, qat_opt, "baseline")
        qat_model, qat_history = apply_qat(qat_model, trainloader, criterion, qat_opt, EPOCHS_QAT, load_if_exists=True)
        if qat_history:
            pd.DataFrame(qat_history).to_csv(os.path.join(METRICS_DIR, 'qat_metrics.csv'), index=False)
        torch.jit.save(torch.jit.script(qat_model), qat_scripted_path)
    all_histories['qat'] = qat_history
    plot_training_history(qat_history, 'qat', PLOTS_DIR)

    # 6) Combined - Multiple Sparsity Levels
    print("\n" + "="*80 + "\n=== 6. Combined: Structured + PTQ ===\n" + "="*80)
    combined_models = {}
    for suffix, sparsity in PRUNING_CONFIGS:
        tag = get_checkpoint_tag("combined", suffix)
        struct_tag = get_checkpoint_tag("structured_pruned", suffix)
        print(f"\n--- Combined (Structured {sparsity*100:.0f}% + PTQ) ---")
        
        combined_quant_saved = os.path.join(MODELS_DIR, f'combined_struct_pruned_ptq_scripted{suffix}.pt')
        
        # Check if combined model quantization is already complete
        if os.path.exists(combined_quant_saved):
            print(f"  [Combined] Pre-quantized model found. Skipping quantization...")
            # Load pruned model WITH masks (for sparsity measurement)
            combined_pruned = create_resnet18()
            combined_pruned = apply_structured_pruning(combined_pruned, sparsity)
            load_checkpoint(combined_pruned, optim.Adam(combined_pruned.parameters(), lr=LEARNING_RATE), struct_tag)
            combined_models[tag] = (combined_pruned, None, sparsity)
            all_histories[tag] = all_histories.get(struct_tag)
            continue
        
        # Otherwise, create quantized version
        if not checkpoint_exists(struct_tag):
            print(f"  [Combined] Structured pruned checkpoint {struct_tag} missing. Skipping...")
            continue
        
        print(f"  [Combined] Creating quantized version...")
        # Load pruned model WITH masks (for sparsity measurement)
        combined_pruned = create_resnet18()
        combined_pruned = apply_structured_pruning(combined_pruned, sparsity)
        load_checkpoint(combined_pruned, optim.Adam(combined_pruned.parameters(), lr=LEARNING_RATE), struct_tag)
        
        # Create dense version for quantization
        dense_model = create_resnet18()
        dense_model = apply_structured_pruning(dense_model, sparsity)
        load_checkpoint(dense_model, optim.Adam(dense_model.parameters()), struct_tag)
        
        # Remove masks from dense version for FX compatibility
        print(f"  [Combined] Removing pruning masks for FX compatibility...")
        for name, module in dense_model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
                try: 
                    prune.remove(module, 'weight')
                except: 
                    pass
        
        # Quantize the dense version - use pre-JIT FX for metrics
        combined_quantized_fx = apply_static_quantization(dense_model, trainloader)
        combined_quantized = combined_quantized_fx
        torch.jit.save(torch.jit.script(combined_quantized_fx), combined_quant_saved)
        
        combined_models[tag] = (combined_pruned, combined_quantized, sparsity)
        all_histories[tag] = all_histories.get(struct_tag)
    
    # 7) Metrics & Plots
    print("\n" + "="*80 + "\n=== Collecting Final Metrics ===\n" + "="*80)
    
    # Build final models dictionary for metrics collection
    final_models = {}
    final_models_sparsity = {}
    
    # Baseline
    baseline_final = create_resnet18()
    load_checkpoint(baseline_final, optim.Adam(baseline_final.parameters()), "baseline")
    final_models['baseline'] = baseline_final
    final_models_sparsity['baseline'] = baseline_final
    
    # Structured pruning variants
    for suffix, sparsity in PRUNING_CONFIGS:
        tag = get_checkpoint_tag("structured_pruned", suffix)
        s_final = create_resnet18()
        s_final = apply_structured_pruning(s_final, sparsity)
        load_checkpoint(s_final, optim.Adam(s_final.parameters()), tag)
        final_models[tag] = s_final
        final_models_sparsity[tag] = s_final
    
    # Global pruning variants
    for suffix, sparsity in PRUNING_CONFIGS:
        tag = get_checkpoint_tag("global_pruned", suffix)
        g_final = create_resnet18()
        g_final = apply_global_pruning(g_final, sparsity)
        load_checkpoint(g_final, optim.Adam(g_final.parameters()), tag)
        final_models[tag] = g_final
        final_models_sparsity[tag] = g_final

    # PTQ
    print("\n  [PTQ] Recreating quantized model for metrics...")
    ptq_final = create_resnet18()
    ptq_opt = optim.Adam(ptq_final.parameters(), lr=LEARNING_RATE)
    _, _ = load_checkpoint(ptq_final, ptq_opt, "baseline")
    ptq_final = apply_static_quantization(ptq_final, trainloader)
    final_models['ptq'] = ptq_final
    final_models_sparsity['ptq'] = ptq_final
    
    # QAT
    print("\n  [QAT] Recreating quantized model for metrics...")
    qat_final = create_resnet18()
    qat_opt = optim.Adam(qat_final.parameters(), lr=LEARNING_RATE)
    _, _ = load_checkpoint(qat_final, qat_opt, "baseline")
    qat_final, _ = apply_qat(qat_final, trainloader, criterion, qat_opt, EPOCHS_QAT, load_if_exists=True)
    final_models['qat'] = qat_final
    final_models_sparsity['qat'] = qat_final
    
    # Combined variants
    print("\n  [Combined] Recreating combined models for metrics...")
    for suffix, sparsity in PRUNING_CONFIGS:
        tag = get_checkpoint_tag("combined", suffix)
        struct_tag = get_checkpoint_tag("structured_pruned", suffix)
        
        if struct_tag not in final_models_sparsity:
            print(f"  [Combined] Skipping {tag} - structured variant not found")
            continue
        
        # Version 1: WITH masks for sparsity measurement
        combined_pruned_final = create_resnet18()
        combined_pruned_final = apply_structured_pruning(combined_pruned_final, sparsity)
        load_checkpoint(combined_pruned_final, optim.Adam(combined_pruned_final.parameters()), struct_tag)
        final_models_sparsity[tag] = combined_pruned_final
        
        # Version 2: Dense quantized for size/latency measurement
        combined_dense_final = create_resnet18()
        combined_dense_final = apply_structured_pruning(combined_dense_final, sparsity)
        load_checkpoint(combined_dense_final, optim.Adam(combined_dense_final.parameters()), struct_tag)
        
        # Remove masks from dense version
        for name, module in combined_dense_final.named_modules():
            if hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
                try: 
                    prune.remove(module, 'weight')
                except: 
                    pass
        
        # Quantize the dense version (use pre-JIT FX for metrics)
        combined_quantized_final = apply_static_quantization(combined_dense_final, trainloader)
        final_models[tag] = combined_quantized_final

    models_dict = final_models
    models_sparsity_dict = final_models_sparsity

    for name, model in models_dict.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING: {name}")
        print(f"{'='*80}")
        print(f"Model type: {type(model).__name__}")
        
        # Use appropriate model for sparsity calculation
        model_for_sparsity = models_sparsity_dict[name]
        
        # Parameter counting
        print(f"\n[PARAMETERS]")
        total_params = param_count(model_for_sparsity, debug=True)
        nonzero_params, total_check = count_nonzero_params(model_for_sparsity, debug=True)
        print(f"  Result: total={total_params:,} | nonzero={nonzero_params:,} | total_from_nonzero_fn={total_check:,}")
        
        # Metrics
        print(f"\n[METRICS]")
        size = model_size(model, name)
        print(f"  Size: {size:.4f} MB")
        
        latency = measure_latency(model)
        print(f"  Latency: {latency:.4f} ms/image")
        
        _, test_acc = evaluate(model, testloader, criterion)
        print(f"  Accuracy: {test_acc:.2f}%")
        
        sparsity = 100.0 * (1.0 - nonzero_params / total_params) if total_params > 0 else 0.0
        print(f"  Sparsity: {sparsity:.2f}%")
        
        # Add note about latency for pruned models
        if 'pruned' in name:
            print(f"\n    NOTE: Latency may be HIGHER than baseline due to masking overhead.")
            print(f"      PyTorch pruning uses masking (weight = weight_orig * mask), not removal.")
            print(f"      For actual speedup, specialized sparse kernels or hardware is needed.")
        
        print(f"\n[SUMMARY]")
        print(f"  params: {total_params:,}")
        print(f"  nonzero_params: {nonzero_params:,}")
        print(f"  sparsity: {sparsity:.2f}%")
        print(f"  size_MB: {size:.4f}")
        print(f"  latency_ms/image: {latency:.4f}")
        print(f"  test_accuracy: {test_acc:.2f}%")
        
        all_records.append({
            "model": name,
            "params": total_params,
            "nonzero_params": nonzero_params,
            "sparsity_%": sparsity,
            "size_MB": size,
            "latency_ms_per_image": latency,
            "test_accuracy": test_acc
        })

    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RESULTS_DIR, "model_summary.csv")
    
    print("\n" + "="*80)
    print("=== WRITING RESULTS ===")
    print("="*80)
    
    # Try multiple approaches to write CSV
    csv_written = False
    
    # Try 1: Direct write with file removal
    try:
        if os.path.exists(csv_path):
            try:
                os.remove(csv_path)
                print(f"  ✓ Removed existing CSV file: {csv_path}")
            except Exception as e:
                print(f"  ✗ Failed to remove CSV: {e}")
        
        df.to_csv(csv_path, index=False)
        print(f"  ✓ Successfully wrote CSV to {csv_path}")
        csv_written = True
    except PermissionError as e:
        print(f"  ✗ Permission error: {e}")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # Try 2: Wait and retry
    if not csv_written:
        print(f"  Attempting retry with delay...")
        try:
            import time
            time.sleep(1)
            df.to_csv(csv_path, index=False)
            print(f"  ✓ Successfully wrote CSV on retry")
            csv_written = True
        except Exception as e:
            print(f"  ✗ Retry failed: {e}")
    
    # Try 3: Write to alternative location
    if not csv_written:
        alt_csv = os.path.join(RESULTS_DIR, "model_summary_alt.csv")
        print(f"  Attempting to write to alternative location: {alt_csv}")
        try:
            df.to_csv(alt_csv, index=False)
            print(f"  ✓ Successfully wrote CSV to alternative location: {alt_csv}")
            csv_written = True
        except Exception as e:
            print(f"  ✗ Alternative write failed: {e}")
    
    print("\n" + "="*80)
    print("=== RESULTS SUMMARY ===")
    print("="*80)
    print(df.to_string(index=False))
    
    print("\n" + "="*80)
    print("=== GENERATING PLOTS ===")
    print("="*80)
    print("Generating comparison plots...")
    plot_comparison(df, PLOTS_DIR)
    print("Generating efficiency plots...")
    plot_efficiency_metrics(df, PLOTS_DIR)
    print("Generating training history comparison...")
    plot_all_models_comparison(all_histories, PLOTS_DIR)
    
    print("\n" + "="*80)
    print("=== EXPERIMENT COMPLETE ===")
    print("="*80)
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"- Models: {MODELS_DIR}")
    print(f"- Plots: {PLOTS_DIR}")
    print(f"- Metrics: {METRICS_DIR}")
    print(f"- Checkpoints: {CHECKPOINT_DIR}")

if __name__ == "__main__":
    run_experiments()