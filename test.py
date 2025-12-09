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

# Directories
RESULTS_DIR = "experiment_results_cifar10_resnet18"
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Hyperparameters
EPOCHS_BASELINE = 10
EPOCHS_PRUNE = 10
EPOCHS_PTQ = 10
EPOCHS_QAT = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 128
STRUCT_PRUNE_AMT = 0.2
GLOBAL_PRUNE_AMT = 0.2

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

trainset = datasets.CIFAR10(".", train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10(".", train=False, download=True, transform=transform_test)
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
def save_checkpoint(model, optimizer, epoch, test_acc, tag):
    model_cpu = model.to(torch.device("cpu"))
    state = {
        'epoch': epoch,
        'model_state_dict': model_cpu.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_test_acc': test_acc,
        'tag': tag
    }
    filepath = os.path.join(CHECKPOINT_DIR, f'{tag}_best.pth.tar')
    torch.save(state, filepath)
    print(f"  [Checkpoint] Saved checkpoint for {tag} -> {filepath}")

def load_checkpoint(model, optimizer, tag):
    filepath = os.path.join(CHECKPOINT_DIR, f'{tag}_best.pth.tar')
    if not os.path.exists(filepath):
        print(f"  [Checkpoint] No checkpoint found for {tag}.")
        return 0, 0.0
    checkpoint = torch.load(filepath, map_location=device)
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
            save_checkpoint(model, optimizer, epoch + 1, best_test_acc, tag)
        
        print(f"[{tag}] Epoch {epoch+1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}% (Best: {best_test_acc:.2f}%)")
    
    if best_test_acc > 0.0:
        print(f"[{tag}] Loading best saved model with accuracy {best_test_acc:.2f}% for final use...")
        dummy_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        load_checkpoint(model, dummy_optimizer, tag)
        
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
    Calculate model size in MB by summing bytes of all weight and bias parameters.
    Handles:
    - Regular models: counts 'weight' and 'bias' parameters
    - Pruned models: counts 'weight_orig' and 'bias_orig' (mask doesn't change storage size)
    - Quantized models: counts quantized weight/bias buffers with appropriate element size
    
    Key insight: For pruned models, the storage size is unchanged because:
    - weight_orig is still float32 (or original dtype)
    - mask is metadata (boolean), typically not serialized with weight
    - When saved to disk, most frameworks only save weight_orig, not masks
    
    For quantized models, buffers are used for quantized weights (typically int8).
    """
    def count_bytes(mod, prefix=""):
        total_bytes = 0
        
        # Iterate over named_parameters with recurse=False
        for name, param in mod.named_parameters(recurse=False):
            # Count weight_orig/bias_orig for pruned models (these replace weight/bias)
            if name in ['weight_orig', 'bias_orig']:
                param_bytes = param.numel() * param.element_size()
                total_bytes += param_bytes
                if debug:
                    print(f"      {prefix}{name} (pruned): {param.numel():,} params × {param.element_size()} bytes = {param_bytes:,} bytes")
            # Count weight/bias for regular unpruned models
            elif name in ['weight', 'bias']:
                param_bytes = param.numel() * param.element_size()
                total_bytes += param_bytes
                if debug:
                    print(f"      {prefix}{name}: {param.numel():,} params × {param.element_size()} bytes = {param_bytes:,} bytes")
        
        # For quantized models, check for quantized weight/bias buffers
        # Note: These are only present if model was converted to quantized format
        for name, buf in mod.named_buffers(recurse=False):
            # Skip masks - they're metadata, not serialized with model weights
            if 'mask' in name:
                continue
            # Count quantized weight/bias buffers (typically int8, smaller than float32)
            if name in ['weight', 'bias'] and isinstance(buf, torch.Tensor):
                buf_bytes = buf.numel() * buf.element_size()
                total_bytes += buf_bytes
                if debug:
                    print(f"      {prefix}{name} (quantized buffer): {buf.numel():,} params × {buf.element_size()} bytes = {buf_bytes:,} bytes")
        
        # Recursively process child modules
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

# ---------------------------------------------------------
# FIXED PARAMETER COUNTING FUNCTIONS
# ---------------------------------------------------------

def param_count(model, debug=False):
    """
    Count total number of parameters in the model.
    
    Properly handles:
    - Regular models: counts 'weight' and 'bias' parameters
    - Pruned models: counts 'weight_orig' and 'bias_orig' (NOT the masks)
    - Quantized models: counts all parameter elements (quantization doesn't reduce param count)
    
    Key distinction:
    - Parameter count = number of elements in the tensor
    - Storage size = param_count × element_size() in bytes
    - Pruning: doesn't change parameter count, only sparsity (# zeros)
    - Quantization: doesn't change parameter count, only precision (element_size)
    
    Returns: Integer count of total trainable/model parameters
    """
    def count_params(mod, prefix=""):
        total = 0
        
        # Use named_parameters() to get only trainable parameters
        # recurse=False means only direct parameters of this module
        for name, param in mod.named_parameters(recurse=False):
            # For pruned models: weight_orig and bias_orig are the parameter tensors
            # The weight_mask and bias_mask are buffers, not parameters
            if name in ['weight_orig', 'bias_orig']:
                num_params = param.numel()
                total += num_params
                if debug:
                    print(f"      {prefix}{name} (pruned param): {num_params:,}")
            # For regular unpruned models: weight and bias are parameters
            elif name in ['weight', 'bias']:
                num_params = param.numel()
                total += num_params
                if debug:
                    print(f"      {prefix}{name}: {num_params:,}")
            # Other parameters (batch norm scale/bias, etc.)
            else:
                num_params = param.numel()
                total += num_params
                if debug:
                    print(f"      {prefix}{name} (other): {num_params:,}")
        
        # IMPORTANT: Do NOT count buffers as parameters
        # Buffers include:
        # - Pruning masks (weight_mask, bias_mask) - these are metadata
        # - Quantization parameters (scale, zero_point) - these are derived from params
        # - Batch norm statistics (running_mean, running_var) - these are not learnable
        
        # Recursively count submodules
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
    Count non-zero parameters, properly handling pruning masks.
    For pruned models: applies mask to weight_orig to get actual nonzero count.
    For regular models: counts nonzero in weight directly.
    For quantized models: counts nonzero in weight buffers.
    """
    def count_nonzero(mod, prefix=""):
        nonzero = 0
        total = 0
        
        # Check if this module has pruning applied
        has_pruning = hasattr(mod, 'weight_mask') or hasattr(mod, 'weight_orig')
        
        for name, param in mod.named_parameters(recurse=False):
            if name in ['weight_orig', 'bias_orig']:
                # Pruned parameter - apply mask if available
                param_total = param.numel()
                total += param_total
                
                if name == 'weight_orig' and hasattr(mod, 'weight_mask'):
                    # Apply mask to get true nonzero count
                    masked_weight = param * mod.weight_mask
                    param_nonzero = torch.count_nonzero(masked_weight).item()
                elif name == 'bias_orig' and hasattr(mod, 'bias_mask'):
                    masked_bias = param * mod.bias_mask
                    param_nonzero = torch.count_nonzero(masked_bias).item()
                else:
                    param_nonzero = torch.count_nonzero(param).item()
                
                if debug:
                    print(f"      {prefix}{name}: {param_nonzero:,} / {param_total:,}")
                nonzero += param_nonzero
                
            elif name in ['weight', 'bias'] and not has_pruning:
                # Regular unpruned parameter
                param_total = param.numel()
                total += param_total
                param_nonzero = torch.count_nonzero(param).item()
                if debug:
                    print(f"      {prefix}{name}: {param_nonzero:,} / {param_total:,}")
                nonzero += param_nonzero
        
        # For quantized models, check buffers
        for name, buf in mod.named_buffers(recurse=False):
            if name == 'weight_mask' or name == 'bias_mask':
                # Skip masks, we already used them above
                continue
            if name in ['weight', 'bias'] and isinstance(buf, torch.Tensor):
                buf_total = buf.numel()
                total += buf_total
                buf_nonzero = torch.count_nonzero(buf).item()
                if debug:
                    print(f"      {prefix}{name} (buffer): {buf_nonzero:,} / {buf_total:,}")
                nonzero += buf_nonzero
        
        # Recursively count submodules
        for subname, submod in mod.named_children():
            sub_nonzero, sub_total = count_nonzero(submod, prefix=prefix+subname+".")
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
    baseline = create_resnet18()
    optimizer = optim.Adam(baseline.parameters(), lr=LEARNING_RATE)
    baseline, baseline_history = train(baseline, trainloader, criterion, optimizer, EPOCHS_BASELINE, tag="baseline", load_if_exists=True)
    all_histories['baseline'] = baseline_history
    pd.DataFrame(baseline_history).to_csv(os.path.join(METRICS_DIR, 'baseline_metrics.csv'), index=False)
    plot_training_history(baseline_history, 'baseline', PLOTS_DIR)

    # 2) Structured Pruning
    print("\n" + "="*80 + "\n=== 2. Structured Pruning ===\n" + "="*80)
    s_model = create_resnet18()
    s_optimizer = optim.Adam(s_model.parameters(), lr=LEARNING_RATE)
    _, _ = load_checkpoint(s_model, s_optimizer, "baseline")
    s_model = apply_structured_pruning(s_model, STRUCT_PRUNE_AMT)
    if checkpoint_exists("structured_pruned"):
        load_checkpoint(s_model, s_optimizer, "structured_pruned")
        try: s_history = pd.read_csv(os.path.join(METRICS_DIR, 'structured_pruned_metrics.csv')).to_dict(orient='list')
        except: s_history = None
    else:
        s_model, s_history = train(s_model, trainloader, criterion, s_optimizer, EPOCHS_PRUNE, tag="structured_pruned", load_if_exists=False)
        pd.DataFrame(s_history).to_csv(os.path.join(METRICS_DIR, 'structured_pruned_metrics.csv'), index=False)
    all_histories['structured_pruned'] = s_history
    plot_training_history(s_history, 'structured_pruned', PLOTS_DIR)

    # 3) Global Pruning
    print("\n" + "="*80 + "\n=== 3. Global Pruning ===\n" + "="*80)
    g_model = create_resnet18()
    g_optimizer = optim.Adam(g_model.parameters(), lr=LEARNING_RATE)
    _, _ = load_checkpoint(g_model, g_optimizer, "baseline")
    g_model = apply_global_pruning(g_model, GLOBAL_PRUNE_AMT)
    if checkpoint_exists("global_pruned"):
        load_checkpoint(g_model, g_optimizer, "global_pruned")
        try: g_history = pd.read_csv(os.path.join(METRICS_DIR, 'global_pruned_metrics.csv')).to_dict(orient='list')
        except: g_history = None
    else:
        g_model, g_history = train(g_model, trainloader, criterion, g_optimizer, EPOCHS_PRUNE, tag="global_pruned", load_if_exists=False)
        pd.DataFrame(g_history).to_csv(os.path.join(METRICS_DIR, 'global_pruned_metrics.csv'), index=False)
    all_histories['global_pruned'] = g_history
    plot_training_history(g_history, 'global_pruned', PLOTS_DIR)

    # 4) PTQ
    print("\n" + "="*80 + "\n=== 4. PTQ (FX Graph Mode) ===\n" + "="*80)
    ptq_model = create_resnet18()
    ptq_scripted_path = os.path.join(MODELS_DIR, 'ptq_scripted.pt')
    
    if os.path.exists(ptq_scripted_path):
        print("  [PTQ] Loading scripted PTQ model.")
        # For metrics, we need the pre-JIT model, so recreate it
        ptq_model_for_metrics = create_resnet18()
        ptq_opt = optim.Adam(ptq_model_for_metrics.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(ptq_model_for_metrics, ptq_opt, "baseline")
        ptq_model_for_metrics = apply_static_quantization(ptq_model_for_metrics, trainloader)
        ptq_model = ptq_model_for_metrics  # Use pre-JIT for metrics
    else:
        ptq_opt = optim.Adam(ptq_model.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(ptq_model, ptq_opt, "baseline")
        ptq_model = apply_static_quantization(ptq_model, trainloader)
        torch.jit.save(torch.jit.script(ptq_model), ptq_scripted_path)
    all_histories['ptq'] = None

    # 5) QAT
    print("\n" + "="*80 + "\n=== 5. QAT (FX Graph Mode) ===\n" + "="*80)
    qat_model = create_resnet18()
    qat_scripted_path = os.path.join(MODELS_DIR, 'qat_scripted.pt')
    
    if os.path.exists(qat_scripted_path) and checkpoint_exists("qat"):
         print("  [QAT] Checkpoint found. Loading saved QAT model...")
         # For metrics, recreate from checkpoint (which will load it via apply_qat with load_if_exists=True)
         qat_model_for_metrics = create_resnet18()
         qat_opt = optim.Adam(qat_model_for_metrics.parameters(), lr=LEARNING_RATE)
         _, _ = load_checkpoint(qat_model_for_metrics, qat_opt, "baseline")
         qat_model_for_metrics, qat_history = apply_qat(qat_model_for_metrics, trainloader, criterion, qat_opt, EPOCHS_QAT, load_if_exists=True)
         qat_model = qat_model_for_metrics  # Use pre-JIT for metrics
         try: qat_history = pd.read_csv(os.path.join(METRICS_DIR, 'qat_metrics.csv')).to_dict(orient='list')
         except: qat_history = None
    else:
        print("  [QAT] No checkpoint found. Training QAT from scratch...")
        qat_opt = optim.Adam(qat_model.parameters(), lr=LEARNING_RATE)
        _, _ = load_checkpoint(qat_model, qat_opt, "baseline")
        qat_model, qat_history = apply_qat(qat_model, trainloader, criterion, qat_opt, EPOCHS_QAT, load_if_exists=False)
        pd.DataFrame(qat_history).to_csv(os.path.join(METRICS_DIR, 'qat_metrics.csv'), index=False)
        torch.jit.save(torch.jit.script(qat_model), qat_scripted_path)
    all_histories['qat'] = qat_history
    plot_training_history(qat_history, 'qat', PLOTS_DIR)

    # 6) Combined - FIXED VERSION (Keep pruning masks for sparsity measurement)
    print("\n" + "="*80 + "\n=== 6. Combined: Structured + PTQ (FIXED) ===\n" + "="*80)
    combined_model = create_resnet18()
    combined_quant_saved = os.path.join(MODELS_DIR, 'combined_struct_pruned_ptq_scripted.pt')
    
    # We need TWO versions:
    # 1. Pruned version WITH masks for sparsity calculation
    # 2. Dense quantized version for actual inference
    
    combined_pruned = create_resnet18()
    c_opt = optim.Adam(combined_pruned.parameters(), lr=LEARNING_RATE)
    
    if not checkpoint_exists("structured_pruned"):
        raise RuntimeError("Structured pruned checkpoint missing.")
    
    # Load pruned model WITH masks (for sparsity measurement)
    combined_pruned = apply_structured_pruning(combined_pruned, STRUCT_PRUNE_AMT)
    load_checkpoint(combined_pruned, c_opt, "structured_pruned")
    
    # Create quantized version
    if os.path.exists(combined_quant_saved):
        print("  [Combined] Loading existing combined PTQ model.")
        combined_quantized = torch.jit.load(combined_quant_saved)
    else:
        print("  [Combined] Creating dense version for quantization...")
        dense_model = create_resnet18()
        dense_model = apply_structured_pruning(dense_model, STRUCT_PRUNE_AMT)
        load_checkpoint(dense_model, optim.Adam(dense_model.parameters()), "structured_pruned")
        
        # Remove masks from dense version for FX compatibility
        print("  [Combined] Removing pruning masks for FX compatibility...")
        for name, module in dense_model.named_modules():
            if hasattr(module, 'weight') and isinstance(module, nn.Conv2d):
                try: 
                    prune.remove(module, 'weight')
                except: 
                    pass
        
        # Quantize the dense version
        combined_quantized = apply_static_quantization(dense_model, trainloader)
        torch.jit.save(torch.jit.script(combined_quantized), combined_quant_saved)
    
    # For evaluation, we'll use the quantized version but report sparsity from pruned version
    combined_model = combined_quantized
    combined_pruned_for_sparsity = combined_pruned  # Keep this for sparsity calculation
    
    all_histories['combined'] = all_histories.get('structured_pruned')
    
    # 7) Metrics & Plots - FIXED VERSION
    print("\n" + "="*80 + "\n=== Collecting Final Metrics (FIXED) ===\n" + "="*80)
    
    # Reload models - KEEP pruning masks on pruned models for proper sparsity measurement
    baseline_final = create_resnet18()
    load_checkpoint(baseline_final, optim.Adam(baseline_final.parameters()), "baseline")
    
    # Structured: load WITH wrapper to keep masks
    s_final = create_resnet18()
    s_final = apply_structured_pruning(s_final, STRUCT_PRUNE_AMT)
    load_checkpoint(s_final, optim.Adam(s_final.parameters()), "structured_pruned")
    # DON'T remove masks - we need them for sparsity calculation
    
    # Global: load WITH wrapper
    g_final = create_resnet18()
    g_final = apply_global_pruning(g_final, GLOBAL_PRUNE_AMT)
    load_checkpoint(g_final, optim.Adam(g_final.parameters()), "global_pruned")
    # DON'T remove masks - we need them for sparsity calculation

    models_dict = {
        "baseline": baseline_final,
        "structured_pruned": s_final,
        "global_pruned": g_final,
        "ptq": ptq_model,
        "qat": qat_model,
        "combined": combined_model
    }
    
    # Special: track the pruned version of combined for sparsity
    models_sparsity_dict = {
        "baseline": baseline_final,
        "structured_pruned": s_final,
        "global_pruned": g_final,
        "ptq": ptq_model,
        "qat": qat_model,
        "combined": combined_pruned_for_sparsity  # Use pruned version for sparsity
    }

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
            print(f"\n  ⚠️  NOTE: Latency may be HIGHER than baseline due to masking overhead.")
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
    print("\nIMPORTANT NOTES:")
    print("1. Sparsity values should now correctly show ~20% for pruned models")
    print("2. Parameter counts should be accurate for all model types")
    print("3. Latency for pruned models may be HIGHER than baseline due to masking overhead")
    print("4. For actual inference speedup, consider:")
    print("   - Using sparse tensor formats (torch.sparse)")
    print("   - Specialized sparse computation libraries (Intel DNNL, NVIDIA cuSPARSE)")
    print("   - Hardware with native sparse support")
    print("   - Model export formats that support sparsity (ONNX, TensorRT)")
    print("="*80)

if __name__ == "__main__":
    run_experiments()