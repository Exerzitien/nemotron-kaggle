# Prompt for 5090 WSL Agent (paste into Claude Code on that machine)

> **Background context**: I'm collaborating with the agent on the Mac side
> who's running a Kaggle Nemotron reasoning competition. We have a trained
> LoRA adapter ("V33") that scored a disastrous 0.73 LB (vs baseline 0.84).
> We need to diagnose **which category broke**. The fastest way is to run the
> V33 adapter through a 950-question local validation set on this RTX 5090.
>
> Your job: set up the env on this RTX 5090 (WSL Ubuntu), download all data,
> and run a self-contained eval script. Report the per-category accuracy back.

---

## 🎯 Goal
Produce `~/eval_results/v33_per_class.txt` — a per-category accuracy table for
V33 adapter on val_set_950. Plus mistake CSVs per category.

## ⚙️ Environment expected
- RTX 5090 (Blackwell sm_120, 32GB VRAM)
- WSL2 Ubuntu (any recent distro)
- ~80GB free disk
- Internet access for HuggingFace + Kaggle downloads

## 📋 Tasks (do each in order)

### Phase 0: Sanity check
```bash
nvidia-smi | head -15
python3 --version
nvcc --version 2>&1 | tail -3 || echo "CUDA toolkit may be auto-installed by torch wheel"
df -h ~
```
Report: GPU model, VRAM, CUDA driver version, free disk.

### Phase 1: Python env + core packages

```bash
# Create venv
python3 -m venv ~/nemo-env
source ~/nemo-env/bin/activate
pip install --upgrade pip wheel

# Install PyTorch with CUDA 12.4+ (Blackwell support — use latest nightly if 5090 not in stable)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# If above fails (Blackwell wheel missing), fall back to nightly:
#   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# Verify GPU detection
python3 -c "import torch; print('CUDA OK:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
**Critical**: confirm `torch.cuda.is_available() == True` and device name shows "5090".
If False, stop and investigate driver/CUDA compatibility before proceeding.

### Phase 2: Install vLLM + bitsandbytes + peft

```bash
pip install vllm  # 0.7+ supports bitsandbytes 4-bit quantization
pip install bitsandbytes peft transformers accelerate huggingface_hub
pip install kaggle  # for downloading datasets

# Verify imports
python3 -c "from vllm import LLM; from vllm.lora.request import LoRARequest; print('vllm OK')"
python3 -c "import bitsandbytes; print('bnb OK', bitsandbytes.__version__)"
```

### Phase 3: HuggingFace login (for Nemotron base model)

The user (me on Mac) provides HF token via env var if model requires login.
First try without login (Nemotron may be public):

```bash
huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 \
    --local-dir ~/models/Nemotron-3-Nano-30B-A3B-BF16 \
    --max-workers 8
```

If 401: ask user for HF token and run `huggingface-cli login`.
This is ~60GB download. Will take 20min-2h depending on bandwidth.

### Phase 4: Kaggle credentials + downloads

Tell user to place `kaggle.json` at `~/.kaggle/kaggle.json` (chmod 600).
If they need to know how to get it: kaggle.com/settings → "Create New Token".

Then:
```bash
mkdir -p ~/data ~/adapters
# V33 adapter (4GB)
kaggle datasets download exerzitien/nemotron-v33-adapter -p ~/adapters/v33-zip --unzip
# Move adapter files out of submission.zip-extracted dir
unzip -o ~/adapters/v33-zip/submission.zip -d ~/adapters/v33
ls ~/adapters/v33  # should have adapter_config.json + adapter_model.safetensors

# val_set_950
kaggle datasets download jiazhuang/nemotron-val-set-950 -p ~/data --unzip
ls ~/data | grep val_set_950  # confirm val_set_950.jsonl present
```

### Phase 5: Get the eval script

The script lives at `/Users/exerzitien/Desktop/nemotron-kaggle/scripts/eval_v33_on_5090.py`
on the Mac side. The user will need to transfer it to this WSL machine.

**Easiest transfer method**: paste the script content into a heredoc.
Ask the user to provide the script content if not already on this machine.

Save it as `~/eval_v33.py`.

### Phase 6: Run eval

```bash
cd ~
source ~/nemo-env/bin/activate
python3 ~/eval_v33.py 2>&1 | tee ~/eval_results/run.log
```

Expected runtime: **30 min - 2h** depending on 5090 throughput with INT4 + LoRA.
First few iters should show vLLM loading; if it OOMs, report immediately
(may need to lower `max_num_seqs` or `gpu_memory_utilization` in script).

### Phase 7: Report back

Copy and paste back to user:

1. **Per-category accuracy table** from `~/eval_results/v33_per_class.txt`
2. **First 3 mistakes per category** from `~/eval_results/v33_mistakes/*.csv`
   (just id, gold, pred, raw_tail columns)
3. **Total runtime**

If you hit any blocking error (OOM, missing CUDA libs, vLLM crash), report
the exact error message + which phase failed.

## ❗ Known gotchas

- **5090 Blackwell**: vLLM may need 0.7+ with sm_120 support. If you see
  "no kernel image is available for execution on the device", upgrade vLLM
  or install nightly PyTorch with CUDA 12.8.
- **Adapter target_parameters**: V33 adapter_config.json has
  `target_parameters: ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]`
  which is for Unsloth MoE injection. vLLM PEFT may not recognize this field
  but should still load the standard LoRA tensors. If vLLM errors on
  config parsing, edit adapter_config.json and set `target_parameters: null`.
- **Disk full**: monitor with `df -h ~` between phases. Nemotron base is 60GB
  + V33 adapter 4GB + val set 100MB + Python deps 5GB + cache.
- **bitsandbytes + vLLM**: ensure `quantization="bitsandbytes"` works on
  Blackwell. If not, try `quantization=None` (BF16) — model is 60GB, will
  OOM on 32GB VRAM. In that case report back and we'll use a different
  quant strategy (AWQ or split inference).

## 🚦 Decision points

Stop and ask the user for guidance if:
- HF token needed (Nemotron model behind gate)
- Kaggle credentials missing
- 5090 not detected by torch
- vLLM crashes on Blackwell
- Disk runs out partway through

Don't make up info — when blocked, list the exact error + your hypothesis,
and let the user resolve.
