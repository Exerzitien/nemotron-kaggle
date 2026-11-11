"""
Tinker checkpoint → PEFT adapter → Kaggle dataset
==================================================

Pipeline:
  1. Download Tinker LoRA checkpoint (raw format)
  2. Apply huikang's SVD fusion (gate_proj + x_proj → in_proj rank 32)
     This is necessary because Mamba stores them as 2× rank 32 in Tinker
     but vLLM/PEFT expects single rank 32 in_proj.
  3. Verify adapter_config.json is vLLM-compatible
  4. Zip as submission.zip + upload to Kaggle dataset

Usage:
    python tinker_to_kaggle.py \\
        --tinker-path tinker://uuid \\
        --dataset-id exerzitien/nemotron-v35-adapter \\
        --version-label "V35: bit_v4 swap"

Or to list latest Tinker checkpoints:
    python tinker_to_kaggle.py --list
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def list_checkpoints():
    """List Tinker checkpoints via SDK."""
    print("Listing latest Tinker checkpoints...")
    try:
        result = subprocess.run(
            ["tinker", "-f", "json", "checkpoint", "list"],
            check=True, capture_output=True, text=True,
        )
        ckpts = json.loads(result.stdout)
        for c in ckpts[:20]:
            print(f"  {c.get('tinker_path', '?'):60s} {c.get('created_at', '?')}")
    except Exception as e:
        print(f"  Failed to list: {e}")
        print("  Try: tinker checkpoint list")


def download_tinker_checkpoint(tinker_path, out_dir):
    """Download Tinker raw checkpoint to local dir."""
    if not os.environ.get("TINKER_API_KEY"):
        key_path = Path("~/.tinker_key").expanduser()
        if key_path.exists():
            os.environ["TINKER_API_KEY"] = key_path.read_text().strip()

    import tinker
    from tinker_cookbook import weights as tw

    print(f"Downloading Tinker checkpoint from {tinker_path}...")
    extracted = tw.download(tinker_path=tinker_path, output_dir=str(out_dir))
    print(f"  ✓ extracted to: {extracted}")
    return extracted


def convert_to_peft(tinker_extracted_dir, peft_out_dir, base_model):
    """Apply build_lora_adapter (handles SVD fusion for Mamba in_proj)."""
    from tinker_cookbook import weights as tw

    print(f"Converting Tinker → PEFT at {peft_out_dir}...")
    tw.build_lora_adapter(
        base_model=base_model,
        adapter_path=tinker_extracted_dir,
        output_path=str(peft_out_dir),
        trust_remote_code=True,
    )
    print(f"  ✓ PEFT adapter saved")

    # Validate
    cfg_path = Path(peft_out_dir) / "adapter_config.json"
    weights_path = Path(peft_out_dir) / "adapter_model.safetensors"
    assert cfg_path.exists(), f"Missing {cfg_path}"
    assert weights_path.exists(), f"Missing {weights_path}"

    with open(cfg_path) as f:
        cfg = json.load(f)
    print(f"  rank: {cfg.get('r')}")
    print(f"  target_modules: {cfg.get('target_modules')}")
    print(f"  adapter_model.safetensors size: "
          f"{weights_path.stat().st_size / 1024**3:.2f} GB")

    # Sanity: rank ≤ 32 (competition rule)
    assert cfg.get("r", 0) <= 32, f"rank > 32: {cfg.get('r')} (competition violation)"

    return cfg


def package_for_kaggle(peft_dir, out_zip):
    """Zip adapter_config.json + adapter_model.safetensors as submission.zip."""
    print(f"Packaging {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for name in ["adapter_config.json", "adapter_model.safetensors"]:
            src = Path(peft_dir) / name
            zf.write(src, name)
    print(f"  ✓ {out_zip} ({out_zip.stat().st_size / 1024**3:.2f} GB)")


def upload_to_kaggle(zip_path, dataset_id, version_label, fresh=False):
    """Push as new dataset version (or create if not exists)."""
    print(f"Uploading to Kaggle dataset {dataset_id}...")
    stage_dir = Path(tempfile.mkdtemp(prefix="kaggle-stage-"))
    shutil.copy(zip_path, stage_dir / "submission.zip")
    (stage_dir / "dataset-metadata.json").write_text(json.dumps({
        "title": dataset_id.split("/")[-1].replace("-", " ").title(),
        "id": dataset_id,
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))

    cmd = ["kaggle", "datasets", "version", "-m", version_label, "-r", "zip", "--dir-mode", "skip"]
    if fresh:
        cmd = ["kaggle", "datasets", "create", "-r", "zip", "--dir-mode", "skip"]

    result = subprocess.run(cmd, cwd=stage_dir, capture_output=True, text=True)
    if result.returncode != 0:
        if "dataset not found" in (result.stdout + result.stderr).lower():
            print("  Dataset doesn't exist — creating fresh...")
            return upload_to_kaggle(zip_path, dataset_id, version_label, fresh=True)
        print(f"  ❌ Upload failed:\n{result.stdout}\n{result.stderr}")
        return False
    print(f"  ✓ Upload submitted. Visit https://www.kaggle.com/datasets/{dataset_id}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tinker-path", help="tinker://uuid checkpoint URI")
    p.add_argument("--dataset-id", default="exerzitien/nemotron-v35-adapter")
    p.add_argument("--version-label", default="V35 adapter from Tinker")
    p.add_argument("--base-model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
    p.add_argument("--list", action="store_true", help="list recent Tinker checkpoints")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="don't delete temp dirs after upload")
    args = p.parse_args()

    if args.list:
        list_checkpoints()
        return

    if not args.tinker_path:
        print("ERROR: --tinker-path required (or use --list to find one)")
        sys.exit(1)

    work_dir = Path(tempfile.mkdtemp(prefix="tinker-to-kaggle-"))
    print(f"Work dir: {work_dir}\n")

    try:
        # Phase 1: download Tinker checkpoint
        raw_dir = work_dir / "raw"
        extracted = download_tinker_checkpoint(args.tinker_path, raw_dir)

        # Phase 2: SVD-fuse + convert to PEFT
        peft_dir = work_dir / "peft"
        peft_dir.mkdir()
        convert_to_peft(extracted, peft_dir, args.base_model)

        # Phase 3: zip + upload
        zip_path = work_dir / "submission.zip"
        package_for_kaggle(peft_dir, zip_path)
        upload_to_kaggle(zip_path, args.dataset_id, args.version_label)

        print(f"\n✅ All done.")
        print(f"\nNext step: in Kaggle eval kernel, set dataset_sources to include")
        print(f"  '{args.dataset_id}'")
        print(f"and resubmit nemotron-lora-sft via 'Submit Predictions' for LB.")
    finally:
        if not args.keep_intermediate:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\nIntermediate files kept at: {work_dir}")


if __name__ == "__main__":
    main()
