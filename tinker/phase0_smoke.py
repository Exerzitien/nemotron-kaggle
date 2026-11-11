"""Phase 0 smoke test for Tinker compatibility with our Nemotron LoRA pipeline.

Goal: in 1-2 hours and <$5, verify the four things that decide whether Tinker
can be used as our extra-compute path:

  1. Nemotron-3-Nano-30B-A3B is in the supported model list.
  2. We can create a LoRA training client at rank=32 (competition cap).
  3. A trivial 5-step training run produces a checkpoint we can export as
     PEFT-format adapter_config.json + adapter_model.safetensors.
  4. The exported adapter_config.json `target_modules` covers the parameters
     dgxchen's Unsloth recipe trained (especially MoE experts -- this is the
     biggest open compatibility question).

Anything past Phase 0 (real training data, multi-epoch, ablation runs) is
deferred until this passes.

Usage:
    export TINKER_API_KEY="..."          # from https://tinker-console.thinkingmachines.ai
    pip install tinker
    python phase0_smoke.py
"""

import asyncio
import json
import os
import sys
import tarfile
import tempfile

# All the modules we care about touch network, so we import lazily and let the
# user see a friendly error if `tinker` isn't installed yet.
try:
    import tinker
    from tinker import types
except ImportError:
    print("ERROR: `tinker` not installed. Run: pip install tinker", file=sys.stderr)
    sys.exit(1)


# The competition uses metric/nemotron-3-nano-30b-a3b-bf16. Tinker's identifier
# is almost certainly one of these; we'll probe until one works.
NEMOTRON_CANDIDATES = [
    "nvidia/Nemotron-3-Nano-30B-A3B",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "Nemotron-3-Nano-30B-A3B",
]


async def main():
    assert os.environ.get("TINKER_API_KEY"), \
        "TINKER_API_KEY env var not set. Get one from https://tinker-console.thinkingmachines.ai"
    sc = tinker.ServiceClient()
    print("ServiceClient initialised OK")

    # --- 1. List supported models -------------------------------------------
    try:
        if hasattr(sc, "get_server_capabilities_async"):
            caps = await sc.get_server_capabilities_async()
        else:
            caps = sc.get_server_capabilities()
        models = getattr(caps, "supported_models", None) or caps
        print("\n=== Supported models (sampled) ===")
        # caps shape may vary; try to render it
        rendered = json.dumps(models, default=lambda o: getattr(o, "__dict__", str(o)), indent=2)
        print(rendered[:3000])
    except Exception as e:
        print(f"WARN: get_server_capabilities() failed: {e}")
        rendered = ""

    # --- 2. Find a Nemotron identifier that Tinker accepts ------------------
    chosen = None
    tc = None
    for candidate in NEMOTRON_CANDIDATES:
        try:
            print(f"\nProbing base_model={candidate!r} ...")
            tc = await sc.create_lora_training_client_async(base_model=candidate, rank=32)
            chosen = candidate
            print(f"  ✓ accepted: {candidate}")
            break
        except Exception as e:
            print(f"  ✗ rejected: {type(e).__name__}: {str(e)[:200]}")
    if not chosen:
        print("\nFAIL: no Nemotron identifier accepted. Inspect the supported_models output above "
              "and adjust NEMOTRON_CANDIDATES in this script.")
        return 1

    # --- 3. Five training steps on trivial data so we have a checkpoint ----
    tokenizer = tc.get_tokenizer()
    text = "We need to infer the transformation rule from the examples."
    ids = tokenizer.encode(text)
    if not ids:
        ids = [1, 2, 3, 4, 5]
    print(f"\nUsing {len(ids)} tokens for the dummy training example.")

    target = ids[1:] + [ids[-1]]
    weights = [1.0] * len(ids)
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(tokens=ids),
        loss_fn_inputs=dict(weights=weights, target_tokens=target),
    )

    print("Running 5 forward/backward + optim steps ...")
    for step in range(5):
        fb = await tc.forward_backward_async(data=[datum], loss_fn="cross_entropy")
        await fb.result_async()
        opt = await tc.optim_step_async(types.AdamParams(learning_rate=1e-4))
        await opt.result_async()
        print(f"  step {step + 1}/5 done")

    # --- 4. Save weights and export PEFT-format adapter ---------------------
    print("\nSaving weights for sampler ...")
    save_future = await tc.save_weights_for_sampler_async(name="phase0-smoke")
    save = await save_future.result_async()
    print(f"  saved path: {save.path}")

    # The PEFT export workflow:
    #   1. download Tinker's checkpoint archive to a local dir
    #   2. build_lora_adapter() converts it to PEFT format (adapter_config.json
    #      + adapter_model.safetensors)
    from tinker_cookbook import weights as tw

    raw_dir = tempfile.mkdtemp(prefix="tinker-raw-")
    print(f"\nDownloading Tinker checkpoint archive to {raw_dir} ...")
    extracted = tw.download(tinker_path=save.path, output_dir=raw_dir)
    print(f"  extracted to: {extracted}")

    out_dir = tempfile.mkdtemp(prefix="tinker-peft-") + "-out"
    print(f"\nBuilding PEFT LoRA adapter at {out_dir} ...")
    tw.build_lora_adapter(
        base_model=chosen,
        adapter_path=extracted,
        output_path=out_dir,
        trust_remote_code=True,
    )

    # --- 5. Inspect the exported adapter_config.json ------------------------
    cfg_path = os.path.join(out_dir, "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        print("\n=== adapter_config.json ===")
        print(json.dumps(cfg, indent=2))

        # The dealbreaker question: do the target_modules include MoE experts?
        targets = cfg.get("target_modules") or []
        moe_hits = [t for t in targets if "expert" in t.lower()]
        print(f"\ntarget_modules count: {len(targets)}")
        print(f"target_modules with 'expert' in name: {len(moe_hits)}  {moe_hits}")
        if moe_hits:
            print("\n✅ GREEN: Tinker's adapter includes MoE expert LoRAs. Phase 1 can proceed.")
        else:
            print("\n⚠️ YELLOW: no MoE expert modules in target_modules. Dgxchen's "
                  "Unsloth recipe trains those too. Decide whether to:\n"
                  "  (a) accept reduced LoRA capacity (proceed),\n"
                  "  (b) ask Tinker support if MoE LoRA is supported via custom config, or\n"
                  "  (c) drop Tinker.")
    else:
        print(f"\nFAIL: adapter_config.json was not produced at {cfg_path}")
        return 2

    print(f"\nPhase 0 finished. Output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
