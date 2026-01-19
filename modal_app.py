"""
Modal Configuration for CMU 10799 Diffusion Homework

Defines Modal environment and training functions for cloud GPU training.

See docs/QUICKSTART-MODAL.md for setup and usage instructions.

All parameters are read from config YAML files first, then overridden by command-line arguments.
"""

import modal

# =============================================================================
# Modal App Definition
# =============================================================================

# Create the Modal app
app = modal.App("cmu-10799-diffusion")

# Define the container image with all dependencies
# This mirrors the CPU-only environment (environments/environment-cpu.yml)
# but installs GPU-enabled PyTorch automatically on Modal's GPU machines
image = (
    modal.Image.debian_slim(python_version="3.11").pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "pillow>=9.0.0",
        "pyyaml>=6.0",
        "einops>=0.6.0",
        "tqdm>=4.64.0",
        "scipy>=1.9.0",
        "wandb>=0.15.0",
        "datasets>=2.0.0",  # For HuggingFace Hub dataset loading
        "torch-fidelity>=0.3.0",  # Comprehensive evaluation metrics
    )
    # Copy the local project directory into the image
    .add_local_dir(
        ".",
        "/root",
        ignore=[
            ".git",
            ".venv*",
            "venv",
            "__pycache__",
            "logs",
            "checkpoints",
            "*.md",
            "docs",
            "environments",
            "notebooks",
        ],
    )
)

# Create a persistent volume for checkpoints and data
volume = modal.Volume.from_name("cmu-10799-diffusion-data", create_if_missing=True)

# =============================================================================
# Training Function
# =============================================================================


def _train_impl(
    method: str,
    config_path: str,
    resume_from: str,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    """
    Internal training implementation.

    Reads config from YAML file, applies command-line overrides.
    """
    import os
    import sys
    import yaml
    import tempfile
    import subprocess

    sys.path.insert(0, "/root")

    # Load config
    config_tag = method
    if config_path is None:
        config_path = f"/root/configs/{method}.yaml"
    else:
        config_path = f"/root/{config_path}"
        config_tag = os.path.splitext(os.path.basename(config_path))[0]

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get num_gpus from config
    config_device = config["infrastructure"].get("device", "cuda")
    num_gpus = config["infrastructure"].get("num_gpus", 1 if config_device == "cuda" else 0)
    if num_gpus is None:
        num_gpus = 1 if config_device == "cuda" else 0

    # Read from_hub from config
    from_hub = config["data"].get("from_hub", False)

    # Apply command-line overrides if provided
    if num_iterations is not None:
        config["training"]["num_iterations"] = num_iterations
    if batch_size is not None:
        config["training"]["batch_size"] = batch_size
    if learning_rate is not None:
        config["training"]["learning_rate"] = learning_rate

    # Set Modal-specific paths
    config["data"]["repo_name"] = "electronickale/cmu-10799-celeba64-subset"
    # Set root path for both modes:
    # - from_hub=true: checks for cached Arrow format first, then downloads from HF
    # - from_hub=false: looks for traditional folder structure (train/images/)
    config["data"]["root"] = "/data/celeba"
    config["checkpoint"]["dir"] = f"/data/checkpoints/{config_tag}"
    config["logging"]["dir"] = f"/data/logs/{config_tag}"

    # Create directories
    os.makedirs(config["checkpoint"]["dir"], exist_ok=True)
    os.makedirs(config["logging"]["dir"], exist_ok=True)

    resume_path = f"/data/{resume_from}" if resume_from else None

    # Use torchrun for multi-GPU, direct import for single GPU
    if num_gpus > 1:
        temp_config_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as temp_file:
                yaml.safe_dump(config, temp_file)
                temp_config_path = temp_file.name

            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={num_gpus}",
                "/root/train.py",
                "--method",
                method,
                "--config",
                temp_config_path,
            ]
            if resume_path:
                cmd.extend(["--resume", resume_path])
            if overfit_single_batch:
                cmd.append("--overfit-single-batch")

            subprocess.run(cmd, check=True)
        finally:
            if temp_config_path and os.path.exists(temp_config_path):
                os.remove(temp_config_path)
    else:
        from train import train as run_training

        run_training(
            method_name=method,
            config=config,
            resume_path=resume_path,
            overfit_single_batch=overfit_single_batch,
        )

    volume.commit()
    return f"Training complete! Checkpoints saved to /data/checkpoints/{method}"


# Create training functions for different GPU counts
@app.function(
    image=image,
    gpu="L40S:1",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_1gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:2",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_2gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:3",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_3gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:4",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_4gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:5",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_5gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:6",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_6gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:7",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_7gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


@app.function(
    image=image,
    gpu="L40S:8",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
    secrets=[modal.Secret.from_name("wandb-api-key")],
)
def train_8gpu(
    method: str = "ddpm",
    config_path: str = None,
    resume_from: str = None,
    num_iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    overfit_single_batch: bool = False,
):
    return _train_impl(
        method,
        config_path,
        resume_from,
        num_iterations,
        batch_size,
        learning_rate,
        overfit_single_batch,
    )


# Map GPU counts to functions
TRAIN_FUNCTIONS = {
    1: train_1gpu,
    2: train_2gpu,
    3: train_3gpu,
    4: train_4gpu,
    5: train_5gpu,
    6: train_6gpu,
    7: train_7gpu,
    8: train_8gpu,
}


# =============================================================================
# Sampling Function
# =============================================================================


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 3,  # 3 hours
    volumes={"/data": volume},
)
def sample(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    num_samples: int = None,
    num_steps: int = None,
):
    """
    Generate samples from a trained model.

    Uses sample.py via subprocess, similar to how training uses train.py.
    """
    import os
    import subprocess
    from datetime import datetime

    # Set up paths
    checkpoint_path = f"/data/{checkpoint}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/data/samples/{method}_{timestamp}.png"

    os.makedirs("/data/samples", exist_ok=True)

    # Build command to run sample.py
    cmd = [
        "python",
        "/root/sample.py",
        "--checkpoint",
        checkpoint_path,
        "--method",
        method,
        "--grid",
        "--output",
        output_path,
    ]

    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if num_steps is not None:
        cmd.extend(["--num_steps", str(num_steps)])

    subprocess.run(cmd, check=True)
    volume.commit()

    return f"Samples saved to {output_path}"


# =============================================================================
# Dataset Download Function
# =============================================================================


@app.function(
    image=image,
    timeout=60 * 60,  # 1 hour
    volumes={"/data": volume},
)
def download_dataset():
    """
    Download the dataset from HuggingFace Hub to Modal volume.

    Caches the dataset in Arrow format at /data/celeba. After downloading,
    training with from_hub=true will automatically use this cached version
    instead of redownloading.
    """
    import sys

    sys.path.insert(0, "/root")

    from datasets import load_dataset
    import os

    print("Downloading dataset from HuggingFace Hub...")
    dataset = load_dataset("electronickale/cmu-10799-celeba64-subset")

    # Save to volume in Arrow format
    os.makedirs("/data/celeba", exist_ok=True)
    dataset.save_to_disk("/data/celeba")

    volume.commit()

    print(f"Dataset cached to /data/celeba")
    print(f"Train set size: {len(dataset['train'])}")
    return "Dataset download complete! Training with from_hub=true with root = '/data/celeba' will now use this cached version."


# =============================================================================
# Evaluation Function (using torch-fidelity)
# =============================================================================


@app.function(
    image=image,
    gpu="L40S",
    timeout=60 * 60 * 8,  # 8 hours
    volumes={"/data": volume},
)
def evaluate_torch_fidelity(
    method: str = "ddpm",
    checkpoint: str = "checkpoints/ddpm/ddpm_final.pt",
    metrics: str = "fid,kid",
    num_samples: int = 5000,
    batch_size: int = 128,
    num_steps: int = None,
    override: bool = False,
):
    """
    Evaluate using torch-fidelity CLI.

    Uses the fidelity command to compute metrics directly.

    Args:
        method: 'ddpm'
        checkpoint: Path to checkpoint (relative to /data)
        metrics: Comma-separated: 'fid', 'kid', 'is' (default: 'fid,kid')
        num_samples: Number of samples to generate
        batch_size: Batch size
        num_steps: Sampling steps (optional)
        override: Force regenerate samples even if they exist
    """
    import sys
    import subprocess
    from pathlib import Path

    sys.path.insert(0, "/root")

    checkpoint_path = f"/data/{checkpoint}"

    # Put samples in same parent dir as checkpoint under samples/
    checkpoint_dir = Path(checkpoint_path).parent
    generated_dir = str(checkpoint_dir / "samples" / "generated")
    cache_dir = str(checkpoint_dir / "samples" / "cache")

    # Prepare dataset path for torch-fidelity
    # torch-fidelity needs actual image files, not Arrow format
    dataset_arrow_path = "/data/celeba"
    dataset_images_path = "/data/celeba_images"

    # Extract images from Arrow format if not already done
    import os

    if not os.path.exists(dataset_images_path):
        print("=" * 60)
        print("Extracting dataset images for torch-fidelity...")
        print("=" * 60)

        from datasets import load_from_disk

        dataset = load_from_disk(dataset_arrow_path)
        train_data = dataset["train"]

        os.makedirs(dataset_images_path, exist_ok=True)

        print(f"Extracting {len(train_data)} images...")
        for idx, item in enumerate(train_data):
            img = item["image"]
            img_path = os.path.join(dataset_images_path, f"{idx:06d}.png")
            img.save(img_path)

            if (idx + 1) % 1000 == 0:
                print(f"  Extracted {idx + 1}/{len(train_data)} images")

        volume.commit()
        print(f"Dataset images saved to {dataset_images_path}")
    else:
        print(f"Using cached dataset images at {dataset_images_path}")

    dataset_path = dataset_images_path

    # Step 1: Generate samples
    print("=" * 60)
    print("Step 1/2: Generating samples...")
    print("=" * 60)

    import os
    import shutil
    import glob

    # Check if samples already exist
    need_generation = True
    if os.path.exists(generated_dir) and not override:
        # Check for both png and jpg files
        existing_samples = (
            glob.glob(os.path.join(generated_dir, "*.png"))
            + glob.glob(os.path.join(generated_dir, "*.jpg"))
            + glob.glob(os.path.join(generated_dir, "*.jpeg"))
        )
        num_existing = len(existing_samples)

        if num_existing >= num_samples:
            print(f"Found {num_existing} existing samples (need {num_samples})")
            print("Skipping sample generation (use --override to force regeneration)")
            need_generation = False
        else:
            print(f"Found {num_existing} existing samples but need {num_samples}")
            print("Regenerating samples...")
            shutil.rmtree(generated_dir)
    elif os.path.exists(generated_dir) and override:
        print("Override flag set, regenerating samples...")
        shutil.rmtree(generated_dir)

    if need_generation:
        sample_cmd = [
            "python",
            "/root/sample.py",
            "--checkpoint",
            checkpoint_path,
            "--method",
            method,
            "--output_dir",
            generated_dir,
            "--num_samples",
            str(num_samples),
            "--batch_size",
            str(batch_size),
        ]

        if num_steps:
            sample_cmd.extend(["--num_steps", str(num_steps)])

        subprocess.run(sample_cmd, check=True)
        print(f"Generated {num_samples} samples to {generated_dir}")
    else:
        print(f"Using existing samples from {generated_dir}")

    # Step 2: Run fidelity
    print("\n" + "=" * 60)
    print("Step 2/2: Running torch-fidelity...")
    print("=" * 60)

    os.makedirs(cache_dir, exist_ok=True)

    fidelity_cmd = [
        "fidelity",
        "--gpu",
        "0",
        "--batch-size",
        str(batch_size),
        "--cache-root",
        cache_dir,
        "--input1",
        generated_dir,
        "--input2",
        dataset_path,
    ]

    if "fid" in metrics:
        fidelity_cmd.append("--fid")
    if "kid" in metrics:
        fidelity_cmd.append("--kid")
    if "is" in metrics or "isc" in metrics:
        fidelity_cmd.append("--isc")

    print(f"\nRunning command: {' '.join(fidelity_cmd)}\n")

    try:
        result = subprocess.run(fidelity_cmd, check=True, capture_output=True, text=True)
        volume.commit()
        return result.stdout
    except subprocess.CalledProcessError as e:
        # Print the error output to help debug
        print(f"\nError running fidelity command:")
        print(f"Command: {' '.join(fidelity_cmd)}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"\nStdout:\n{e.stdout}")
        if e.stderr:
            print(f"\nStderr:\n{e.stderr}")
        raise


# =============================================================================
# CLI Entry Points
# =============================================================================


@app.local_entrypoint()
def main(
    action: str = "train",
    method: str = "ddpm",
    config: str = None,
    checkpoint: str = None,
    iterations: int = None,
    batch_size: int = None,
    learning_rate: float = None,
    num_samples: int = None,
    num_steps: int = None,
    metrics: str = None,
    overfit_single_batch: bool = False,
    override: bool = False,
):
    """
    Main entry point for Modal CLI.

    See docs/QUICKSTART-MODAL.md for usage instructions.

    All parameters are read from config YAML files first, then overridden by command-line arguments.
    """
    if action == "download":
        result = download_dataset.remote()
        print(result)
    elif action == "train":
        # Read config to determine GPU count
        import yaml

        local_config_path = config or f"configs/{method}.yaml"
        with open(local_config_path, "r") as f:
            local_config = yaml.safe_load(f)

        # Get num_gpus from config
        config_device = local_config["infrastructure"].get("device", "cuda")
        num_gpus = local_config["infrastructure"].get(
            "num_gpus", 1 if config_device == "cuda" else 0
        )
        if num_gpus is None:
            num_gpus = 1 if config_device == "cuda" else 0

        # Get the appropriate training function
        train_fn = TRAIN_FUNCTIONS.get(num_gpus)
        if train_fn is None:
            raise ValueError(f"Unsupported num_gpus={num_gpus} in config. " f"Supported: 1-8")

        result = train_fn.remote(
            method=method,
            config_path=config,
            num_iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
            overfit_single_batch=overfit_single_batch,
        )
        print(result)
    elif action == "sample":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"
        result = sample.remote(
            method=method,
            checkpoint=checkpoint,
            num_samples=num_samples,
            num_steps=num_steps,
        )
        print(result)
    elif action == "evaluate" or action == "evaluate_torch_fidelity":
        if checkpoint is None:
            checkpoint = f"checkpoints/{method}/{method}_final.pt"

        eval_kwargs = {
            "method": method,
            "checkpoint": checkpoint,
            "override": override,
        }
        if metrics is not None:
            eval_kwargs["metrics"] = metrics
        if num_samples is not None:
            eval_kwargs["num_samples"] = num_samples
        if batch_size is not None:
            eval_kwargs["batch_size"] = batch_size
        if num_steps is not None:
            eval_kwargs["num_steps"] = num_steps

        result = evaluate_torch_fidelity.remote(**eval_kwargs)
        print(result)
    else:
        print(f"Unknown action: {action}")
        print("Valid actions: download, train, sample, evaluate")
