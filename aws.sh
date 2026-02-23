ssh -i "gautam-diffusion-aws.pem" ubuntu@ec2-54-90-214-79.compute-1.amazonaws.com
sudo apt update && sudo apt install -y nvidia-driver-580 nvidia-utils-580 python3.12-venv python3.12-dev build-essential tmux
nvidia-smi
mkdir hw4 && cd hw4
git clone https://github.com/Gautam-Diwan/Diffusion-and-Flow-Matching.git
cd Diffusion-and-Flow-Matching
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
./setup-uv.sh
source .venv-cuda121/bin/activate
tmux new -s training_session

# Mean Flow
python train.py --method mean_flow --config configs/mean_flow.yaml
python sample.py --method mean_flow --config configs/mean_flow.yaml

python scripts/export_celeba_images.py --cache-dir ./data/celeba --output-dir ./data/celeba_images --split train

# Progressive Distillation
python train.py --method progressive_distillation --config configs/progressive_distillation_20_to_10.yaml
/scripts/evaluate_torch_fidelity.sh --checkpoint <distill_ckpt.pt> --method progressive_distillation --num-samples 1000 --num-steps 10 --metrics kid

# Optional: Sample from Progressive Distillation
python sample.py --method progressive_distillation --config configs/progressive_distillation_20_to_10.yaml

# Progressive Distillation 10 to 5
python train.py --method progressive_distillation --config configs/progressive_distillation_10_to_5.yaml
/scripts/evaluate_torch_fidelity.sh --checkpoint <distill_ckpt.pt> --method progressive_distillation --num-samples 1000 --num-steps 10 --metrics kid

# Optional: Sample from Progressive Distillation 10 to 5
python sample.py --method progressive_distillation --config configs/progressive_distillation_10_to_5.yaml

tmux kill-session -t training_session
# Evaluate metrics (FID/KID): generates samples from checkpoint then runs torch-fidelity
./scripts/evaluate_torch_fidelity.sh \
  --method mean_flow \
  --checkpoint logs/mean_flow_20260222_195259/checkpoints/mean_flow_0025000.pt \
  --num-steps 1
  --metrics fid,kid
  --num-samples 1000
  --dataset-path ./data/celeba-subset/train/images
# Or on Modal:
# ./scripts/evaluate_modal_torch_fidelity.sh --method mean_flow --checkpoint logs/mean_flow_20260222_195259/checkpoints/mean_flow_0025000.pt --num-steps 1 --metrics fid,kid