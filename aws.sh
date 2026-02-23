ssh -i "gautam-diffusion-aws.pem" ubuntu@ec2-54-90-214-79.compute-1.amazonaws.com
sudo apt update && sudo apt install -y nvidia-driver-580 nvidia-utils-580 python3.12-venv
nvidia-smi
mkdir hw4 && cd hw4
git clone https://github.com/Gautam-Diwan/Diffusion-and-Flow-Matching.git
cd Diffusion-and-Flow-Matching
./setup-uv.sh
source .venv-cuda121/bin/activate
python train.py --method mean_flow --config configs/mean_flow.yaml
python sample.py --method mean_flow --config configs/mean_flow.yaml
python scripts/export_celeba_images.py --cache-dir ./data/celeba --output-dir ./data/celeba_images --split train
# Evaluate metrics (FID/KID): generates samples from checkpoint then runs torch-fidelity
./scripts/evaluate_torch_fidelity.sh \
  --method mean_flow \
  --checkpoint logs/mean_flow_20260222_195259/checkpoints/mean_flow_0025000.pt \
  --num-steps 1
# Or on Modal:
# ./scripts/evaluate_modal_torch_fidelity.sh --method mean_flow --checkpoint logs/mean_flow_20260222_195259/checkpoints/mean_flow_0025000.pt --num-steps 1 --metrics fid,kid