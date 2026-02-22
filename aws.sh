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
python evaluate_torch_fidelity.sh mean_flow checkpoints/mean_flow/mean_flow_final.pt
python evaluate_modal_torch_fidelity.sh mean_flow checkpoints/mean_flow/mean_flow_final.pt