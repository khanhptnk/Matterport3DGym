# Matterport3DGym
A gym-like implementation of the Matterport3D simulator (Anderson et al., 2018).

# Run experiments

1. Install Docker Image and Build the Matterport3D simulator.
2. Download data: `bash download_data.sh`.
3. Download [PREVALENT pretrained models](https://drive.google.com/drive/folders/1sW2xVaSaciZiQ7ViKzm_KbrLD_XvOq5y) (`pytorch_model.bin`) and put it in `code/tasks/R2R/pretrain_models/PREVALENT`.
4. Inside `code/tasks/R2R`, train model: `python train.py -config configs/dagger_PREVALENT.yaml`. Models and log file will be saved to `code/task/R2R/experiments/dagger_PREVALENT` (the log file will be named `run.log`). You can customize the save folder by giving a different `-name` flag value. 
5. To evaluate a saved model: `python evaluate.py -config configs/dagger_PREVALENT.yaml -nav_agent.model.load_from $PATH_TO_MODEL.ckpt`. 



