# DiffPPO: Combining Diffusion Models with PPO to Improve Sample Efficiency and Exploration in Reinforcement Learning

## Overview
**DiffPPO** is a reinforcement learning framework that integrates diffusion models with Proximal Policy Optimization (PPO) to enhance sample efficiency and exploration. Built using the [robomimic](https://robomimic.github.io/) framework and the [D4RL](https://robomimic.github.io/docs/datasets/d4rl.html) dataset, this project demonstrates improved performance in environments with limited data.

<p align="center">
  <img src="https://github.com/TianciGao/PPO/blob/main/%E6%9C%AA%E5%91%BD%E5%90%8D%E7%BB%98%E5%9B%BE%20(2).png" alt="DiffPPO Overview" width="400"/>
</p>

## Training Artifacts
All training datasets, pretrained models, training logs, and videos are available in the following Google Drive folder:

[Google Drive - DiffPPO Training Artifacts](https://drive.google.com/drive/folders/1OhC2U6xYehcEmxVHKvi483HxJtzhQ3g4)

## Citation
If you find **DiffPPO** helpful for your research, please consider citing our work:

- **Paper**: *Enhancing Sample Efficiency and Exploration in Reinforcement Learning through the Integration of Diffusion Models and Proximal Policy Optimization*  
- **Authors**: Tianci Gao, Dmitriev D. Dmitry, Neusypin A. Konstantin, Bo Yang, Shengren Rao  
- **Year**: 2024  
- **Link**: [https://arxiv.org/pdf/2409.01427v4](https://arxiv.org/pdf/2409.01427v4)

## Project Structure
```plaintext
├── datasets/                   # Directory for storing datasets
├── models/                     # Pretrained models
├── scripts/                    # Scripts for training, evaluation, and visualization
│   ├── train.py                # Script for training the model
│   ├── evaluate.py             # Script for evaluating the model
│   └── visualize_results.py    # Script for visualizing results
├── notebooks/                  # Jupyter Notebooks for analysis and visualization
├── configs/                    # Configuration files
│   └── PPO.json                # Configuration for the PPO algorithm
├── README.md                   # Project documentation
└── requirements.txt            # Python dependencies
Getting Started
Prerequisites
Python 3.8
Conda (recommended for environment management)
Installation
Clone the repository:

git clone https://github.com/yourusername/DiffPPO.git
cd DiffPPO
Create and activate a virtual environment (optional but recommended):

conda create -n diffppo_env python=3.8
conda activate diffppo_env
Install required dependencies:

pip install -r requirements.txt
Dataset
DiffPPO utilizes the D4RL dataset. You can download it using the included script:

bash scripts/download_dataset.sh
Alternatively, refer to the D4RL documentation for more information.

Usage
Training
Train the model using:

python scripts/train.py --config configs/PPO.json
Evaluation
Evaluate the model with:


python scripts/evaluate.py --model-path models/my_trained_model.pth
Visualization
Visualize the training results:


python scripts/visualize_results.py --log-dir logs/
Results
Experiments indicate that leveraging diffusion models to generate synthetic trajectories significantly improves both sample efficiency and exploration capabilities for PPO. Below is an example of cumulative rewards across various tasks:

<p align="center"> <img src="https://github.com/user-attachments/assets/da4c862f-4698-46fe-9137-d09dfe1dd51c" alt="Experimental Results" width="600"/> </p>
Contribution
We welcome contributions to DiffPPO. To contribute:

Fork the repository.
Create a new branch (git checkout -b new-feature).
Commit your changes (git commit -am "Add new feature").
Push to the branch (git push origin new-feature).
Open a Pull Request.
License
This project is licensed under the MIT License. Please see the LICENSE file for details.
