# DiffPPO: Combining Diffusion Models with PPO to Improve Sample Efficiency and Exploration in Reinforcement Learning
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2409.01427v4-b31b1b.svg)](https://arxiv.org/abs/2409.01427v4)
## Overview

**DiffPPO** is a reinforcement learning framework that integrates diffusion models with Proximal Policy Optimization (PPO) to enhance sample efficiency and exploration capabilities. This project, implemented using the [robomimic](https://robomimic.github.io/) framework, utilizes the [D4RL](https://robomimic.github.io/docs/datasets/d4rl.html) dataset for experiments, demonstrating improved performance in environments with limited data.
![image](https://github.com/TianciGao/PPO/blob/main/%E6%9C%AA%E5%91%BD%E5%90%8D%E7%BB%98%E5%9B%BE%20(2).png)

## Training Artifacts

All training datasets, pretrained models, training logs, and videos can be accessed through the following Google Drive link:

[Google Drive - DiffPPO Training Artifacts](https://drive.google.com/drive/folders/1OhC2U6xYehcEmxVHKvi483HxJtzhQ3g4)

## Citation

If you find this project useful for your research, please consider citing our work:

**Paper**: Enhancing Sample Efficiency and Exploration in Reinforcement Learning through the Integration of Diffusion Models and Proximal Policy Optimization 

**Authors**: Tianci Gao, Dmitriev D. Dmitry, Neusypin A. Konstantin, Bo Yang, Shengren Rao

**Year**: 2024  

**Link**: https://arxiv.org/pdf/2409.01427v4

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
```

## Getting Started

### Prerequisites

To get started with DiffPPO, ensure that you have the following software installed:

- Python 3.8
- Conda (optional, but recommended for managing environments)

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/DiffPPO.git
    cd DiffPPO
    ```

2. Create and activate a Python virtual environment:

    ```bash
    conda create -n diffppo_env python=3.8
    conda activate diffppo_env
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

### Dataset

The project utilizes the [D4RL](https://robomimic.github.io/docs/datasets/d4rl.html) dataset. You can download the dataset using the provided script:

```bash
bash scripts/download_dataset.sh
```

Alternatively, you can refer to the [D4RL documentation](https://robomimic.github.io/docs/datasets/d4rl.html) for more details.

## Usage

### Training

To train the model, use the following command:

```bash
python scripts/train.py --config configs/PPO.json
```

### Evaluation

After training, evaluate the model's performance using:

```bash
python scripts/evaluate.py --model-path models/my_trained_model.pth
```

### Visualization

Visualize the training results with:

```bash
python scripts/visualize_results.py --log-dir logs/
```

## Results

The experiments conducted in this project demonstrate that integrating diffusion models to generate synthetic trajectories significantly enhances the sample efficiency and exploration capabilities of the PPO algorithm. Below is an example of the cumulative rewards achieved across different tasks:

![Experimental Results](https://github.com/user-attachments/assets/da4c862f-4698-46fe-9137-d09dfe1dd51c)


## Contribution

We welcome contributions to DiffPPO. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin new-feature`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
