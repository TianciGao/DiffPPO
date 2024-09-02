# DiffPPO: Combining Diffusion Models with PPO to Improve Sample Efficiency and Exploration in Reinforcement Learning

## Overview

**DiffPPO** is a reinforcement learning framework that integrates diffusion models with Proximal Policy Optimization (PPO) to enhance sample efficiency and exploration capabilities. This project, implemented using the [robomimic](https://robomimic.github.io/) framework, utilizes the [D4RL](https://robomimic.github.io/docs/datasets/d4rl.html) dataset for experiments, demonstrating improved performance in environments with limited data.
![image](https://github.com/user-attachments/assets/432e6712-ddbf-476f-9217-bfbb705162f1)

## Citation

If you find this project useful for your research, please consider citing our work:

**Paper Title**: Combining Diffusion Models with PPO to Improve Sample Efficiency and Exploration in Reinforcement Learning  
**Authors**: Tianci Gao, Dmitry, Bo Yang, Shengren Rao, Lao Nie  
**Year**: 2023  
**Link**: [arXiv Link](#) (Replace with the actual link)

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

- Python 3.8+
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

## Training Artifacts

All training datasets, pretrained models, training logs, and videos can be accessed through the following Google Drive link:

[Google Drive - DiffPPO Training Artifacts](https://drive.google.com/drive/folders/1OhC2U6xYehcEmxVHKvi483HxJtzhQ3g4)

## Contribution

We welcome contributions to DiffPPO. If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b new-feature`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin new-feature`).
5. Create a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
[1]	Wang, X., Wang, S., Liang, X., Zhao, D., Huang, J., Xu, X., ... & Miao, Q. (2022). Deep reinforcement learning: A survey. IEEE Transactions on Neural Networks and Learning Systems, 35(4), 5064-5078.
[2]	Schulman, J. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
[3]	Wang, Y., He, H., & Tan, X. (2020, August). Truly proximal policy optimization. In Uncertainty in artificial intelligence (pp. 113-122). PMLR.
[4]	Sun, Y., Yuan, X., Liu, W., & Sun, C. (2019, November). Model-based reinforcement learning via proximal policy optimization. In 2019 Chinese Automation Congress (CAC) (pp. 4736-4740). IEEE.
[5]	Zhu, Z., Zhao, H., He, H., Zhong, Y., Zhang, S., Yu, Y., & Zhang, W. (2023). Diffusion models for reinforcement learning: A survey. arXiv preprint arXiv:2311.01223.
[6]	Li, D., Xie, W., Wang, Z., Lu, Y., Li, Y., & Fang, L. (2024). FedDiff: Diffusion model driven federated learning for multi-modal and multi-clients. IEEE Transactions on Circuits and Systems for Video Technology.
[7]	Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34, 8780-8794.
[8]	Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34, 8780-8794.
[9]	Macaluso, G., Sestini, A., & Bagdanov, A. D. (2024, February). Small Dataset, Big Gains: Enhancing Reinforcement Learning by Offline Pre-Training with Model-Based Augmentation. In Computer Sciences & Mathematics Forum (Vol. 9, No. 1, p. 4). MDPI.
[10]	Wang, Z., Wang, C., Dong, Z., & Ross, K. (2023). Pre-training with Synthetic Data Helps Offline Reinforcement Learning. arXiv preprint arXiv:2310.00771.
[11]	Macaluso, G., Sestini, A., & Bagdanov, A. D. (2024, February). Small Dataset, Big Gains: Enhancing Reinforcement Learning by Offline Pre-Training with Model-Based Augmentation. In Computer Sciences & Mathematics Forum (Vol. 9, No. 1, p. 4). MDPI.
[12]	Ball, P. J., Smith, L., Kostrikov, I., & Levine, S. (2023, July). Efficient online reinforcement learning with offline data. In International Conference on Machine Learning (pp. 1577-1594). PMLR.
[13]	Wang, Z., Wang, C., Dong, Z., & Ross, K. (2023). Pre-training with Synthetic Data Helps Offline Reinforcement Learning. arXiv preprint arXiv:2310.00771.
[14]	Ho, J., & Ermon, S. (2016). Generative adversarial imitation learning. In Advances in Neural Information Processing Systems (NeurIPS) (pp. 4565-4573).
[15]	Kurach, K., Lucic, M., Zhai, X., Michalski, M., & Gelly, S. (2019). A large-scale study on regularization and normalization in GANs. In International Conference on Machine Learning (ICML).
[16]	Hafner, D., Lillicrap, T., Norouzi, M., & Ba, J. (2019). Learning latent dynamics for planning from pixels. In International Conference on Machine Learning (ICML).
[17]	Lee, K., Lu, K., Zhao, T., & Batra, D. (2020). Model-based reinforcement learning for offline RL tasks. In Advances in Neural Information Processing Systems (NeurIPS).
[18]	Rajeswar, S., Goyal, A., & Bengio, Y. (2023). Enhancing online RL algorithms with synthetic experience replay. In Proceedings of the International Conference on Learning Representations (ICLR).
[19]	Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. In Advances in Neural Information Processing Systems (NeurIPS).
[20]	Janner, M., Lu, Y., & Levine, S. (2022). Planning in latent space using diffusion models for complex tasks. In International Conference on Machine Learning (ICML).
[21]	Kostrikov, I., Yarats, D., & Fergus, R. (2020). Image augmentation is all you need: Regularizing deep reinforcement learning from pixels. In Proceedings of the 38th International Conference on Machine Learning (ICML).
[22]	Laskin, M., Srinivas, A., & Abbeel, P. (2020). CURL: Contrastive unsupervised representations for reinforcement learning. In Proceedings of the 37th International Conference on Machine Learning (ICML).
[23]	Nichol, A., & Dhariwal, P. (2021). "Improved Denoising Diffusion Probabilistic Models." In Proceedings of the 38th International Conference on Machine Learning (ICML).
[24]	Dhariwal, P., & Nichol, A. (2021). "Diffusion Models Beat GANs on Image Synthesis." In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS).
[25]	Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347.
[26]	Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T., Harley, T., ... & Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Proceedings of the 33rd International Conference on Machine Learning (ICML).
[27]	Levine, S., Kumar, A., Tucker, G., & Fu, J. (2020). Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems. arXiv preprint arXiv:2005.01643.
[28]	Fujimoto, S., Meger, D., & Precup, D. (2019). Off-Policy Deep Reinforcement Learning without Exploration. Proceedings of the 36th International Conference on Machine Learning (ICML).
[29]	Agarwal, R., Schuurmans, D., & Norouzi, M. (2021). An Optimistic Perspective on Offline Reinforcement Learning. Proceedings of the 38th International Conference on Machine Learning (ICML).
[30]	Zhan, Y., Lu, Y., & Zhao, J. (2021). Online Reinforcement Learning with Offline Data: Applications and Challenges. Proceedings of the 38th International Conference on Machine Learning (ICML).
[31]	Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2021). From Offline Reinforcement Learning to Online Learning. arXiv preprint arXiv:2106.01345.
[32]	Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd Edition). MIT Press.
[33]	Bellman, R. (1957). Dynamic Programming. Princeton University Press.
[34]	Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation. In Proceedings of the 12th International Conference on Neural Information Processing Systems (NIPS).
[35]	Konda, V. R., & Tsitsiklis, J. N. (2000). Actor-Critic Algorithms. In Proceedings of the 13th Conference on Neural Information Processing Systems (NIPS).
[36]	Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). High-Dimensional Continuous Control Using Generalized Advantage Estimation. arXiv preprint arXiv:1506.02438.
[37]	Nichol, A., & Dhariwal, P. (2021). Improved Denoising Diffusion Probabilistic Models. In Proceedings of the 38th International Conference on Machine Learning (ICML).
[38]	Dhariwal, P., & Nichol, A. (2021). Diffusion Models Beat GANs on Image Synthesis. In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS).
[39]	Janner, M., Fu, J., & Levine, S. (2022). Planning in Latent Space using Diffusion Models. In International Conference on Learning Representations (ICLR).

