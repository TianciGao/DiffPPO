<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DiffPPO README</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        code {
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        a {
            color: #3498db;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
</head>
<body>
    <h1>DiffPPO: Combining Diffusion Models with PPO to Improve Sample Efficiency and Exploration in Reinforcement Learning</h1>

    <h2>Overview</h2>
    <p><strong>DiffPPO</strong> is an advanced reinforcement learning framework that synergizes diffusion models with Proximal Policy Optimization (PPO) to significantly boost sample efficiency and exploration capabilities. Leveraging the <a href="https://robomimic.github.io/">robomimic</a> framework, this project employs the <a href="https://robomimic.github.io/docs/datasets/d4rl.html">D4RL</a> dataset for rigorous experimentation, showcasing substantial performance gains in data-constrained environments.</p>
    <img src="https://github.com/TianciGao/PPO/blob/main/%E6%9C%AA%E5%91%BD%E5%90%8D%E7%BB%98%E5%9B%BE%20(2).png" alt="Project Overview">

    <h2>Training Artifacts</h2>
    <p>All training datasets, pretrained models, training logs, and demonstration videos are available via the following Google Drive link:</p>
    <p><a href="https://drive.google.com/drive/folders/1OhC2U6xYehcEmxVHKvi483HxJtzhQ3g4">DiffPPO Training Artifacts - Google Drive</a></p>

    <h2>Citation</h2>
    <p>If you find DiffPPO beneficial for your research, please cite our work as follows:</p>
    <p><strong>Paper</strong>: Enhancing Sample Efficiency and Exploration in Reinforcement Learning through the Integration of Diffusion Models and Proximal Policy Optimization<br>
    <strong>Authors</strong>: Tianci Gao, Dmitriev D. Dmitry, Neusypin A. Konstantin, Bo Yang, Shengren Rao<br>
    <strong>Year</strong>: 2024<br>
    <strong>Link</strong>: <a href="https://arxiv.org/pdf/2409.01427v4">https://arxiv.org/pdf/2409.01427v4</a></p>

    <h2>Project Structure</h2>
    <pre>
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
    </pre>

    <h2>Getting Started</h2>
    <h3>Prerequisites</h3>
    <p>To begin using DiffPPO, please ensure the following software is installed on your system:</p>
    <ul>
        <li><strong>Python 3.8</strong></li>
        <li><strong>Conda</strong> (optional, but highly recommended for environment management)</li>
    </ul>

    <h3>Installation</h3>
    <ol>
        <li><strong>Clone the repository</strong>:<pre><code>git clone https://github.com/yourusername/DiffPPO.git
cd DiffPPO</code></pre></li>
        <li><strong>Create and activate a Python virtual environment</strong>:<pre><code>conda create -n diffppo_env python=3.8
conda activate diffppo_env</code></pre></li>
        <li><strong>Install the required dependencies</strong>:<pre><code>pip install -r requirements.txt</code></pre></li>
    </ol>

    <h3>Dataset</h3>
    <p>DiffPPO leverages the <a href="https://robomimic.github.io/docs/datasets/d4rl.html">D4RL</a> dataset. To download the dataset, execute the following script:</p>
    <pre><code>bash scripts/download_dataset.sh</code></pre>
    <p>For further information, please consult the <a href="https://robomimic.github.io/docs/datasets/d4rl.html">D4RL documentation</a>.</p>

    <h2>Usage</h2>
    <h3>Training</h3>
    <p>To initiate model training, execute the following command:</p>
    <pre><code>python scripts/train.py --config configs/PPO.json</code></pre>

    <h3>Evaluation</h3>
    <p>Post-training, assess the model's performance with the following command:</p>
    <pre><code>python scripts/evaluate.py --model-path models/my_trained_model.pth</code></pre>

    <h3>Visualization</h3>
    <p>To visualize the training outcomes, use the following command:</p>
    <pre><code>python scripts/visualize_results.py --log-dir logs/</code></pre>

    <h2>Results</h2>
    <p>Our experimental results substantiate that the incorporation of diffusion models for generating synthetic trajectories markedly improves the sample efficiency and exploration prowess of the PPO algorithm. An illustrative example of the cumulative rewards across various tasks is presented below:</p>
    <img src="https://github.com/user-attachments/assets/da4c862f-4698-46fe-9137-d09dfe1dd51c" alt="Experimental Results">

    <h2>Contribution</h2>
    <p>Contributions to DiffPPO are highly encouraged. To contribute, please adhere to the following procedure:</p>
    <ol>
        <li>Fork the repository.</li>
        <li>Create a new branch: <code>git checkout -b new-feature</code></li>
        <li>Commit your changes: <code>git commit -am 'Add new feature'</code></li>
        <li>Push to the branch: <code>git push origin new-feature</code></li>
        <li>Submit a Pull Request.</li>
    </ol>

    <h2>License</h2>
    <p>DiffPPO is released under the MIT License. For detailed terms, please refer to the <a href="LICENSE">LICENSE</a> file.</p>
</body>
</html>
