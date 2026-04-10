## Overview

CtrlBench-Rec is an evolutionary multi-agent
framework with three modules: Initialization, Dynamic Interaction, and Collaborative Fusion. Operating as a closed-loop system, it iterates through initialization, policy alignment, and agent
fusion to accelerate group exploration and cultivate elite agents.
The central objective is to transform novice agents into a refined
set of high-capability super probes that serve as a standardized
benchmark for system controllability. The framework operates in
two sequential phases: (1) Training phase, refining
super probes through interaction and fusion; and (2) Inference
and evaluation phase, deploying the probes for multi-dimensional
controllability assessments.

## Project Structure

```text

├── data/                        # Datasets (ML-1M, preprocessed Amazon Toys & Games)
├── models/                      # Sequential recommendation model definitions (e.g., SASRec, BGE)
├── rec_models/                  # Non-sequential recommendation models (e.g., twhin-bert-base)
├── generated_user_profile/      # User profiles generated at different stages
├── tool/                        # Data loaders and embedding processors
├── runner/                      # Scripts for training, inference, and evaluation (e.g., epoch.py, evaluation.py)
└── requirements.txt             # Project dependencies
```

## Framework and workflow

![FrameWork](readme_images/framework.png)

**Phase I: Evolutionary Training**
1. **Multi-Agent Initialization** :Extract static attributes and dynamic trajectories from raw datasets like ML1M.Instantiate agents with a profile expert, an LLM-based decision engine, and tool-calling modules.
2. **Environment Interaction & Behavior Alignment** :Synchronize the Black-Box system's state with the agent's persona by injecting a continuous stream of profile-aligned interaction behaviors.
3. **Multi-Agent Strategy Fusion** :Group agents via K-means clustering to facilitate intra-cluster discussions, followed by a fusion expert integrating these records and profiles into new Super Probes.

**Phase II: Inference & Evaluation**
1. **Interaction & Behavior Acquisition** :Execute multi-turn interactions with the Black-Box recommender to generate a profile-aligned behavioral stream for the Super Probes.
2. **Systematic Evaluation**

## 🚀 Quick Start

### Prerequisites

| Tool | Version | Description | Check Installation |
|------|---------|-------------|-------------------|
| **Python** | 3.10    | Backend runtime | `python --version` |

### Installation & Setup
1.**Environment Configuration**\
Run the following command in your terminal to install the necessary dependencies:
```bash
pip install -r requirements.txt
```
2.**Load Bert Encoder**\
Execute the script to load the twhin-bert-base model:
```bash
python runner/load_twhin_bert.py
```

3.**Configuration (API Key)**\
To use the DeepSeek LLM features, you need to provide your API key from https://platform.deepseek.com/api_keys. You can pass it as an environment variable at runtime without permanently modifying your system settings.

For Linux / macOS / WSL
Prefix your command with the variable:

```bash
DEEPSEEK_API_KEY="your_api_key_here" python ../runner/user_profile_initialize.py
```

For Windows (PowerShell)
In PowerShell, variables must be set for the current session before running the script:

```PowerShell
$env:DEEPSEEK_API_KEY="your_api_key_here"; python ../runner/user_profile_initialize.py
```
You can also download this model from:https://huggingface.co/Twitter/twhin-bert-base;
After downloading,place it in the "../rec_models/" directory.

### Experiments

We provide experiments using the SASRec recommendation model on the ML-1M dataset, centered around the Task 1 Target Content Discovery Analysis.

**Phase I: Evolutionary Training**

1. **Multi-Agent Initialization** :Initialize the agent metadata
``` bash
python runner/user_profile_initialize.py
```
2. **Interaction & Fusion** :Update the entry point in ***epoch.py*** to call ***runner.epoch.sasrec_ml1m_merge***, then run the script.
``` bash
python runner/epoch.py
```
**Phase II: Evolutionary Training**

1. **Interaction & Behavior Acquisition** :Update the entry point in ***epoch.py*** to call ***runner.epoch.sasrec_ml1m_debate_epoch20***, then run the script.
``` bash
python runner/epoch.py
```

2. **Systematic Evaluation** : Invoke ***runner.evaluation.compare_two_profile***, modify the original and evaluation profile paths, and run ***evaluation.py*** for results.
``` bash
python runner/evaluation.py
```