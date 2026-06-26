# Reinforcement Learning — Taxi & Pong with DQN

Deep Q-Network (DQN) agents for two Gymnasium environments, built from scratch in PyTorch.

## Notebooks

| Notebook | Environment | What it does |
|---|---|---|
| [`QLearning_Taxi.ipynb`](QLearning_Taxi.ipynb) | **Taxi** (500 discrete states) | Tabular Q-learning with four hyperparameter search strategies compared head-to-head: manual tuning, grid search, random search, and Hyperband (Optuna). All four reach 100% success; Hyperband finds a competitive config at ~¼ of grid search's compute cost. |
| [`DQN_Taxi.ipynb`](DQN_Taxi.ipynb) | **Taxi** (500 discrete states) | DQN with one-hot state encoding, experience replay, and a target network. Plus a hyperparameter study (network size & regularization → RL hyperparameters) and a replay-buffer × target-network ablation. Solves Taxi at 100% success. |
| [`DQN_Pong.ipynb`](DQN_Pong.ipynb) | **Atari Pong** (raw pixels) | Pixel-based DQN — Nature-style CNN + 4-frame stacking, **Double DQN**, experience replay, target network, and checkpointing. Tuned for a free Colab T4 GPU. |

## Setup

### Option A — Google Colab (recommended, especially for Pong)
Open a notebook in Colab and run the first (install) cell — it installs everything it needs.
For Pong, switch on a GPU first: **Runtime → Change runtime type → T4 GPU**.

### Option B — Local
```bash
pip install -r requirements.txt
jupyter notebook
```

> **Atari ROMs:** bundled with `ale-py >= 0.8`, so the install above is usually enough.
> If you ever see a "ROM not found" error:
> ```bash
> pip install "autorom[accept-rom-license]" && AutoROM --accept-license
> ```

## Running

- **Q-Learning Taxi** — runs on **CPU** in a few minutes. Four parts (manual → grid search → random search → Hyperband); run top to bottom.
- **DQN Taxi** — runs on **CPU** in a few minutes. Run top to bottom; a roadmap table at the top links every section (build → hyperparameter study → ablation → wrap-up).
- **Pong** — needs a **GPU** and **hours** (~1–2M frames; positive scores around ~1–1.5M).
  1. Run the **install** cell and the **smoke-test** cell first — the observation must print `(4, 84, 84)`.
  2. Then run training. It **checkpoints** periodically, so you can resume after a Colab disconnect (uncomment the resume block).
  - On Apple Silicon it will use the **MPS** GPU automatically; on Colab it uses CUDA.

## Videos

### Q-Learning Taxi

<video src="videos/qlearning_taxi.mp4" controls width="720"></video>

[Open the Taxi video](videos/qlearning_taxi.mp4)

### DQN Pong

<video src="videos/pong.mp4" controls width="720"></video>

[Open the Pong video](videos/pong.mp4)

## Requirements

See [`requirements.txt`](requirements.txt). Core stack: **PyTorch**, **Gymnasium** (`toy-text` for Taxi, `atari` for Pong), NumPy, Matplotlib, pandas, OpenCV, imageio, **Optuna** (Hyperband search in Q-Learning Taxi), Plotly.
