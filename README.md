# MARL Autonomous Driving System

## Overview

This project implements a **Multi-Agent Reinforcement Learning (MARL) Autonomous Driving System** using **Deep Q-Networks (DQN)** and the **highway-env** simulation environment.  
The system simulates multiple autonomous vehicles navigating an intersection while learning safe driving behavior through reinforcement learning.

The project includes:
- Multi-Agent DQN Training
- Autonomous Driving Simulation
- Streamlit Dashboard
- Dynamic Hyperparameter Configuration
- Real-Time Evaluation Metrics
- Traffic Signal Handling
- Safety Reward Mechanisms
- Plot Visualization and Analytics

---

# Features

- Multi-Agent Reinforcement Learning (3 DQN Agents)
- Independent Agent Configurations
- Streamlit-Based Interactive Dashboard
- Dynamic Hyperparameter Tuning
- Traffic Signal Awareness
- Reward Shaping Mechanism
- Collision Detection
- Lane Violation Monitoring
- Speed Analysis
- Front Vehicle Distance Monitoring
- Evaluation Metrics Visualization
- Training Simulation Recording

---

# Technologies Used

- Python
- PyTorch
- Streamlit
- OpenCV
- NumPy
- Matplotlib
- highway-env
- Gymnasium

---

# Project Structure

```bash
MARL-Autonomous-Driving/
│
├── app.py
├── main.py
├── evaluate.py
├── agent.py
├── model.py
├── plots.py
├── config.json
├── requirements.txt
├── README.md
│
├── models/
├── logs/
├── plots/
├── videos/
```

---

# System Architecture

The system architecture consists of:

1. Streamlit Dashboard
2. Config Generator
3. MARL Training Engine
4. Highway Environment
5. Multi-Agent DQN System
6. Evaluation Engine
7. Visualization Dashboard

---

# MARL Workflow

```text
User Inputs
     ↓
Streamlit Dashboard
     ↓
Config Generator
     ↓
MARL Training Engine
     ↓
Highway-Env Simulation
     ↓
Multi-Agent DQN Agents
     ↓
Reward Computation
     ↓
Evaluation Metrics
     ↓
Plots and Visualization
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yathinkumar03/MARL-Autonomous-Driving.git
```

## Move to Project Directory

```bash
cd MARL-Autonomous-Driving
```

## Create Virtual Environment

```bash
python -m venv venv
```

## Activate Virtual Environment

### Windows

```bash
venv\Scripts\activate
```

### Linux/Mac

```bash
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Streamlit Dashboard

```bash
python -m streamlit run app.py
```

---

# Dashboard Features

The Streamlit dashboard allows users to:

- Configure environment settings
- Configure individual agent parameters
- Start MARL training
- Run evaluation
- Visualize training metrics
- View agent-wise plots
- Monitor simulation output

---

# Agent Parameters

Each agent can independently configure:

- Learning Rate
- Gamma
- Epsilon
- Epsilon Decay
- Collision Penalty
- Lane Penalty
- Speed Reward
- Signal Reward
- Safe Distance Reward
- Overspeed Penalty
- Emergency Brake Penalty

---

# Evaluation Metrics

The system evaluates:

- Total Reward
- Average Reward
- Collision Rate
- Lane Violations
- Average Speed
- Front Vehicle Distance
- Training Stability

---

# Generated Outputs

The project generates:

- Reward Plots
- Moving Average Plots
- Collision Analysis
- Lane Violation Analysis
- Speed Analysis
- Front Distance Analysis
- Training Videos
- Saved Models

---

# DQN Architecture

The DQN model contains:

- Input State Layer
- Hidden Dense Layers
- ReLU Activation
- Output Q-Value Layer
- Experience Replay Buffer
- Target Network

---

# Future Enhancements

- PPO and SAC Algorithms
- Real-Time Multi-Agent Coordination
- Cloud Deployment
- TensorBoard Integration
- Real Vehicle Integration
- Advanced Traffic Signal Control
- Federated MARL Systems

---

# Author

Yathin Kumar

---

# GitHub Repository

https://github.com/yathinkumar03/MARL-Autonomous-Driving