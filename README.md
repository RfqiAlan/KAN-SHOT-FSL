# KAN-SHOT: Kolmogorov-Arnold Networks as Interpretable Distance Metrics for Cross-Domain Few-Shot Histopathology

This repository contains the official implementation of the KAN-SHOT method as described in the paper.

## Repository Structure

- `src/` & `kan_shot/`: Core method implementations including the KAN-ProtoNet model, standard MLP-ProtoNet baselines, and dataset loaders.
- `scripts/`: Scripts for evaluating the model on target domains, performing episodic fine-tuning, and extracting Spline Feature Attribution (SFA).
- `requirements.txt`: Python dependencies required to run the code.

## Key Features

1. **Parameter Efficiency**: Replaces dense linear layers in metric learning with highly efficient B-splines (~5K parameters).
2. **White-Box Interpretability**: Introduces Spline Feature Attribution (SFA) to identify the exact feature dimensions driving medical diagnoses.
3. **Cross-Domain Adaptation**: Evaluated under extreme cross-organ domain shifts (e.g., Colorectal to Lung tissues).

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Evaluate the pre-trained model:
   ```bash
   python scripts/eval_multiseed.py
   ```
