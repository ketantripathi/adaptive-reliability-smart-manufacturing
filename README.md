# Modular Petri-Net Inspired Reliability Modeling for IoT Smart Manufacturing

This project builds a complete multi-notebook research workflow inspired by the paper
"Reliability Modeling and Assessment of IoT in Smart Manufacturing Systems: A Modular
Petri Net Approach".

The goal is not only reproduction. The notebooks re-implement a baseline system, identify
limitations, introduce an adaptive predictive reliability-aware control model, and compare
baseline and improved performance with datasets and paper-ready plots.

## Project Structure

```text
.
├── 01_system_modeling.ipynb
├── 02_baseline_simulation.ipynb
├── 03_dataset_generation.ipynb
├── 04_limitation_analysis.ipynb
├── 05_proposed_model.ipynb
├── 06_improved_simulation.ipynb
├── 07_comparative_analysis.ipynb
├── 08_ablation_study.ipynb
├── core/
│   ├── production_model.py
│   ├── network_model.py
│   ├── reliability_model.py
│   └── simulation_engine.py
└── outputs/
    ├── datasets/
    ├── graphs/
    └── logs/
```

## System Design

The manufacturing system contains four production units. Each unit has four states:

- `0`: failed
- `1`: degraded
- `2`: nominal
- `3`: boost

The production units follow stress-aware Markov transitions. High production commands and
network overload increase degradation risk and the probability of moving to lower states.

The information network has end nodes, route nodes, and a gateway. Loads propagate upward
through the network. Gateway overload causes nonlinear transmission accuracy degradation and
reduces effective delivered production.

Reliability is computed as:

```text
Rs(t) = 1 if production_rate(t) >= demand else 0
```

## Baseline Model

The baseline uses a fixed production command:

```text
gamma(t) = constant
```

It has no prediction, no feedback correction, and no adaptive load control. This intentionally
creates overload under high traffic, which lowers transmission accuracy and increases failure
frequency.

## Proposed Model

The improved model is an Adaptive Predictive Reliability-Aware Controller. At each time step it
estimates next-step reliability and chooses a production command by maximizing:

```text
J = R_hat(t+1)
    - lambda_degradation * degradation_risk
    - lambda_overload * overload_risk
    - lambda_smoothing * |gamma(t) - gamma(t-1)|
```

The controller adds:

- Dynamic load-aware production scaling
- Overload risk penalty
- Probabilistic next-step reliability prediction
- Smoothing to reduce production oscillation

## How to Run

Open the notebooks in Google Colab or VS Code and run them in order:

1. `01_system_modeling.ipynb`
2. `02_baseline_simulation.ipynb`
3. `03_dataset_generation.ipynb`
4. `04_limitation_analysis.ipynb`
5. `05_proposed_model.ipynb`
6. `06_improved_simulation.ipynb`
7. `07_comparative_analysis.ipynb`
8. `08_ablation_study.ipynb`

Each notebook is also independently executable. If a required dataset is missing, the notebook
regenerates it.

Required Python packages:

```text
numpy
pandas
matplotlib
```

## Generated Outputs

Datasets are saved in `outputs/datasets/`, including:

- `baseline_dataset.csv`
- `baseline_raw_runs.csv`
- `improved_dataset.csv`
- `comparative_summary.csv`
- `comparative_improvement_metrics.csv`
- `ablation_dataset.csv`

Graphs are saved in `outputs/graphs/`, including:

- System state evolution
- Baseline performance
- Limitation analysis
- Controller objective curve
- Baseline vs improved reliability
- Baseline vs improved production rate
- Baseline vs improved network load
- Baseline vs improved failure rate
- Baseline vs improved transmission accuracy
- Baseline vs improved throughput
- Ablation study charts

## Expected Result Pattern

Under the default experiment conditions, the improved model should show:

- Higher mean reliability
- Lower failure rate
- Reduced network load
- Higher transmission accuracy
- Lower production variance
- Better throughput stability

The exact values can vary slightly with random seed, but `07_comparative_analysis.ipynb`
computes the final percentage improvements and saves them for reporting.
