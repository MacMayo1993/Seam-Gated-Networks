# Z2-Equivariant Seam-Gated RNN

Complete implementation of the Z2-equivariant recurrent neural network with seam gating for regime-switching time series prediction.

## Overview

This repository implements four models for comparison:
1. **Standard LSTM**: Baseline neural model
2. **Hamilton/IMM Switching Kalman Filter**: Optimal baseline (known parameters)
3. **Z2-Equivariant RNN**: Ablation model without seam mechanism
4. **Z2 Seam-Gated RNN**: Full proposed model with parity-based seam detection

## Project Structure

```
├── data/
│   ├── generator.py           # Antipodal regime switching data generator
│   └── transition_mask.py     # Transition window computation
├── models/
│   ├── lstm.py                # Standard LSTM baseline
│   ├── hamilton_imm.py        # Switching Kalman filter
│   ├── z2_equivariant.py      # Z2-equivariant RNN (no seam)
│   └── z2_seam_gated.py       # Full seam-gated model
├── experiments/
│   ├── train.py               # Training loop
│   ├── evaluate.py            # Evaluation metrics
│   ├── diagnostics.py         # Statistical tests
│   ├── visualize.py           # Figure generation
│   └── ablations.py           # Ablation studies
├── main.py                    # Full experimental pipeline
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run the complete experimental pipeline:
```bash
python main.py --seed 42 --epochs 50
```

Run ablation studies:
```bash
python experiments/ablations.py --ablation window_radius
```

## Mathematical Framework

The model implements Z2 equivariance through:
- Fixed involution operator S with S² = I
- Parity projectors P₊ = (I + S)/2, P₋ = (I - S)/2
- Commuting weight matrix [W_comm, S] = 0
- Seam gating based on parity energy α₋(h) = ‖P₋h‖²/‖h‖²

Critical threshold: k* = 1/(2ln2) ≈ 0.721347...

## Results

Results are saved to:
- `results/`: Metrics tables and statistics
- `figures/`: Visualizations

## Citation

[Paper reference to be added]
