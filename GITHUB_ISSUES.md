# GitHub Issues for Seam-Gated-Networks Repository

Copy-paste these directly into GitHub Issues. Each is self-contained with acceptance criteria.

---

## Issue #1: Add Real Dirichlet Character Data Integration

**Labels:** `enhancement`, `data`, `priority-high`

**Description:**

Currently, the repository only supports synthetic data generation. To validate the SGN architecture on real number-theoretic sequences, we need to integrate Dirichlet character sign data from the MDL methodology paper.

**Proposed Changes:**

1. Create `data/dirichlet_loader.py` with functions:
   - `generate_kronecker_symbols(q, N)`: Generate character signs for conductor q
   - `load_precomputed_sequences()`: Load cached character sequences
   - `get_mdl_ground_truth(q)`: Return known dependence structure from theory

2. Add data format documentation:
   ```python
   # Expected format:
   {
       'sequence': np.ndarray,  # Binary sequence (0/1 or ±1)
       'conductor': int,         # q value
       'length': int,           # N
       'ground_truth_bf': float, # Analytical Bayes factor (if known)
       'metadata': dict
   }
   ```

3. Update `experiments/run_experiment.py` to accept `--data_source dirichlet`

**Acceptance Criteria:**

- [ ] Can load sequences for q ∈ {3, 4, 5, 7, 8, 11, 13}
- [ ] Loader validates sequence length and conductor
- [ ] Includes unit tests with known q=3 sequence
- [ ] Documentation added to README

**Related Files:**
- `data/synthetic_generation.py` (reference for format)
- `utils/loaders.py` (may need updates)

---

## Issue #2: Benchmark Against Analytical Bayes Factor Detector

**Labels:** `enhancement`, `experiments`, `validation`

**Description:**

The SGN's learned detection should be compared against the analytical Bayes factor computation from the MDL paper. This validates whether the neural approach captures the same information-theoretic signals.

**Proposed Implementation:**

1. Add `baselines/analytical_detector.py`:
   ```python
   class AnalyticalBayesFactorDetector:
       def compute_log_bf(self, sequence):
           """Compute exact log Bayes factor: log P(x|M1) - log P(x|M0)"""
           # M0: independence model
           # M1: first-order Markov model
           pass

       def detect_dependency(self, sequence, threshold=0.0):
           """Return boolean decision: is dependence detected?"""
           pass
   ```

2. Create comparison experiment script:
   ```python
   # experiments/compare_detectors.py
   results = {
       'sgn_log_bf': [],
       'analytical_log_bf': [],
       'sgn_decision': [],
       'analytical_decision': [],
       'agreement_rate': float
   }
   ```

3. Generate plots:
   - Scatter: SGN vs Analytical log BF
   - Confusion matrix: Decision agreement
   - Calibration curve: Does SGN estimate match theory?

**Acceptance Criteria:**

- [ ] Analytical detector implemented with unit tests
- [ ] Comparison script runs on synthetic data (100 sequences)
- [ ] Correlation between SGN and analytical log BF > 0.8
- [ ] Plots saved to `plots/detector_comparison/`
- [ ] Results table in README

**Metrics to Compare:**
- Log Bayes Factor estimate
- Binary detection decision (threshold at 0)
- AUC-ROC for detection power
- Calibration error (mean squared difference)

---

## Issue #3: Make Training Configuration-Driven (Hydra/YAML)

**Labels:** `enhancement`, `usability`, `config`

**Description:**

Currently, hyperparameters are hardcoded in `run_experiment.py`. To enable systematic hyperparameter sweeps and reproducible experiments, migrate to a configuration-driven system.

**Proposed Architecture:**

Use Hydra for hierarchical configs:

```
configs/
├── config.yaml              # Main config
├── model/
│   ├── sgn_small.yaml
│   ├── sgn_large.yaml
│   └── sgn_lstm_variant.yaml
├── data/
│   ├── synthetic.yaml
│   └── dirichlet.yaml
└── training/
    ├── default.yaml
    └── sweep.yaml
```

**Example config.yaml:**
```yaml
defaults:
  - model: sgn_small
  - data: synthetic
  - training: default

experiment:
  name: sgn_detection
  seed: 42
  output_dir: results/${experiment.name}

model:
  hidden_dim: 64
  gate_temperature: 0.1
  num_layers: 2

training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  early_stopping_patience: 10

data:
  sequence_length: 1000
  num_sequences: 500
  dependency_probability: 0.3
```

**Usage:**
```bash
# Run with default config
python experiments/run_experiment.py

# Override specific parameters
python experiments/run_experiment.py model.hidden_dim=128 training.learning_rate=1e-4

# Hyperparameter sweep
python experiments/run_experiment.py -m \
    model.hidden_dim=32,64,128 \
    model.gate_temperature=0.05,0.1,0.2
```

**Acceptance Criteria:**

- [ ] Hydra integrated (add `hydra-core` to requirements.txt)
- [ ] All hardcoded params moved to YAML configs
- [ ] Backward compatibility: old scripts still work
- [ ] Sweep example added to documentation
- [ ] Config files version-controlled with results

**Benefits:**
- Reproducible experiments (config saved with results)
- Easy hyperparameter tuning
- Cleaner code (separation of concerns)

---

## Issue #4: Add Seam Activation Visualization

**Labels:** `enhancement`, `visualization`, `interpretability`

**Description:**

The core innovation of SGNs is the learned seam gate. We need visualizations showing:
1. Where in the sequence the gate activates
2. How this compares to known dependency windows
3. Whether the gate learns meaningful changepoints

**Proposed Tools:**

Create `visualization/plot_seam_activations.py`:

**Function 1: Gate Activation Heatmap**
```python
def plot_gate_heatmap(
    sequence: np.ndarray,
    gate_activations: np.ndarray,
    ground_truth_window: Optional[Tuple[int, int]] = None,
    save_path: str = None
):
    """
    Plot gate activations along sequence.

    Args:
        sequence: Binary sequence (N,)
        gate_activations: Gate values from SGN (N,)
        ground_truth_window: (start, end) of known dependency
    """
    # 2-panel plot:
    # Top: Binary sequence
    # Bottom: Gate activation with threshold line
```

**Function 2: Changepoint Detection**
```python
def detect_seam_points(
    gate_activations: np.ndarray,
    threshold: float = 0.5
) -> List[int]:
    """Find indices where gate crosses threshold."""
    pass
```

**Function 3: Comparison with Ground Truth**
```python
def plot_detection_comparison(
    sgn_activations: np.ndarray,
    analytical_changepoints: List[int],
    sequence: np.ndarray
):
    """Compare learned vs theoretical changepoints."""
    pass
```

**Example Output:**

```
Seam Gate Analysis for Sequence #42
=====================================
Detected changepoints: [247, 568, 891]
Ground truth window: [250, 600]
Detection accuracy: 85%
False positive rate: 2.1%
```

**Acceptance Criteria:**

- [ ] Heatmap function implemented and tested
- [ ] Changepoint detection with configurable threshold
- [ ] Comparison plot showing SGN vs analytical
- [ ] Example notebook: `notebooks/visualize_seam_gates.ipynb`
- [ ] High-quality figures saved to `plots/seam_activations/`

**Related Issues:**
- Pairs well with Issue #2 (analytical comparison)

---

## Issue #5: Expand README with Architecture Diagram and Examples

**Labels:** `documentation`, `good-first-issue`

**Description:**

Current README is minimal. Expand it to include:
1. **Overview**: What are Seam-Gated Networks?
2. **Architecture Diagram**: Visual representation of the gate mechanism
3. **Quick Start**: Complete example from installation to results
4. **Example Outputs**: Show what users should expect

**Proposed Sections:**

### 1. Overview
```markdown
## What are Seam-Gated Networks?

SGNs are neural architectures designed to detect **minimal dependence structure**
in binary sequences using information-theoretic principles from **Minimum Description Length (MDL)** theory.

**Key Innovation:** A learnable "seam gate" that identifies when a sequence
transitions from independence to first-order Markov behavior.

**Applications:**
- Dirichlet character sign detection
- Changepoint detection in binary time series
- MDL-based model comparison
```

### 2. Architecture Diagram

Create diagram showing:
```
Input Sequence → Embedding → [Seam Gate] → RNN → Output
                                   ↓
                            Dependency Detector
```

Use mermaid.js or ASCII art.

### 3. Quick Start
```markdown
## Quick Start

### Installation
\`\`\`bash
pip install -r requirements.txt
\`\`\`

### Run Experiment
\`\`\`bash
# Synthetic data
python experiments/run_experiment.py --data synthetic --epochs 50

# Real Dirichlet data (after Issue #1)
python experiments/run_experiment.py --data dirichlet --conductor 5
\`\`\`

### Visualize Results
\`\`\`python
from visualization.plot_seam_activations import plot_gate_heatmap

# Load trained model
model = torch.load('checkpoints/best_model.pt')
activations = model.get_gate_activations(test_sequence)

# Plot
plot_gate_heatmap(test_sequence, activations, save_path='gate_activation.png')
\`\`\`
```

### 4. Expected Outputs

Add section showing:
```markdown
## Example Results

**Synthetic Data (500 sequences, length 1000)**

| Metric | Value |
|--------|-------|
| Detection Accuracy | 94.2% |
| Mean Log BF Error | 0.15 |
| AUC-ROC | 0.98 |

**Sample Visualization:**

![Gate Activation](plots/example_gate_activation.png)
```

### 5. Citation

```markdown
## Citation

If you use this code, please cite:

\`\`\`bibtex
@software{seam_gated_networks,
  author = {MacMayo1993},
  title = {Seam-Gated Networks for Minimal Dependence Detection},
  year = {2024},
  url = {https://github.com/MacMayo1993/Seam-Gated-Networks}
}
\`\`\`
```

**Acceptance Criteria:**

- [ ] All sections added to README
- [ ] Architecture diagram included (SVG or mermaid)
- [ ] Quick start tested on fresh clone
- [ ] Example outputs verified
- [ ] Links to papers/references added

---

## Issue #6: Add Ablation Experiments (No Gating vs. Gated)

**Labels:** `experiments`, `validation`, `ablation`

**Description:**

To validate the seam gate's contribution, we need ablation studies comparing:
1. **Full SGN** (with seam gate)
2. **Ablated SGN** (gate always on / always off)
3. **Baseline RNN** (no gate mechanism)

**Proposed Experiments:**

Create `experiments/ablation_study.py`:

```python
models = {
    'sgn_full': SeamGatedNetwork(gate_enabled=True),
    'sgn_no_gate': SeamGatedNetwork(gate_enabled=False),
    'sgn_always_on': SeamGatedNetwork(gate_value=1.0),
    'baseline_rnn': StandardRNN()
}

results = {}
for name, model in models.items():
    results[name] = evaluate(model, test_data)
```

**Metrics to Compare:**

| Model | Log BF MAE | Detection Accuracy | Params | Training Time |
|-------|-----------|-------------------|--------|---------------|
| Full SGN | ? | ? | ? | ? |
| No Gate | ? | ? | ? | ? |
| Always On | ? | ? | ? | ? |
| Baseline RNN | ? | ? | ? | ? |

**Analysis Questions:**

1. Does the gate improve detection accuracy?
2. Is the improvement statistically significant?
3. Does the gate learn to activate at the right times?
4. What's the cost in parameters/compute?

**Visualization:**

Plot:
- Detection accuracy vs model type (bar chart)
- Log BF estimation error (box plot)
- Training curves (loss over epochs)

**Acceptance Criteria:**

- [ ] Ablation script implemented
- [ ] 4 model variants trained on same data
- [ ] Statistical significance testing (t-test, bootstrap)
- [ ] Results table in markdown format
- [ ] Plots saved to `plots/ablations/`
- [ ] Analysis writeup added to docs

**Expected Outcome:**

Full SGN should outperform ablations, demonstrating the gate's utility.

---

## Issue #7: Add Noise Robustness Testing

**Labels:** `experiments`, `robustness`, `testing`

**Description:**

Real sequences contain noise (measurement errors, data corruption). Test SGN robustness to:
1. **Random bit flips**: Flip bits with probability p
2. **Burst errors**: Consecutive corrupted bits
3. **Missing data**: Random dropout of observations

**Proposed Implementation:**

Create `experiments/noise_robustness.py`:

```python
def add_noise(sequence, noise_type, noise_level):
    """
    Add noise to binary sequence.

    Args:
        sequence: Original sequence
        noise_type: 'flip', 'burst', 'dropout'
        noise_level: 0.0 to 1.0
    """
    if noise_type == 'flip':
        # Random bit flips
        mask = np.random.rand(len(sequence)) < noise_level
        noisy = sequence.copy()
        noisy[mask] = 1 - noisy[mask]
        return noisy

    elif noise_type == 'burst':
        # Consecutive corrupted regions
        pass

    elif noise_type == 'dropout':
        # Missing observations (NaN or -1)
        pass

noise_levels = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]
results = {}

for level in noise_levels:
    noisy_sequences = [add_noise(seq, 'flip', level) for seq in test_sequences]
    results[level] = evaluate_sgn(model, noisy_sequences)
```

**Metrics:**

Track how performance degrades:
- Detection accuracy vs noise level
- Log BF estimation error vs noise
- Calibration error vs noise

**Visualization:**

```python
plt.plot(noise_levels, detection_accuracy, label='Detection Accuracy')
plt.plot(noise_levels, [baseline_accuracy]*len(noise_levels),
         label='Baseline (no noise)', linestyle='--')
plt.xlabel('Noise Level (flip probability)')
plt.ylabel('Detection Accuracy')
plt.title('SGN Robustness to Random Bit Flips')
```

**Acceptance Criteria:**

- [ ] 3 noise types implemented (flip, burst, dropout)
- [ ] Tested across noise levels [0.0, 0.5]
- [ ] Performance curves plotted
- [ ] Comparison with analytical detector under noise
- [ ] Results table showing graceful degradation
- [ ] Recommendations: "SGN robust up to 10% noise"

**Related Issues:**
- Issue #2 (compare with analytical under noise)

---

## Issue #8: Test Block Length Variation (Scale Sensitivity)

**Labels:** `experiments`, `feature-request`

**Description:**

Dependence structure can occur at different scales:
- **Local**: 1st-order Markov (depends on previous bit)
- **Mesoscopic**: Block-level structure (depends on last k bits)
- **Global**: Long-range correlations

Test how SGN handles different block lengths:

```python
block_lengths = [1, 2, 5, 10, 20, 50]  # Markov order k

for k in block_lengths:
    # Generate k-th order Markov sequence
    data = generate_kth_order_markov(N=1000, k=k)

    # Train SGN
    model = train_sgn(data)

    # Evaluate: Can it detect k-th order structure?
    results[k] = evaluate(model, test_data_k)
```

**Research Questions:**

1. Can SGN detect higher-order Markov structure?
2. Does detection accuracy degrade with k?
3. Do we need to increase hidden_dim for larger k?

**Proposed Analysis:**

```markdown
| Block Length (k) | Detection Acc. | Recommended hidden_dim |
|------------------|----------------|------------------------|
| 1 (1st-order)    | 94.2%          | 64                     |
| 2                | 89.5%          | 64                     |
| 5                | 82.1%          | 128                    |
| 10               | 71.3%          | 256                    |
| 20               | 58.4%          | 512                    |
```

**Acceptance Criteria:**

- [ ] k-th order Markov generator implemented
- [ ] Experiments run for k ∈ {1, 2, 5, 10, 20}
- [ ] Results table with detection accuracy
- [ ] Analysis: relationship between k and required capacity
- [ ] Recommendation: SGN effective for k ≤ 5

---

## Issue #9: Create Example Jupyter Notebooks

**Labels:** `documentation`, `examples`, `good-first-issue`

**Description:**

Add interactive notebooks for:
1. **Tutorial**: Introduction to SGNs
2. **Quickstart**: Train a model in <5 minutes
3. **Advanced**: Hyperparameter tuning and analysis

**Proposed Notebooks:**

### 1. `notebooks/01_introduction.ipynb`

```markdown
# Introduction to Seam-Gated Networks

## 1. What is Minimal Dependence Detection?

[Explanation with visualizations]

## 2. The Seam Gate Mechanism

[Interactive diagram]

## 3. Toy Example: Coin Flip Sequences

[Code cells generating and detecting dependence]

## 4. Comparison with Analytical Bayes Factor

[Side-by-side comparison]
```

### 2. `notebooks/02_quickstart.ipynb`

```python
# 5-Minute Quickstart

# 1. Generate data
from data.synthetic_generation import generate_markov_sequence
X, y = generate_markov_sequence(N=1000, dependency=True)

# 2. Train model
from models.sgn import SeamGatedNetwork
model = SeamGatedNetwork(hidden_dim=64)
model.fit(X, y, epochs=50)

# 3. Evaluate
from utils.metrics import compute_log_bf
log_bf = compute_log_bf(model, X_test)
print(f"Detected log Bayes Factor: {log_bf:.2f}")

# 4. Visualize
from visualization.plot_seam_activations import plot_gate_heatmap
plot_gate_heatmap(X_test[0], model.get_activations(X_test[0]))
```

### 3. `notebooks/03_hyperparameter_tuning.ipynb`

- Grid search over hidden_dim, gate_temperature
- Optuna integration for Bayesian optimization
- Visualization of hyperparameter effects

### 4. `notebooks/04_dirichlet_analysis.ipynb` (after Issue #1)

- Load real Dirichlet character data
- Compare SGN vs analytical detection
- Reproduce results from MDL paper

**Acceptance Criteria:**

- [ ] 4 notebooks created in `notebooks/` directory
- [ ] All cells execute without errors
- [ ] Clear markdown explanations between code
- [ ] Visualizations render correctly
- [ ] Notebooks added to README table of contents

---

## Issue #10: Add Unit Tests and CI/CD Pipeline

**Labels:** `testing`, `infrastructure`, `good-first-issue`

**Description:**

Currently no automated testing. Add:
1. **Unit tests** for core modules
2. **Integration tests** for full pipeline
3. **GitHub Actions CI** for automated testing

**Proposed Structure:**

```
tests/
├── test_models.py
├── test_data.py
├── test_metrics.py
├── test_visualization.py
└── test_integration.py
```

**Example Test:**

```python
# tests/test_models.py

import pytest
import torch
from models.seam_gate import SeamGate

def test_seam_gate_shape():
    """Test gate output has correct shape."""
    gate = SeamGate(input_dim=64, temperature=0.1)
    x = torch.randn(32, 100, 64)  # (batch, seq, features)

    activations = gate(x)

    assert activations.shape == (32, 100), \
        f"Expected shape (32, 100), got {activations.shape}"

def test_seam_gate_range():
    """Test gate outputs are in [0, 1]."""
    gate = SeamGate(input_dim=64)
    x = torch.randn(32, 100, 64)

    activations = gate(x)

    assert torch.all(activations >= 0) and torch.all(activations <= 1), \
        "Gate activations must be in [0, 1]"

def test_seam_gate_temperature():
    """Test temperature affects sharpness."""
    x = torch.randn(32, 100, 64)

    gate_sharp = SeamGate(input_dim=64, temperature=0.01)
    gate_soft = SeamGate(input_dim=64, temperature=10.0)

    act_sharp = gate_sharp(x)
    act_soft = gate_soft(x)

    # Sharp gate should have lower entropy (more decisive)
    entropy_sharp = -torch.mean(act_sharp * torch.log(act_sharp + 1e-8))
    entropy_soft = -torch.mean(act_soft * torch.log(act_soft + 1e-8))

    assert entropy_sharp < entropy_soft, \
        "Sharp gate should have lower entropy"
```

**GitHub Actions Workflow:**

```yaml
# .github/workflows/tests.yml

name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov

    - name: Run tests
      run: pytest tests/ --cov=. --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

**Acceptance Criteria:**

- [ ] 20+ unit tests covering core functionality
- [ ] Integration test for full training pipeline
- [ ] GitHub Actions workflow configured
- [ ] Tests pass on Python 3.8, 3.9, 3.10, 3.11
- [ ] Code coverage > 80%
- [ ] Badge added to README: ![Tests](https://github.com/.../badge.svg)

---

## Priority / Ordering Suggestion

**High Priority (Do First):**
1. Issue #1 (Real data integration) - Validates on actual problem
2. Issue #5 (README upgrade) - Makes repo accessible
3. Issue #2 (Analytical comparison) - Core validation

**Medium Priority:**
4. Issue #3 (Config system) - Makes experiments reproducible
5. Issue #4 (Visualization) - Interpretability
6. Issue #6 (Ablations) - Validates design choices

**Low Priority (Nice to Have):**
7. Issue #7 (Noise robustness)
8. Issue #8 (Block length variation)
9. Issue #9 (Notebooks)
10. Issue #10 (Testing/CI)

---

## Meta-Issue: Release Preparation

**Labels:** `meta`, `release`

Once Issues #1-#6 are complete, create a release checklist:

- [ ] All documentation complete
- [ ] Real data experiments validated
- [ ] Comparison with analytical detector published
- [ ] Example notebooks working
- [ ] README comprehensive
- [ ] LICENSE file added
- [ ] CITATION.cff created
- [ ] Zenodo DOI requested
- [ ] Preprint/paper reference added

**Target:** v1.0.0 release ready for publication

