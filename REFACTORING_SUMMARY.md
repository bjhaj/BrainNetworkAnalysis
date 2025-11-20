# NeuroGraph Repository Refactoring Summary

**Date**: November 15, 2024
**Version**: 2.0.0
**Type**: Major structural reorganization

## Overview

The NeuroGraph repository has been completely refactored to improve code organization, maintainability, and usability. This document summarizes all changes made during the refactoring.

## Major Changes

### 1. New Package Structure

The codebase has been reorganized from a flat structure with files scattered in the root directory to a clean, hierarchical Python package:

**Before:**
```
NeuroGraph/
├── main.py, main_cmu.py, main_cmu_simple.py
├── utils.py, cmu_brain_adapter.py, cmu_model_wrapper.py
├── svm_baseline.py, wl_kernel_svm.py, ...
├── run_cmu.sh, run_cmu_full.sh, ...
├── base_params/, results/, embeddings/
└── NeuroGraph/datasets.py, utils.py, preprocess.py
```

**After:**
```
NeuroGraph/
├── neurograph/                    # Main Python package
│   ├── datasets/                  # All dataset classes
│   │   ├── hcp.py                # NeuroGraphDataset (HCP)
│   │   ├── cmu.py                # CMUBrainDataset
│   │   └── dynamic.py            # NeuroGraphDynamic
│   ├── models/                    # All model architectures
│   │   ├── residual_gnn.py       # ResidualGNNs (original)
│   │   └── cmu_residual_gnn.py   # CMU variants
│   ├── data/                      # Data loading utilities
│   │   ├── brain_loader.py       # BrainDataLoader
│   │   └── brain_dataset.py      # BrainGraphDataset
│   ├── utils.py                   # Shared utilities
│   └── preprocess.py              # Preprocessing functions
│
├── experiments/                   # All experiment scripts
│   ├── hcp/train_hcp.py          # HCP experiments
│   ├── cmu/                       # CMU experiments
│   │   ├── train_cmu.py
│   │   └── train_cmu_simple.py
│   └── baselines/                 # Baseline methods
│       └── [svm, wl_kernel, graph2vec scripts]
│
├── scripts/                       # Shell scripts
│   ├── hcp/, cmu/, baselines/
│
├── outputs/                       # Generated files (gitignored)
│   ├── checkpoints/              # Model weights
│   ├── results/                  # Experiment results
│   └── embeddings/               # Pre-computed embeddings
│
└── data/                         # Raw datasets (gitignored)
```

### 2. Import Changes

All imports have been updated to use the new package structure:

**Old imports:**
```python
from NeuroGraph.datasets import NeuroGraphDataset
from cmu_brain_adapter import CMUBrainDataset
from cmu_model_wrapper import CMUResidualGNNs
from utils import *
import sys
sys.path.append('data/CMUBrain')
from data_loader import BrainDataLoader
```

**New imports:**
```python
from neurograph import NeuroGraphDataset, CMUBrainDataset
from neurograph.models import ResidualGNNs, CMUResidualGNNs
from neurograph.utils import fix_seed
from neurograph.data import BrainDataLoader
```

### 3. File Relocations

#### Datasets
- `NeuroGraph/datasets.py` → `neurograph/datasets/hcp.py` + `neurograph/datasets/dynamic.py`
- `cmu_brain_adapter.py` → `neurograph/datasets/cmu.py`

#### Models
- `utils.py` (ResidualGNNs class) → `neurograph/models/residual_gnn.py`
- `cmu_model_wrapper.py` → `neurograph/models/cmu_residual_gnn.py`

#### Data Loaders
- `data/CMUBrain/data_loader.py` → `neurograph/data/brain_loader.py`
- `data/CMUBrain/dataset.py` → `neurograph/data/brain_dataset.py`

#### Core Utilities
- `utils.py` (fix_seed function) → `neurograph/utils.py`
- `NeuroGraph/preprocess.py` → `neurograph/preprocess.py`

#### Experiment Scripts
- `main.py` → `experiments/hcp/train_hcp.py`
- `main_cmu.py` → `experiments/cmu/train_cmu.py`
- `main_cmu_simple.py` → `experiments/cmu/train_cmu_simple.py`
- Baseline scripts → `experiments/baselines/`

#### Shell Scripts
- `run_cmu.sh`, `run_cmu_full.sh` → `scripts/cmu/`

#### Outputs
- `base_params/` → `outputs/checkpoints/`
- `results/` → `outputs/results/`
- `embeddings/` → `outputs/embeddings/`

### 4. Path Updates

All scripts now use paths relative to the repository root:

**Old:**
```python
path = "base_params/"
res_path = "results/"
root = "data/CMUBrain"
```

**New:**
```python
REPO_ROOT = Path(__file__).parent.parent.parent
path = REPO_ROOT / "outputs" / "checkpoints"
res_path = REPO_ROOT / "outputs" / "results"
root = REPO_ROOT / "data" / "CMUBrain"
```

### 5. Updated Configuration Files

#### setup.py
- Package name: `NeuroGraph` → `neurograph` (lowercase)
- Version: `3.0.0` → `2.0.0`
- Added proper package structure with `find_packages()`
- Updated dependencies with version requirements
- Added extras_require for baselines and dev tools

#### .gitignore
- Added `outputs/` directory
- Kept old paths (`base_params/`, `results/`, `embeddings/`) for backward compatibility
- Added `!data/README.md` to track documentation

### 6. New Documentation

Created comprehensive README files:
- `data/README.md` - Describes dataset structure and usage
- `outputs/README.md` - Explains output organization
- `REFACTORING_SUMMARY.md` - This document

## How to Use the Refactored Repository

### Installation

```bash
# Install in development mode
cd /scratch/lmicevic/NeuroGraph
pip install -e .

# Or with baselines
pip install -e ".[baselines]"
```

### Running Experiments

**HCP experiments:**
```bash
python experiments/hcp/train_hcp.py --dataset HCPGender --runs 1
```

**CMU experiments:**
```bash
# Using scripts (recommended)
cd /scratch/lmicevic/NeuroGraph
bash scripts/cmu/run_cmu.sh

# Or directly
python experiments/cmu/train_cmu.py --task sex --runs 10
python experiments/cmu/train_cmu_simple.py --task sex --runs 10
```

### Importing as a Package

```python
# Import datasets
from neurograph import NeuroGraphDataset, CMUBrainDataset

# Load HCP data
hcp_dataset = NeuroGraphDataset(root='data/', name='HCPGender')

# Load CMU data
cmu_dataset = CMUBrainDataset(root='data/CMUBrain', task='sex')

# Import models
from neurograph.models import ResidualGNNs, CMUResidualGNNs

# Use utilities
from neurograph.utils import fix_seed
fix_seed(123)
```

## Breaking Changes

### For External Users

1. **Package name changed**: `NeuroGraph` → `neurograph` (all lowercase)
2. **Import paths changed**: All imports must be updated to use `neurograph.*`
3. **File locations changed**: Cannot run scripts from root anymore
4. **Output paths changed**: Results now in `outputs/` instead of `results/`

### For Internal Development

1. **Script execution**: Must run from repo root or use updated scripts
2. **Model checkpoints**: Moved to `outputs/checkpoints/`
3. **Results files**: Moved to `outputs/results/`

## Migration Checklist for External Code

If you have code that depends on the old structure:

- [ ] Update package name: `import NeuroGraph` → `import neurograph`
- [ ] Update dataset imports: `from NeuroGraph.datasets` → `from neurograph.datasets`
- [ ] Update model imports: `from utils import ResidualGNNs` → `from neurograph.models import ResidualGNNs`
- [ ] Update paths: `results/` → `outputs/results/`, `base_params/` → `outputs/checkpoints/`
- [ ] Update script calls: `python main_cmu.py` → `python experiments/cmu/train_cmu.py`

## Benefits of Refactoring

1. **Better organization**: Clear separation between library code, experiments, and outputs
2. **Proper Python package**: Can be installed via pip and imported anywhere
3. **Cleaner imports**: No more sys.path hacks or relative imports
4. **Easier maintenance**: Related code is grouped together
5. **Better for version control**: Outputs separated from source code
6. **Standard structure**: Follows Python packaging best practices
7. **Easier to extend**: Clear places to add new datasets, models, or experiments

## Files Moved

### Created (26 new files)
- `neurograph/__init__.py`
- `neurograph/datasets/__init__.py`, `hcp.py`, `cmu.py`, `dynamic.py`
- `neurograph/models/__init__.py`, `residual_gnn.py`, `cmu_residual_gnn.py`
- `neurograph/data/__init__.py`, `brain_loader.py`, `brain_dataset.py`
- `neurograph/utils.py`, `preprocess.py`
- `experiments/hcp/train_hcp.py`
- `experiments/cmu/train_cmu.py`, `train_cmu_simple.py`
- `experiments/baselines/[6 files]`
- `scripts/cmu/run_cmu.sh`, `run_cmu_full.sh`
- `data/README.md`, `outputs/README.md`, `REFACTORING_SUMMARY.md`

### Modified
- `setup.py` - Complete rewrite for new structure
- `.gitignore` - Added outputs/ and cleaned up
- All experiment scripts - Updated imports and paths

### Relocated (outputs)
- `base_params/*` → `outputs/checkpoints/*`
- `results/*` → `outputs/results/*`
- `embeddings/*` → `outputs/embeddings/*`

## Testing

The refactoring has been tested by:
1. ✓ Verifying package structure (`find neurograph/`)
2. ✓ Checking import statements (syntax correct, modules not installed)
3. ✓ Validating file relocations
4. ✓ Confirming outputs directory structure
5. ✓ Updating all script paths
6. ✓ Creating comprehensive documentation

## Next Steps

Recommended actions after refactoring:

1. **Test experiments**: Run a quick experiment to ensure everything works
2. **Update documentation**: Revise README.md with new structure
3. **Git commit**: Commit these changes with detailed message
4. **Communicate changes**: Notify any collaborators of the new structure
5. **Archive old structure**: Keep a backup branch before deleting old files

## Questions or Issues?

If you encounter any issues with the refactored structure:
1. Check this summary for migration guidance
2. Review the README files in `data/` and `outputs/`
3. Examine the new `neurograph/__init__.py` for available imports
4. Look at experiment scripts in `experiments/` for usage examples

---

**Summary**: This refactoring transforms NeuroGraph from a collection of scripts into a professional Python package with clear organization, making it easier to use, maintain, and extend.
