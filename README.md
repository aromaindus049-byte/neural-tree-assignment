# Neural Tree Assignment
Author: Aroma Indus

## Overview
Neural-network correction trees for CIFAR-10 and Cats vs Dogs.
A trained CNN supplies shared features to small neural-network
nodes. Correction children adjust their parent's output score.

## Architecture
The CNN has two convolution layers, ReLU activations, and max
pooling. It produces 2,048 features from 32 × 32 RGB images.
After initial training, the CNN is frozen. Cached features are
standardized using training-set statistics.

CIFAR-10: ten binary roots with 256 hidden neurons each, plus
a 128-neuron downward correction child for the airplane root.
The class with the largest corrected score is selected.

Cats vs Dogs: an 8-neuron binary root and an 8-neuron downward
correction child. Labels are Cat=0 and Dog=1.

## Final Results

| Model | Train accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| CIFAR-10 correction tree | 100.00% | 70.96% | 72.40% |
| Compact Cats vs Dogs tree | 99.98% | 78.62% | 77.26% |

The CIFAR-10 tree uses 5,515,500 parameters.
The compact Cats vs Dogs tree uses 37,891 parameters.

## Experiments and Findings
- CIFAR-10 root widths: 8, 32, 64, 128, 256, and 512.
- Double descent was not observed in this capacity sweep.
- Root and child input-gradient sensitivity maps were visualized
  for both datasets.
- The compact Cats vs Dogs tree uses 85.8% fewer parameters than
  the 128-neuron root model, with slightly lower accuracy.
- Opposite-label Cats vs Dogs examples have identical images
  after resizing. They impose at least two unavoidable training
  errors on deterministic classifiers using these inputs.
- Better training fit did not produce substantial test gains.

## Running the Code
Main notebook: ASSIGNMENT TREE 3CNN.ipynb

Dependencies: Python, PyTorch, torchvision, NumPy, Matplotlib,
Pillow, certifi, and Jupyter.

Install dependencies:
    pip install torch torchvision numpy matplotlib pillow certifi jupyter

The notebook records exploratory runs and their outputs.
Review cells before execution: model-creation cells reset weights,
and training cells perform additional training when rerun.
Some early exploratory cells depend on interactive execution
order; the notebook is not a verified restart-and-run-all pipeline.

CIFAR-10 downloads through torchvision. Cats vs Dogs downloads
from Microsoft. Change the Cats vs Dogs path to your local
PetImages folder; the recorded path is:
    /Users/aroma/data/cats_vs_dogs/PetImages

Keep dataset images and large feature caches outside GitHub.

## Limitations and Future Work
Use multiple seeds and longer capacity sweeps around interpolation.
Audit duplicate images across dataset splits. Compare shared and
separate trunks, explore pruning and adaptive tree growth, and
compare gradient sensitivity with occlusion-based visualizations.
