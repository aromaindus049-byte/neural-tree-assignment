# Neural Tree Assignment

**Author:** Aroma Indus, IIT Delhi

Neural correction trees for CIFAR-10 and Cats vs Dogs.
A trained CNN provides shared features to small neural-network nodes.

## Files

- `Final_Report.pdf`: architecture, results, interpretations and future work.
- `Neural_Tree_Final.ipynb`: exploratory code, recorded outputs and figures.

## Architecture

Each dataset uses its own CNN:
two convolution/ReLU/max-pooling blocks produce 2,048 features.

The CNN is frozen after initial training. Features are standardized
using training statistics and supplied to all nodes.

Each node is a small network:
Linear(2048, hidden width) → ReLU → Linear(hidden width, 1).

Correction children adjust root scores:
- Child A raises scores to recover missed positives.
- Child B lowers scores to suppress false positives.

CIFAR-10 uses ten class roots and predicts the largest corrected score.
Cats vs Dogs uses one binary root.

This is a correction-tree architecture; samples are not exclusively
routed down left/right branches.

## Original Results

| Model | Training accuracy | Validation accuracy | Test accuracy |
|---|---:|---:|---:|
| CIFAR-10 roots + airplane child | 100.00% | 70.96% | 72.40% |
| Cats vs Dogs compact tree | 99.98% | 78.62% | 77.26% |

These results use the original dataset splits.

## Additional Experiments

### Duplicate audit

The Cats vs Dogs audit found 28 identical preprocessed-input groups.
Ten crossed split boundaries.

Four validation images and six test images matching training inputs
were excluded from revised evaluation subsets. Training was unchanged.

Two training groups contained identical inputs with conflicting labels,
forcing at least two errors for a deterministic classifier.

### Adaptive growth and matched-size comparison

Children are added according to the root's training errors.
A fixed hidden-unit budget is divided among the needed children.

For Cats vs Dogs, three seeds were evaluated with a fixed CNN:
- Width-16 root: mean validation accuracy 78.79%.
- Adaptive tree: mean validation accuracy 77.96%.

The trees made fewer training errors on average, but the wider root
generalized better.

### CIFAR pruning

The adaptive CIFAR tree reduced training errors from 239 to 55.
Removing five zero-strength children reduced parameters from
333,136 to 283,926, a 14.77% reduction.

Training and validation predictions were preserved after pruning.
The matched-size width-16 roots performed better than the tree.
This comparison used one seed.

### Interpolation and double descent

Six CIFAR root widths were investigated: 8, 32, 64, 128, 256 and 512.

After extending training, widths 32–512 reached zero training errors.
No clear double descent was observed.

Weighted validation BCE increased while validation accuracy remained
nearly stable. Confidence was not measured directly.

### Node visualizations

Input-gradient sensitivity maps compare roots and children.
Examples show both successful suppression of competing classes
and harmful suppression of the true class.

Maps describe local continuous-score sensitivity; they are not
semantic attention maps or derivatives of the hard correction flags.

## Running the Notebook

Dependencies:
Python, PyTorch, torchvision, NumPy, Pillow and Matplotlib.

Training was performed in Jupyter on macOS using PyTorch MPS.
CPU can be used if MPS is unavailable.

Download CIFAR-10 through torchvision and obtain the Microsoft
Cats vs Dogs dataset. Update local dataset paths before running.

Follow the notebook sections in order:
1. Dataset preparation and splits.
2. CNN training and feature caching.
3. Root and correction-child training.
4. Capacity experiments and original evaluation.
5. Duplicate audit and revised validation subsets.
6. Adaptive comparisons, pruning and visualizations.

The notebook contains exploratory reruns and debugging history.
A clean restart-and-run-all execution has not been verified.
Some later cells require checkpoints produced by earlier experiments.

## Limitations and Future Work

- CIFAR adaptive comparisons need more seeds.
- Head-seed repeats use one fixed CNN and dataset split.
- Parameter counts are matched; training compute is not.
- Revised test-subset evaluation has not been performed.
- Near-duplicates and original-resolution label conflicts need auditing.
- Future work includes grouped splits, deeper correction trees,
  validation-guided growth, feature compression and occlusion analysis.
