# Example on MNIST
## Using ProtoTree
### Training
```bash
cabrnet train --device cpu --seed 42 --verbose --logger-level INFO  \
  --model-config configs/prototree/mnist/model.yml \
  --dataset configs/prototree/mnist/data.yml \
  --training configs/prototree/mnist/training.yml \
  --output-dir runs/mnist_prototree \
  --visualization configs/prototree/mnist/visualization.yml
```
This command trains a ProtoTree during one epoch, and stores the resulting checkpoint in 
`runs/mnist_prototree/final`.

### Global explanation
```bash
cabrnet explain_global --verbose \
  --model-config runs/mnist_prototree/final/model.yml \
  --model-state-dict runs/mnist_prototree/final/model_state.pth \
  --output-dir runs/mnist_prototree/global_explanation --prototype-dir runs/mnist_prototree/prototypes/
```
This command generates a global explanation for the ProtoTree model and stores the result in 
`runs/mnist_prototree/global_explanation`.

![prototree mnist global explanation](../imgs/prototree_mnist_global_explanation.png)

### Local explanation
```bash
cabrnet explain_local --verbose \
  --model-config runs/mnist_prototree/final/model.yml  \
  --model-state-dict runs/mnist_prototree/final/model_state.pth \
  --dataset configs/prototree/mnist/data.yml \
  --visualization configs/prototree/mnist/visualization.yml \
  --prototype-dir runs/mnist_prototree/prototypes/ \
  --output-dir runs/mnist_prototree/local_explanations/  \
  --image examples/images/mnist_sample.png
```
This command generates a local explanation for the image stored in `examples/images/mnist_sample.png` and stores the result in 
`runs/mnist_prototree/local_explanation`.

![prototree mnist local explanation](../imgs/prototree_mnist_local_explanation.png)