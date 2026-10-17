# emnist-letter-classifier

A convolutional neural network in PyTorch that recognizes handwritten letters (A-Z) from the EMNIST dataset.

## Dataset

EMNIST extends MNIST to letters. The **letters** split has 26 classes, with upper- and lowercase merged into one class per letter, and roughly 145,600 images (124,800 train / 20,800 test), 28x28 grayscale.

```python
from torchvision import datasets, transforms
tf = transforms.ToTensor()
train = datasets.EMNIST("data", split="letters", train=True,  download=True, transform=tf)
test  = datasets.EMNIST("data", split="letters", train=False, download=True, transform=tf)
```

Two quirks of the raw data:

- Images are stored transposed, so they are transposed back before training.
- Labels run 1-26, so they are shifted to 0-25.

Because uppercase and lowercase share a class, confusable pairs like `I`/`L` and `Q`/`G` are expected failure modes.

## Model

```
Input        1 x 28 x 28
Conv 3x3,32  → ReLU → Conv 3x3,64 → ReLU → MaxPool 2x2 → Dropout 0.25
Flatten      → Linear 128 → ReLU → Dropout 0.5 → Linear 26
```

Two unpadded 3x3 convs take 28 to 24, pooling takes it to 12, so the flatten size is `64 x 12 x 12 = 9216`.

## Project layout

```
emnist-letter-classifier/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py         # datasets, transforms, loaders
│   ├── model.py        # CNN definition
│   ├── train.py        # training loop, saves best checkpoint
│   └── evaluate.py     # metrics, confusion matrix
├── notebooks/01_demo.ipynb
└── results/
```

## Quickstart

```bash
pip install -r requirements.txt
python -m src.train --epochs 10 --batch-size 128
python -m src.evaluate --checkpoint checkpoints/best.pt
```

## Tech

Python, PyTorch, torchvision, NumPy, matplotlib.
