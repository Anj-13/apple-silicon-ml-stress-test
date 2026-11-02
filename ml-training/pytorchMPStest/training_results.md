# PyTorch MPS Training Results

Results from the PyTorch MPS experiment.

## Script

`ml-training/pytorchMPStest/pytorch_mps_test.py`

Trains a simple CNN (`SimpleCNN`, 3 conv layers + 2 FC layers) on CIFAR-10
using MPS acceleration, 3 epochs, batch size 128, SGD lr=0.001.

## System Info

- **OS:** macOS (Apple Silicon)
- **Python:** venv, Python 3.11.7
- **Device:** MPS
- **RAM:** 16 GB total (approx. 5.82 GB free during this run)

## Run — CIFAR-10 on MPS

- **Result:** Completed successfully on `mps`.
- **Duration:** 111.03 s (3 epochs, 391 batches per epoch)
- **Loss progression:** barely moved — hovered around 2.30 all 3 epochs,
  ending at 2.298 by Epoch 3. The model did not learn; loss plateaued near
  random-guess level (~2.30 for 10 classes).

### Console output

```
device: mps
RAM: 5.82 GB free
Start training... 🙈
Epoch 1, Batch 100, Loss: 2.303
Epoch 1, Batch 200, Loss: 2.302
Epoch 1, Batch 300, Loss: 2.302
Epoch 2, Batch 100, Loss: 2.301
Epoch 2, Batch 200, Loss: 2.300
Epoch 2, Batch 300, Loss: 2.300
Epoch 3, Batch 100, Loss: 2.299
Epoch 3, Batch 200, Loss: 2.298
Epoch 3, Batch 300, Loss: 2.298
Training completed in 111.03 seconds on mps
```

## Notes

- Loss barely decreased across all epochs — likely an issue with
  learning rate / initialization rather than MPS (the hardware
  acceleration itself worked fine).
