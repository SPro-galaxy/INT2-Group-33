# Testing notes
All tests carried out to 90 epochs. To view performance logs...
```sh
tensorboard --logdir="logs"
```

# tc1
Original code, no changes made.

# tc2
Lowered dropout on all Conv layers to 0.15.

# tc3 
Adding a new Conv layer with 256 filters.

**This shows promise as the val_accuracy is still increasing at epoch 90.**

# tc4
Adding two new Conv layers with 256 and 512 filters.

**Failed.**

# tc5
Adding a new Conv layer with 256 filters and another fully connected layer.

**Results were almost the same as `tc3` but more sporadic, some clear outliers.**