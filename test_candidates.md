# Testing notes
To view performance logs...
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