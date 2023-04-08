Train_input.npy is the polymer sequence. Every input is a 20 dimensional one-hot encoding for 20 bead length AB copolymer, like [11000100001110000111].
Train_output.npy is the corresponding adhesive free energy with a random patterned surface.
It is a regression task which uses the polymer sequence information to predict the adhesive free energy with a patterned surface.

load the dataset with the following code:
```
input_Seqs = np.load("Train_input.npy")
output_Fs = np.load("Train_output.npy")
```

The baseline accuracy on an untouched test dataset: R^2 is 0.869 and MAE is 0.090 kT.