hparams = [(10, 8, 2e-05), (10, 8, 3e-05), (10, 8, 5e-05), (10, 16, 2e-05), (10, 16, 3e-05), (10, 16, 5e-05), (10, 32, 2e-05), (10, 32, 3e-05), (10, 32, 5e-05), (5, 8, 2e-05), (5, 8, 3e-05), (5, 8, 5e-05), (5, 16, 2e-05), (5, 16, 3e-05), (5, 16, 5e-05), (5, 32, 2e-05), (5, 32, 3e-05), (5, 32, 5e-05), (15, 8, 2e-05), (15, 8, 3e-05), (15, 8, 5e-05), (15, 16, 2e-05), (15, 16, 3e-05), (15, 16, 5e-05), (15, 32, 2e-05), (15, 32, 3e-05), (15, 32, 5e-05)]

print("Index,Epoch,Batch Size,Learning Rate")
for i, params in enumerate(hparams):
    print(f"{i},{params[0]},{params[1]},{params[2]}")