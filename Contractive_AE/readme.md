# Sparse AutoEncoders 

### Train-test Loss


<p align="center">
  <img src="https://github.com/kashyap333/Types-of-Autoencoders/blob/main/Contractive_AE/output/loss.png" width=500 height=400 align="center">
</p>

### Project Structure

```bash
autoencoders
|--Contractive_AE
   |--output
   |  |--images
   |--Contractive_AE.py
|--helper
   |--helper.py

```

### Run cmd

You can provide your own values for experimenting

```cmd
python Contractive_AE.py --epochs=19 --batch_sz=8 --lr=1e-2 --lamda 1e-3
```

### Notes

- While the foundational method of penalising the loss is similar to Sparse functions. The difference is in penalizing loss in each layer and loss in the hidden layer. 
- The way loss is calculated is unique, in that the Frobenius Norm of the Jacobian matrix of only the ***encoder*** is added with the mse loss.
- The Jacobian matrix is partially differentiated from the hidden layer by the input layer. Where the differentiation is backward function or back propagation. 
- Therefore we first initilase gradients for the images, (which doesnot have any) and calculate backward from the hidden layer after mse loss, then calculate the Frobenius Norm, which is added to mse to create a loss for normal model backward or back propagation.
