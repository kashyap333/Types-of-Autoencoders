# Variational AutoEncoders 

### Project Structure

```bash
autoencoders
|--Variational_AE
   |--output
   |  |--model
   |--Variational_AE.py
|--helper
   |--helper.py

```

### Run cmd

You can provide your own values for experimenting

```cmd
python Variational_AE.py --epochs=19 --batch_sz=8 --latent_dims=4
```

### Notes

- We encode the input as a distribution over the latent space, instead of considering it as a single point. 
This encoded distribution is chosen to be normal so that the encoder can be trained to return the mean matrix and the covariance matrix.
- In the second step, we sample a point from that encoded distribution.
- After, we can decode the sampled point and calculate the reconstruction error 
- We backpropagate the reconstruction error through the network. Since the sampling procedure is a discrete process, so it’s not continuous, 
- we need to apply a reparameterisation trick to make the backpropagation work
