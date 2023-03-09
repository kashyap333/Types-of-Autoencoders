### Sparse AutoEncoders

Sparse autoencoders are similar to the undercomplete autoencoders in that they use the same image as input and ground truth. 
However the means via which encoding of information is regulated is significantly different. While undercomplete autoencoders are regulated and fine-tuned by regulating the size of the bottleneck, the sparse autoencoder is regulated by changing the number of nodes at each hidden layer.
Since it is not possible to design a neural network that has a flexible number of nodes at its hidden layers, sparse autoencoders work by penalizing the activation of some neurons in hidden layers.
In other words, the loss function has a term that calculates the number of neurons that have been activated and provides a penalty that is directly proportional to that.
This penalty, called the sparsity function, prevents the neural network from activating more neurons and serves as a regularizer.

![sparse AE](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-28_at_3.36.11_PM_wfLA8dB.png)

While typical regularizers work by creating a penalty on the size of the weights at the nodes, sparsity regularizer works by creating a penalty on the number of nodes activated.
This form of regularization allows the network to have nodes in hidden layers dedicated to find specific features in images during training and treating the regularization problem as a problem separate from the latent space problem.
We can thus set latent space dimensionality at the bottleneck without worrying about regularization.
There are two primary ways in which the sparsity regularizer term can be incorporated into the loss function.
1) L1 Loss which I have used here and
2) KL-divergence

Usage: 
##### Run cmd line with python sparse_encoder.py --epochs=(custom_value) --add_sparse=(True/False) --reg_parameter=(custom_value) --batch_sz=(custom_value) --lr=(custom_value)
