## Denoising AutoEncoder

Read more about Denoising autoencoders [here](https://iq.opengenus.org/types-of-autoencoder/)

The dataset can be found [here](https://www.kaggle.com/competitions/denoising-dirty-documents/data)

### Project Structure

```bash
autoencoders
|--helper
   |--helper.py
|--Denoising_AE
   |--denoising-dirty-documents
   |--output
   |  |--images
   |  |--model
   |     |--model_epochs96.pt
   |--Denoising_AE.py

```

### Run cmd

Provide your own values for experimenting

```bash
python Denoising_AE.py --epochs 96 --batch_sz 8 --lr 1e-2 --weight_decay 1e-6
```

### Notes

- Honestly after doing the undercomplete autoencoder and denoising autoencoder, both have similar architectures but its just how loss is calculated that differs.
- Model uses Maxpooling and unpooling, mainly used for trying new layers and decreasing training time, removing the layer should increase results as Pooling ovewrites across kernel.
- Model was trained on cpu (as I unfortunately do not have GPU and did not want to use precious limited GPU available on Kaggle 😞) and took around 15 min. 
- Images are converted to grayscale, again for speed purposes, a 3-channel input can also be tried but honestly I dont think it should matter much.  
 

