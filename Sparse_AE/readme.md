# Sparse AutoEncoders 

### Train-test Loss


<p align="center">
  <img src="https://github.com/kashyap333/Types-of-Autoencoders/blob/main/Sparse_AE/output/loss.png" width=500 height=400 align="center">
</p>



### Project Structure

```bash
autoencoders
|--Sparse_AE
   |--output
   |  |--images
   |--sparse_encoder.py
   |--helper.py

```

### Run cmd

You can provide your own values for experimenting

```cmd
python sparse_encoder.py --epochs=(custom_value) --add_sparse=(True/False) --reg_parameter=(custom_value) --batch_sz=(custom_value) --lr=(custom_value)
```

### Notes

- The sparse loss used can be any type of loss, but ensure its suitable for the task and also where in the NN it is penalising.
- Model was trained on GPU but can also be run on CPU, since there is a lot of images, just use a part of the dataset and it should be sufficient.


