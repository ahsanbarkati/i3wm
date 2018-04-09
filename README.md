# i3 fancy face lock

This is i3 lock with face unlock feature. You need to train the model by giving your images.
The grayer is a script that transforms the images to grayscale and 50x50.

```
python grayer.py <positive images directory> <negative images directory>
```

Next train your model. This will save the trained model in current directory.
```
python train.py
```

Now use it to predict.

```
python predict <name of model> <test image>
```
