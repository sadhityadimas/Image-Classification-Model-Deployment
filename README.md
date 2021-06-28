# Image-Classification-Model-Deployment
# Convert Trained Image Classification model into Tflite
# Dicoding-Image-Classification-Submission

* Dataset is provided by Kaggle uploaded by Puneet Bansal
* Link to File : https://www.kaggle.com/puneet6060/intel-image-classification
* The dataset can be fetch using kaggle library on colab 
* 13.500samples
* 80:20 train test split


## Library Used
* Tensorflow
* Keras
* matplotlib
* Numpy
* Pandas
* Kaggle

## Data set composition                 

|    | type   |   buildings |   forest |   mountain |   sea |   street |   total |
|---:|:-------|------------:|---------:|-----------:|------:|---------:|--------:|
|  0 | train  |        2191 |     2271 |       2512 |  2274 |     2382 |   11630 |
|  1 | val    |         437 |      474 |        525 |   510 |      501 |    2447 |

## Step by Step Model Building

### Display random images to examine
![](/img/download.png)

### Rescaling Image
### Image Augmenting (train set only)
* Rotation
* Horizontal flip
* Zoom range
* Shear
* Fill mode nearest
* Data generating
* resizing image to 150x150
* batch size 128
* class mode categorical
* 
### Build Convnet
* 1 hidden layer (perceptron 128 units)
* output layer 'Softmax'
* model = keras.Sequential([
*    layers.Conv2D(32, (3,3), activation = 'relu', input_shape= (150,150,3)),
*    layers.MaxPooling2D(pool_size=(2, 2)),
*    layers.Conv2D(64,(3,3), activation= 'relu'),
*    layers.MaxPooling2D(pool_size=(2, 2)),
*    layers.Conv2D(128,(3,3), activation= 'relu'),
*    layers.MaxPooling2D(pool_size=(2, 2)),
*    layers.Flatten(),
*    layers.Dropout(0.5),
*    layers.Dense(128, activation= 'relu'),
*    layers.Dense(5, activation= 'softmax')
* ])

### Adding Loss Function and Optimizer
* loss : categorical cross entropy
* optimizer adam
* metrics accuracy

### Adding Custom Early stop function using Callback from tensorflow 
* Model will stop training once val accuracy reaches atleast 85%


### Training the model
* epoch : 40
* steps per epoch : 20
* verbose : 1
* validation steps : 10
* callbacks
 
## Results
* Best Validation Loss: 0.41
* Best Validation Accuracy: 0.86

* Loss

![](/img/download%20(1).png)

* Accuracy

![](/img/download%20(2).png)


## Convert Model to Tflite
* Name trainde model to my model.tflite
