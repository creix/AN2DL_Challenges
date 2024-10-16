# Artificial Neural Network and Deep Learning Challenges
Challenges for the 22/23 Artificial Neural Network and Deep Learning course at Polimi

## Team Members
- Giacomo Ballabio - 10769576
- Christian Biffi - 10787158
- Alberto Cavallotti - 10721275

---

# First Challenge - Plant Health Classification Challenge

The first challenge consisted in classifying plant images as healthy or unhealthy.

The objective was to design a network for optimal classification of plant health using a dataset of 5100 images. Our approach evolved from basic CNNs to advanced transfer learning techniques, achieving a top accuracy of 93% in the development phase.

## Key Components

### 1. Dataset Analysis
- Examined dataset composition and removed outliers
- Applied one-hot encoding for labels
- Experimented with 80/20 and 90/10 train/validation splits

### 2. Initial Approach
- Implemented a basic hand-crafted CNN inspired by LeNet
- Experimented with various configurations (filter numbers, kernel sizes, FC layer composition)
- Implemented batch normalization and data augmentation techniques

### 3. Transfer Learning and Fine-tuning
We explored several pre-trained models from Keras Applications, including:
- Xception
- MobileNetV2
- NASNetMobile
- ResNet (various dimensions)
- EfficientNetV2 (S variant performed best)
- ConvNeXt (Base variant showed great potential)

### 4. Preprocessing and Data Augmentation
- Upscaled images to match ImageNet dimensions (128x128 for EfficientNetV2, 224x224 for ConvNeXt)
- Implemented advanced augmentation techniques:
  - MixUp
  - CutMix
  - RandAugment from KerasCV

### 5. Fully Connected Network
- Experimented with different FC layer configurations
- Best results achieved with two FC layers (512 and 256 neurons) with Batch Normalization and Dropout

### 6. Fine-tuning
- Unfroze entire pre-trained network for EfficientNetV2
- Partial unfreezing for ConvNeXt Base due to GPU memory constraints

### 7. Test Time Augmentation (TTA)
Applied various TTA techniques, including:
- Flipping
- Rotation
- Central crop
- Brightness and contrast adjustments
- Translation

## Results

Our best models achieved 93% accuracy in the development phase and 89% in the test phase.

Overall, we positioned at the 9th position as a team.

| Model                        | FC Layers          | Dropout | Accuracy (%) |
|------------------------------|--------------------|---------|--------------|
| ConvNeXt Base                 | 512+256 neurons    | 0.16667 | 93           |
| EfficientNetV2 S              | 512+256 neurons    | 0.16667 | 93           |
| ConvNeXt Base                 | 512 neurons        | 0.16667 | 93           |

---

# Second Challenge - Time Series Forecasting Challenge

The second challenge was focusing on time series forecasting using various deep learning approaches.

The objective was to train a neural network for time series forecasting on a concealed test set containing 60 time series. The task involved predicting the next 9 points in the first phase and 18 points in the second phase for each time series. Our approach evolved from basic LSTM models to more advanced architectures, achieving our best Test MSE of 0.01018455.

## Key Components

### 1. Dataset Analysis and Processing
- Analyzed 48,000 time series, each with 2776 points, categorized into 6 groups
- Computed statistics including category distribution and time series lengths
- Split dataset into training and validation sets (90/10 ratio)
- Created sequences with a window size of 200 and a stride of 20

### 2. Initial Approach
- Implemented basic networks with LSTM and GRU layers
- Experimented with various configurations (unit numbers, Conv1D layers, Batch Normalization, Dropout)
- Best initial model: Bidirectional LSTM with 128 units, followed by Conv1D layers and another Bidirectional LSTM

### 3. Advanced Architectures
We explored several advanced architectures, including:
- Autoencoder structures
- ResNet-like architecture
- Ensemble models
- Transformer-based models
- Linear models
- N-BEATS architecture

### 4. Preprocessing and Encoding Techniques
- Maintained original normalization (0 to 1)
- Experimented with Robust Scaler (no improvement)
- Implemented Time2Vec encoding

### 5. Attention Mechanism
- Applied attention mechanism to various models
- Aimed to improve model understanding by focusing on different parts of the time series

### 6. Autoregression vs Direct Approach
- Implemented autoregression approach
- Retrained models to predict one point at a time
- Modified prediction function to concatenate predictions

### 7. Final Model Iterations
- Fine-tuned best-performing models
- Experimented with combining Time2Vec and Attention mechanisms

## Results
Our best models achieved the following MSE scores:

| Model                             | Validation MSE  | Test MSE     |
|----------------------------------- |----------------|--------------|
| BLSTM + 3 Conv1D + BLSTM           | 0.008520       | 0.01018455   |
| BLSTM + AutoEncoder + BLSTM        | 0.00637854     | 0.01058138   |
| ResNet 2 blocks + 4 Conv1D layers  | 0.00843970     | 0.01069852   |
