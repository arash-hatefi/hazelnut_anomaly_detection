import torch
from torch import optim
import numpy as np
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from dataset import prepare_dataset
from model import AutoEncoder




###############################################################################
############################ Loading the Dataset ##############################
###############################################################################

print("Loading the dataset...")

TRAIN_DATA_DIRECTORY = r"./hazelnut/train"
TEST_DATA_DIRECTORY = r"./hazelnut/test"
GROUND_TRUTH_DIRECTORY = r"./hazelnut/ground_truth"

BATCH_SIZE = 32
SHUFFLE_DATASET = True

INPUT_SIZE = (128,128)

# Change this number for adding augmented data to dataset
N_AUGMENTED_DATA = 0

TRANSFORM_SEQ = None

TRANSFORM_SEQ = iaa.Sequential([iaa.geometric.Affine(rotate=(-45, 45))])

train_loader = prepare_dataset(image_dir=TRAIN_DATA_DIRECTORY,
                               imsize=INPUT_SIZE,
                               transform=TRANSFORM_SEQ,
                               n_augmented_data=N_AUGMENTED_DATA,
                               train_batch_size=BATCH_SIZE,
                               shuffle=SHUFFLE_DATASET)

test_loader = prepare_dataset(image_dir=TEST_DATA_DIRECTORY,
                              imsize=INPUT_SIZE,
                              binary_image_dir = GROUND_TRUTH_DIRECTORY,
                              shuffle=SHUFFLE_DATASET)
  
print("Loading completed!")




###############################################################################
############################## Training Process ###############################
###############################################################################


USE_BATCH_NORM = True
LEARNING_RATE = 1e-3
EPOCHS = 700
LOG_EVERY = 1
USE_GPU = True

device = torch.device("cuda:0" if torch.cuda.is_available() and USE_GPU else "cpu")

print("Building and training the network...")
 
autoencoder = AutoEncoder(device=device, apply_batch_normalization=USE_BATCH_NORM)
autoencoder.reset_log()

optimizer = optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS+1):
                        
    train_results = autoencoder.train_model(train_loader=train_loader, optimizer=optimizer, epoch=epoch)
    
    if LOG_EVERY != None:
        if not epoch % LOG_EVERY:
          print("Epoch {}:".format(epoch))
          print("\tAverage Train-Set Loss: {:.6f}".format(*train_results))

print("Training completed!")



###############################################################################
################################## Results ####################################
###############################################################################

## For getting the Training Loss Plot, uncomment the next line  
autoencoder.get_loss_plots()


## For getting the input-output difference and binary images uncommnet the next five lines
BINARY_THRESHOLD = 0.35
IMAGE_PATH = r"./hazelnut/test/crack/001.png"
image = np.array(Image.open(IMAGE_PATH))
image = iaa.Resize(({"height": INPUT_SIZE[0], "width": INPUT_SIZE[1]}))(image=image)
autoencoder.input_output_illustrations(image, threshold=BINARY_THRESHOLD)
      
    
## For getting the accuracy of network with respect to ground truth images, uncommnet the next two lines
BINARY_THRESHOLD = 0.35   
autoencoder.get_model_accuracy(test_loader, threshold=BINARY_THRESHOLD)

