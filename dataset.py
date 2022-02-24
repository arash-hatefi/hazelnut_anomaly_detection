import torch
from torchvision import transforms, datasets
import numpy as np
import os
import PIL
from PIL import Image
import imgaug.augmenters as iaa



'''
HazelNutDataset Class: 
    
    Functionality: Used for building the HazelNut dataset
                   
    Inputs: PLI images
            Image size for the input of the network
            Binary PIL grond truth image (Optional)
            Sequential Augmentations (Optional)
            Number of augmented data (Optinal)
'''

class HazelNutDataset:


    def __init__(self, PIL_images, imsize, PIL_binary_images=None, sequential_augmentation=None, n_augmented_data=None):
        
        assert not(PIL_binary_images and sequential_augmentation), "sequential_augmentation can not be apply to PIL_binary_images!"

        resize = iaa.Resize(({"height": imsize[0], "width": imsize[1]}))
        resize_det = resize.to_deterministic()
        images = [np.array(PIL_image) for PIL_image in PIL_images]
        binary_images = [np.array(binary_image) for binary_image in PIL_binary_images] if PIL_binary_images else None
        
        if sequential_augmentation != None and n_augmented_data!=None:
            augmented_images = self.__get_augmented_images(images, sequential_augmentation, n_augmented_data)
            images.extend(augmented_images)

        images = resize_det.augment_images(images)
        if PIL_binary_images: binary_images = resize_det.augment_images(binary_images)

        self.images = [transforms.ToTensor()(image) for image in images]
        self.binary_images = [transforms.ToTensor()(binary_image) for binary_image in binary_images] if PIL_binary_images else [0] * len(images)
            
        self.samples = [(self.images[i], self.binary_images[i]) for i in range(len(self.images))]


    def __get_augmented_images(self, images, sequential_augmentation, n_augmented_data):

        indices = np.random.choice(len(images), n_augmented_data)
        seq_det = sequential_augmentation.to_deterministic()
        return seq_det.augment_images([images[i] for i in indices])


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, index):
        return self.samples[index]





'''
prepare_dataset Function:
    
        Functionality: Used for loading images set and building dataloaders
        
        Inputs: Images Directory
                Image size for the input of the network
                Binary PIL grond truth image (Optional)
                Sequential Augmentations (Optional)
                Number of augmented data (Optinal)
                Training Batch Size (Optional)
                Shuffle or Not (Optional)
                
                
'''

def prepare_dataset(image_dir, imsize, binary_image_dir=None, transform=None, n_augmented_data=None, train_batch_size=None, shuffle=True):
    
    image_paths=[]
    binary_image_paths = [] if binary_image_dir else None

    for image_gp in os.listdir(image_dir):
        image_gp_dir = image_dir + '/' + image_gp
        founded_image_paths=[]
        for image in os.listdir(image_gp_dir):
            founded_image_paths.append(image_gp_dir+'/'+image)
        if binary_image_dir:
            founded_binary_image_paths = []
            binary_image_gp_dir = binary_image_dir + '/' + image_gp
            if not os.path.isdir(binary_image_gp_dir):
                print("Warning: Some Directories in image_dir was not found in binary_image_dir! The contents will not be included in dataloader")
                continue
            for image in os.listdir(binary_image_gp_dir):
                founded_binary_image_paths.append(binary_image_gp_dir+'/'+image)
            binary_image_paths.extend(sorted(founded_binary_image_paths))
        image_paths.extend(sorted(founded_image_paths))

    if binary_image_dir:
      assert len(image_paths)==len(binary_image_paths), "Different number of data was found in image_dir and binary_image_dir"
    
    images = [np.array(Image.open(path)) for path in image_paths]
    binary_images = [np.array(Image.open(path)) for path in binary_image_paths] if binary_image_dir else None

    if shuffle:
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images = [images[i] for i in range(len(images))]
        if binary_image_dir:
            binary_images = [binary_images[i] for i in range(len(binary_images))]
    
    dataset = HazelNutDataset(PIL_images=images, imsize=imsize, PIL_binary_images=binary_images, sequential_augmentation=transform, n_augmented_data=n_augmented_data)

    if train_batch_size==None: train_batch_size=len(images)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_batch_size)

    return dataloader

    