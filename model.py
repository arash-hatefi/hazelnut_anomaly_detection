import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image



'''
AutoEncoder Class: 
    
    Functionality: Used for building an Auto Encoder Network for anomaly
                   ditection in HazelNut dataset
                   
    Inputs: It takes the device which the process of traning is going to run on
            it as input
    
    Public Methods:
        1) forward: Used for computing the output of the network for a given 
                    input
        2) get_train_log: Used for getting the train data information 
                          (loss, accuracy, etc) from the last training process
                          
        3) get_train_log: Used for getting the test data information 
                          (loss, accuracy, etc) from the last training process
                          
        4) train_model: Used for training the model for one epoch on the given 
                        dataloader and using the given optimization method
                        
        5) reset_log: Used for reseting the train and teat log befor the 
                      training process
                        
        6) test_model: Used for computing the test or validation dataloader
                       output 
    
        7) get_model_accuracy: Used for elementwise comparing the binary form 
                               of networks output with the ground truth image
                            
        8) get_loss_plots: Used for getting the loss plots from the previous 
                           training 
                           
        9) input_output_illustrations: returns the illustation of the input 
                                       image, output of the network, differecne 
                                       between input and output, and binary 
                                       form of the difference 
'''
 
class AutoEncoder(nn.Module):
    
    def __init__(self, device, apply_batch_normalization=False):
        
        super(AutoEncoder, self).__init__()

        self.device = device
        self.apply_batch_normalization = apply_batch_normalization
        self.__train_log = []

        if (apply_batch_normalization):

            self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                                         nn.BatchNorm2d(num_features=32),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                                         nn.BatchNorm2d(num_features=32),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=32),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
                                         nn.BatchNorm2d(num_features=64),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=64),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1),
                                         nn.BatchNorm2d(num_features=128),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=64),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
                                         nn.BatchNorm2d(num_features=32),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=100, kernel_size=(8,8), stride=1, padding=0),
                                         nn.BatchNorm2d(num_features=100),
                                         nn.LeakyReLU(negative_slope=0.01))
            
            
            self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=32, kernel_size=(8,8), stride=1, padding=0),
                            nn.BatchNorm2d(num_features=32),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                            nn.BatchNorm2d(num_features=64),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
                            nn.BatchNorm2d(num_features=128),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
                            nn.BatchNorm2d(num_features=64),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                            nn.BatchNorm2d(num_features=64),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                            nn.BatchNorm2d(num_features=32),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
                            nn.BatchNorm2d(num_features=32),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                            nn.BatchNorm2d(num_features=32),
                            nn.LeakyReLU(negative_slope=0.01),
                            nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4,4), stride=2, padding=1),
                            nn.BatchNorm2d(num_features=3),
                            nn.Sigmoid())

        

        else:

            self.encoder = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.Conv2d(in_channels=32, out_channels=100, kernel_size=(8,8), stride=1, padding=0),
                                         nn.LeakyReLU(negative_slope=0.01))
            
            
            self.decoder = nn.Sequential(nn.ConvTranspose2d(in_channels=100, out_channels=32, kernel_size=(8,8), stride=1, padding=0),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=(4,4), stride=2, padding=1),
                                         nn.LeakyReLU(negative_slope=0.01),
                                         nn.ConvTranspose2d(in_channels=32, out_channels=3, kernel_size=(4,4), stride=2, padding=1),
                                         nn.Sigmoid())
        

        self.to(self.device)
        

    
    def forward(self, x):
        
        x=self.encoder(x)
        x = self.decoder(x)
        
        return x
    

    def get_train_log(self):
        
        return self.__train_log
    

    def reset_log(self):
        
        self.__train_log = []
        self.__test_log = []




    def train_model(self, train_loader, optimizer, epoch):
        
        self.train()
        train_loss = 0
        for (data, _) in train_loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = self.__call__(data)
            loss = F.mse_loss(output, data, reduction='sum')
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
               
        average_loss = train_loss/len(train_loader.dataset)
        
        results = (average_loss, )
        
        self.__train_log.append((epoch, results))
                    
        return results
            
       
        
    def get_model_accuracy(self, test_loader, threshold=0.5):
        
        self.eval()
        sum_loss = 0
        total_elements = 0
        with torch.no_grad():
            for (data, target) in test_loader:
                data = data.to(self.device)
                output = self.__call__(data)
                difference = torch.sum(torch.abs((output-data).to("cpu")).pow(2),1)
                binary_difference = difference>threshold
                total_elements += np.size(np.array(binary_difference))
                sum_loss += np.sum(np.abs(np.array(binary_difference==torch.sum(target,1))))
        return sum_loss/total_elements


    def get_loss_plots(self):

        assert self.__train_log, "No trained model!"

        train_epochs = [log[0] for log in self.__train_log]
        train_loss = [log[1][0] for log in self.__train_log]

        plt.figure()
        plt.plot(train_epochs, train_loss, color='green', label='Average Loss on train data')
        plt.xlabel('Epoch')
        plt.ylabel('Average Epoch Loss')
        plt.legend()
        plt.show()

      
    def __get_output_for_single_image(self, network_input):
        
        self.eval()
        input_tensor = None
        if type(network_input) == PIL.PngImagePlugin.PngImageFile or type(network_input) == np.ndarray:
          input_tensor = transforms.ToTensor()(network_input)
        elif type(network_input) == torch.Tensor:
          input_tensor = network_input
        else:
          raise TypeError("Expected PIL Image or Torch Tensor as network_input but instead {} was received!".format(type(network_input)))
        
        network_input = input_tensor.view(([1]+list(input_tensor.size())))
        network_input = network_input.to(self.device)
        network_output = self.__call__(network_input).to("cpu")
        return network_output[0]

    
    def input_output_illustrations(self, image, threshold, figsize=(8,8)):

        output_tensor = self.__get_output_for_single_image(image)
        PIL_input = transforms.ToPILImage(mode="RGB")(image)
        PIL_output = transforms.ToPILImage(mode="RGB")(output_tensor)

        difference_tensor =torch.abs(torch.sum(transforms.ToTensor()(image) - output_tensor,0))
        binary_difference_tensor = (difference_tensor>threshold).float()

        PIL_difference = transforms.ToPILImage()(difference_tensor)
        PIL_difference_binary = transforms.ToPILImage()(binary_difference_tensor)

        plt.figure(figsize=figsize)

        plt.subplot(2, 2, 1)
        plt.imshow(PIL_input)
        plt.title("Network Input")
        plt.axis('off')
        plt.subplot(2, 2, 2)
        plt.imshow(PIL_output)
        plt.title("Network Output")
        plt.axis('off')
        plt.subplot(2, 2, 3)
        plt.imshow(PIL_difference, cmap="gray")
        plt.title("Single Channel Difference")
        plt.axis('off')
        plt.subplot(2, 2, 4)
        plt.imshow(PIL_difference_binary, cmap="gray")
        plt.title("Difference In Binary Form")
        plt.axis('off')

        plt.show()

    
    def get_difference_between_in_and_out_tensors(self, input_tensor):

        if type(input_tensor) != torch.Tensor:
          raise TypeError("Expected Torch Tensor as input_tensor but instead {} was received!".format(type(input_tensor)))

        return torch.abs(input_tensor - self.__get_output_for_single_image(network_input=input_tensor))        

