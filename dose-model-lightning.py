import torch
import torch.nn as nn 
import torch.nn.functional as F # Gives us Activation Functions
import torch.optim as optim

import lightning as L
from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

class BasicLightning(L.LightningModule):
    
    def __init__(self): # Create and initialize the weights and biases
    
        super().__init__() # Call the initialization method for the parent class, nn.Module
        
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # Setting require+grad to True means this variable should be optimized
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        
        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
        self.learning_rate = 0.01 # Because we are using Lightning function to improve the Learning Rate
    
    def forward(self, input): # Forward path through the neural network by taking an input value and calculating the output value with weights, biases, and activation functions
        
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relu_output = top_relu_output * self.w01
        
        input_to_bottom_relu = input * self.w10 + self.b10
        bottom_relu_output = F.relu(input_to_bottom_relu)
        scaled_bottom_relu_output = bottom_relu_output * self.w11
        
        input_to_final_relu = scaled_top_relu_output + scaled_bottom_relu_output + self.final_bias
        
        output = F.relu(input_to_final_relu)
        
        return output
    
    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss = F.mse_loss(output_i, label_i)
        
        return loss
            
if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_inputs = torch.tensor([0.0, 0.5, 1.0], device=device)
    train_labels = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    dataset = TensorDataset(train_inputs, train_labels)
    dataloader = DataLoader(dataset)
    
    model = BasicLightning().to(device)
    trainer = L.Trainer(max_epochs=34) # Lightning let us add additional epochs right after were we left off
    tuner = L.pytorch.tuner.Tuner(trainer)
    
    lr_find_results = tuner.lr_find(model, 
                                        train_dataloaders=dataloader,
                                        min_lr=0.001,
                                        max_lr=1.0,
                                        early_stop_threshold=None)

    new_lr = lr_find_results.suggestions()
    
    print(f"lr_find() suggests {new_lr:.5f} for the learning rate.")
    
    model.learning_rate = new_lr
    
    trainer.fit(model, train_dataloaders=dataloader)
    
    print(f"{model.final_bias.item():.4f}")
    # trainer_class = ModelTrainer(model, train_inputs, train_labels)
    # trainer_class.train()
    
    # input_doses = torch.linspace(start=0, end=1, steps=11).to(device)
    
    # output_value = model(input_doses) # By default it calls the forward method
    
    # sns.set(style="whitegrid")
    # sns.lineplot(
    #     x=input_doses.detach().cpu(),
    #     y=output_value.detach().cpu(), # detach has been called to create a new tensor that only has the values and is not for taking the gradient
    #     color='green',
    #     linewidth=2.5
    # )
    # plt.ylabel('Effectiveness')
    # plt.xlabel('Dose')