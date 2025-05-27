import torch
import torch.nn as nn 
import torch.nn.functional as F # Gives us Activation Functions
import torch.optim as optim

import matplotlib.pyplot as plt
import seaborn as sns


class BasicNN(nn.Module):
    
    def __init__(self): # Create and initialize the weights and biases
        super().__init__() # Call the initialization method for the parent class, nn.Module
        
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False) # Setting require+grad to True means this variable should be optimized
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        
        self.final_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        
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
        
        
class ModelTrainer:
    
    def __init__(self, model, train_inputs, train_labels):
        self.model = model
        self.train_inputs = train_inputs
        self.train_labels = train_labels
        
    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        
        print("Final bias, before optimization: " + str(self.model.final_bias.data) + "\n")
        
        for epoch in range(100):
            
            total_loss = 0
            
            for iteration in range(len(self.train_inputs)):
                
                input_i = self.train_inputs[iteration]
                label_i = self.train_labels[iteration]
                
                output_i = self.model(input_i)
                
                loss = (output_i - label_i) ** 2 # Loss function: Squared residual
                
                loss.backward() # to calculate the derivative of the loss function with respect to the parameter(s) we want to optimize
                
                total_loss += float(loss)
                
            if (total_loss < 0.0001):
                print("Num steps: " + str(epoch))
                break
            
            optimizer.step() # to take a small step toward a better value for parameter(s) we want to optimize. It has access to derivatives stored in the model.
            optimizer.zero_grad() # to zero out the derivatives that we're storing in model
            
            print("Step: " + str(epoch) + ", Final Bias: " + str(model.final_bias.data.detach().cpu()) + "\n")
            
if __name__ == "__main__":
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    train_inputs = torch.tensor([0.0, 0.5, 1.0], device=device)
    train_labels = torch.tensor([0.0, 1.0, 0.0], device=device)
    
    model = BasicNN().to(device)
    trainer_class = ModelTrainer(model, train_inputs, train_labels)
    trainer_class.train()
    
    
    input_doses = torch.linspace(start=0, end=1, steps=11).to(device)
    
    output_value = model(input_doses).detach().cpu() # By default it calls the forward method
    
    sns.set(style="whitegrid")
    sns.lineplot(
        x=input_doses.detach().cpu(),
        y=output_value.detach().cpu(), # detach has been called to create a new tensor that only has the values and is not for taking the gradient
        color='green',
        linewidth=2.5
    )
    plt.ylabel('Effectiveness')
    plt.xlabel('Dose')
