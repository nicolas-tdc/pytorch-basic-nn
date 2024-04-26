import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

import matplotlib.pyplot as plt
import seaborn as sns

class BasicNN(nn.Module):

    def __init__(self):

        super().__init__()

        # First layer parameters.
        self.w00 = nn.Parameter(torch.tensor(1.7), requires_grad=False)
        self.b00 = nn.Parameter(torch.tensor(-0.85), requires_grad=False)
        self.w01 = nn.Parameter(torch.tensor(-40.8), requires_grad=False)
        # Second layer parameters.
        self.w10 = nn.Parameter(torch.tensor(12.6), requires_grad=False)
        self.b10 = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self.w11 = nn.Parameter(torch.tensor(2.7), requires_grad=False)
        # Final layer parameters
        self.final_bias = nn.Parameter(torch.tensor(-16.), requires_grad=False)

    def forward(self, input):

        # First layer.
        input_to_top_relu = input * self.w00 + self.b00
        top_relu_output = F.relu(input_to_top_relu)
        scaled_top_relou_output = top_relu_output * self.w01
        # Second layer.
        input_to_bot_relu = input * self.w10 + self.b10
        bot_relu_output = F.relu(input_to_bot_relu)
        scaled_bot_relou_output = bot_relu_output * self.w11
        # Final layer.
        input_to_final_relu = scaled_bot_relou_output + scaled_top_relou_output + self.final_bias

        return F.relu(input_to_final_relu)
    

# Create input values.
input_doses = torch.linspace(start=0, end=1, steps=11)

# Create model and get output.
model = BasicNN()
output_values = model(input_doses)

# Display graph.
sns.set_theme(style="whitegrid")
sns.lineplot(x=input_doses, y=output_values, color="green", linewidth=2.5)
plt.ylabel('Effectiveness')
plt.xlabel('Dose')
plt.show()
