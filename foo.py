from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import torch
from tensorflow.keras.applications import InceptionV3, ResNet50, InceptionResNetV2, DenseNet121
# Assuming 'model' is your pre-trained Keras model
n_features = 128    
model =  ResNet50(weights='imagenet', include_top=False, pooling='avg')
backbone_type = 'keras'
 # Print the output shape of the last layer before the dense laye

# Remove the top layer
base_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Get the number of features from the last layer
in_features = base_model.output.shape[-1]

# Add a new Dense layer with n_features outputs
new_output = Dense(n_features, name='new_fc')(base_model.output)

# Create a new model
backbone = Model(inputs=base_model.input, outputs=new_output)

x = torch.randn(1, 224, 224, 3)  # Example input, adjust according to your needs
output = model.predict(x)
print(output.shape)
import torch
from torch import nn
from transformers import AutoModel

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
backbone = AutoModel.from_pretrained("facebook/dinov2-base", output_hidden_states=True).to(device)

# Get the number of features in the last layer
in_features = backbone.config.hidden_size

# Define the number of output features you want
n_features = 1000  # Change this to your desired number of features

# Create a new linear layer
new_head = nn.Linear(in_features, n_features).to(device)

# Replace the classifier
# Note: DINOv2 doesn't have a classifier by default, so we're adding one
backbone.classifier = new_head

# If you want to freeze the backbone and only train the new head
for param in backbone.parameters():
    param.requires_grad = False
for param in backbone.classifier.parameters():
    param.requires_grad = True

# Example forward pass
def forward(x):
    outputs = backbone(x)
    # Use the last hidden state
    last_hidden_state = outputs.last_hidden_state
    # Global average pooling
    pooled_output = torch.mean(last_hidden_state, dim=1)
    # Pass through the new classifier
    logits = backbone.classifier(pooled_output)
    return logits

# Example usage
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # Adjust input size as needed
output = forward(input_tensor)
print(output.shape)  # Should print torch.Size([1, n_features])
