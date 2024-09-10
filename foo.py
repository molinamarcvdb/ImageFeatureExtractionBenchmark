from privacy_benchmark import initialize_model, SiameseNetwork

network_name = 'resnet50'
backbone_model, backbone_type, processor = initialize_model(network_name)
        
model = SiameseNetwork(backbone_model, backbone_type, processor, in_channels=3,  n_features = 768)

for name, param in model.named_parameters(): 
    if 'fc' in name and 'backbone' not in name:
        print(name)

