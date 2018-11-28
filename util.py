import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image
import json


def get_loaders(data_dir):
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# Define your transforms for the training, validation, and testing sets
	train_transforms = transforms.Compose([transforms.RandomRotation(30),
										   transforms.RandomResizedCrop(224),
										   transforms.RandomHorizontalFlip(),
										   transforms.ToTensor(),
										   transforms.Normalize([0.485, 0.456, 0.406], 
																[0.229, 0.224, 0.225])])

	valid_transforms = transforms.Compose([transforms.Resize(256),
										  transforms.CenterCrop(224),
										  transforms.ToTensor(),
										  transforms.Normalize([0.485, 0.456, 0.406], 
															   [0.229, 0.224, 0.225])])

	test_transforms = transforms.Compose([transforms.Resize(256),
										  transforms.CenterCrop(224),
										  transforms.ToTensor(),
										  transforms.Normalize([0.485, 0.456, 0.406], 
															   [0.229, 0.224, 0.225])])


	# Load the datasets with ImageFolder
	train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
	valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
	test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)


	# Using the image datasets and the trainforms, define the dataloaders
	trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
	validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=32)
	testloader = torch.utils.data.DataLoader(test_datasets, batch_size=32)
	
	return trainloader, validloader, testloader, train_datasets.class_to_idx
	
def label_mapping(json_file):
	with open(json_file, 'r') as f:
		cat_to_name = json.load(f)
	return cat_to_name

def build_model(arch, hidden_units, output_size, dropout, lr):
	if arch == 'vgg16':
		model = models.vgg16(pretrained=True)
	elif arch == 'alexnet':
		model = models.alexnet(pretrained=True)
	else:
		raise ValueError('Unexpected arch', arch)
        
        
	# Freeze parameters so we don't backprop through them
	for param in model.parameters():
		param.requires_grad = False

	# Adding new classifier
	input_size = model.classifier[0].in_features
	
	from collections import OrderedDict
	classifier = nn.Sequential(OrderedDict([
							  ('fc1', nn.Linear(input_size, hidden_units)),
							  ('relu1', nn.ReLU()),
							  ('Dropout', nn.Dropout(p=dropout)),
							  ('fc2', nn.Linear(hidden_units, hidden_units)),
							  ('relu2', nn.ReLU()),
							  ('Dropout', nn.Dropout(p=dropout)),
							  ('fc3', nn.Linear(hidden_units, output_size)),
							  ('output', nn.LogSoftmax(dim=1))
							  ]))

	model.classifier = classifier
	
	criterion = nn.NLLLoss()
	optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
	
	return model, criterion, optimizer
	
	
# Training Function
def training(model, trainloader, validloader, epochs, print_every, criterion, optimizer, gpu=False):
    epochs = epochs
    print_every = print_every
    steps = 0

    # change to device
    if gpu:
        model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
			
            if gpu:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(model, validloader, criterion, gpu)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0
            
                # Make sure training is back on
                model.train()
                
    return model, optimizer
 
# Validation Function
def validation(model, validloader, criterion, gpu=False):
    test_loss = 0
    accuracy = 0
    for images, labels in validloader:
		
        if gpu:
            images, labels = images.to('cuda'), labels.to('cuda')

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return test_loss, accuracy
	
	
def save_checkpoint(checkpoint, model, optimizer, arch, input_size, output_size, hidden_units, dropout, class_to_idx, epoch, lr):
	# Save the checkpoint 
	checkpoint_dict = {'arch': arch,
				  'input_size': input_size,
				  'output_size': output_size,
				  'hidden_units': hidden_units,
				  'dropout': dropout,
				  'state_dict': model.state_dict(),
				  'class_to_idx': class_to_idx,
				  'optimizer_state_dict': optimizer.state_dict,
				  'epoch': epoch,
				  'lr': lr
				 }
	torch.save(checkpoint_dict, checkpoint)
	
	
def load_checkpoint(checkpoint):
    checkpoint_dict = torch.load(checkpoint)
    
    if checkpoint_dict['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint_dict['arch'] == 'alexnet':
        model = models.alexnet(pretrained=True)
    else:
        raise ValueError('Unexpected arch', arch)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # Adding new classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(checkpoint_dict['input_size'], checkpoint_dict['hidden_units'])),
                              ('relu1', nn.ReLU()),
                              ('Dropout', nn.Dropout(p=checkpoint_dict['dropout'])),
                              ('fc2', nn.Linear(checkpoint_dict['hidden_units'], checkpoint_dict['hidden_units'])),
                              ('relu2', nn.ReLU()),
                              ('Dropout', nn.Dropout(p=checkpoint_dict['dropout'])),
                              ('fc3', nn.Linear(checkpoint_dict['hidden_units'], checkpoint_dict['output_size'])),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.load_state_dict(checkpoint_dict['state_dict'])
    model.class_to_idx = checkpoint_dict['class_to_idx']
    return model, checkpoint_dict['epoch'], checkpoint_dict['lr']
	
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Process a PIL image for use in a PyTorch model
    im = Image.open(image_path)
    
    # Resize using shortest side.
    width, height = im.size   
    new_size = max(width, height) * 256 // min(width, height)
    shape = (256, new_size)
    if width > height:
        shape = (new_size, 256)
    im = im.resize(shape, Image.ANTIALIAS)
    
    width, height = im.size   # Get dimensions
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2

    im = im.crop((left, top, right, bottom))
    
    trans = transforms.Compose([ transforms.ToTensor() ])
    im = trans(im)
    
    np_image = np.array(im)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean) / std
    
    #print (np_image)
    #print (np_image.shape)
    return np.transpose(np_image, (2, 0, 1))
	

def predict(image_path, checkpoint, json_file, topk=5, gpu=False):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    img = process_image(image_path)
    
    model, e, lr = load_checkpoint(checkpoint)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    if gpu:
        model.to('cuda')
    model.eval()
	
    if gpu:
        img_i = torch.FloatTensor(img).cuda()
    else:
        img_i = torch.FloatTensor(img)
    img_i.unsqueeze_(0)

    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_i)

    ps = torch.exp(output)
    t_topk = torch.topk(ps, topk)
	
    classes = np.asarray(t_topk[1])[0]
    names = []
    if json_file:
        cat_to_name = label_mapping(json_file)
        for c in classes: 
            for m in model.class_to_idx:
                if model.class_to_idx[m] != c:
                    continue
                names.append(cat_to_name[m])

    return (np.asarray(t_topk[0])[0], np.asarray(t_topk[1])[0], names)