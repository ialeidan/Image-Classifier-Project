from util import get_loaders, build_model, training, save_checkpoint
import argparse


def train(data_dir, arch, hidden_units, output_size, dropout, lr, epochs, gpu, checkpoint):
	
	print ('Dir: {},\t Arch:{},\t HiddenUints: {},\t lr: {},\t Epochs: {},\t gpu: {}\n'.format(data_dir, arch, hidden_units, lr, epochs, gpu))
    
	print ('Loading Images from Directory...')
	trainloader, validloader, testloader, class_to_idx = get_loaders(data_dir)
	print ('Images Loaded.\n')
    
	print ('Building the Model...')
	model, criterion, optimizer = build_model(arch, hidden_units, output_size, dropout, lr)
	print ('Model Built.\n')
    
	print ('Beggining the Training...')
	model, optimizer = training(model, trainloader, validloader, epochs, 20, criterion, optimizer, gpu)
	print ('Training Done.\n')
    
	if checkpoint:
		print ('Saving the Checkpoint...')
		save_checkpoint(checkpoint, model, optimizer, arch, model.classifier[0].in_features, output_size, hidden_units, dropout, class_to_idx, epochs, lr)
		print ('Done.')

def main():
    # Define command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('data_directory', metavar='data_directory', type=str, help='Path to Image Dataset')
	parser.add_argument('--gpu', action='store_true', help='Enable GPU, preferable if available', default=False)
	parser.add_argument('--epochs', type=int, help='Number of Epochs', default=8)
	parser.add_argument('--arch', type=str, help='Model Architecture:{"vgg16", "alexnet"}', default='vgg16')
	parser.add_argument('--learning_rate', type=float, help='Learning Rate', default=0.0001)
	parser.add_argument('--hidden_units', type=int, help='Number of hidden units', default=4096)
	parser.add_argument('--save_dir', type=str, help='Save trained model checkpoint to file', default='')

	args, _ = parser.parse_known_args()
	
	train(args.data_directory, args.arch, args.hidden_units, 102, 
			0.5, args.learning_rate, args.epochs, args.gpu, args.save_dir)

if __name__ == "__main__":
    main()