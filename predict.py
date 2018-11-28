from util import predict, process_image, load_checkpoint
import argparse

def call_predict(image_path, checkpoint, json_file, topk, gpu):

    print ('image_path: {},\t checkpoint:{},\t json_file: {},\t topk: {},\t gpu: {}\n'.format(image_path, checkpoint, json_file, topk, gpu))

    print ('Predicting...')
    probs, classes, names = predict(image_path, checkpoint, json_file, topk, gpu)
    print ('Predicting Done.\n')

    results = []
    if len(names):
        results = list(zip(probs, names))
    else:
        results = list(zip(probs, classes))

    max_guess = max(results,key=lambda item:item[0])

    print ('Max Guess => Prob.: {}, Class: {}\n'.format(max_guess[0], max_guess[1]))
    print ('top{} Results:'.format(topk))
    print (results)
    
    
    
def main():
    # Define command line arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('image_path', metavar='image_path', type=str, help='Path to the Image')
	parser.add_argument('checkpoint', metavar='checkpoint', type=str, help='Load trained model checkpoint from file')
	parser.add_argument('--gpu', action='store_true', help='Enable GPU, preferable if available', default=False)
	parser.add_argument('--topk', type=int, help='Return top K most likely classes', default=5)
	parser.add_argument('--category_names', type=str, help='mapping of categories to real names JSON file', default='')

	args, _ = parser.parse_known_args()
	
	call_predict(args.image_path, args.checkpoint, args.category_names, args.topk, args.gpu)

if __name__ == "__main__":
    main()