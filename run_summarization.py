import sys
sys.path.append('examples/pytorch/summarization/')
from run_summarization_BART_Extended import main

# Setting GPUS for Phong's machines
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

#training_args_json = "/home/ec2-user/moymarce/transformers/training_args/training_args_double_encoder.json"
training_args_json = "./training_args/training_args_bart_double_varying_alpha.json"
sys.argv.append(training_args_json)

main()