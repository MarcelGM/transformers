import sys
sys.path.append('examples/pytorch/summarization/')
from run_summarization_BART_Extended import main

#training_args_json = "/home/ec2-user/moymarce/transformers/training_args/training_args_double_encoder.json"
training_args_json = "./training_args/training_args_bart_double.json"
sys.argv.append(training_args_json)

main()