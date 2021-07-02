import sys
sys.path.append('examples/pytorch/summarization/')
from run_summarization_BART_Extended import main

training_args_json = "/home/ec2-user/moymarce/transformers/training_args.json"
sys.argv.append(training_args_json)

main()