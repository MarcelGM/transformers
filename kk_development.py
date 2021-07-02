import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, BartTokenizer, BartForConditionalGeneration, BartExtendedForConditionalGeneration, BartConfig, BartExtendedModel

# Loading trained model
PATH = "/home/ec2-user/moymarce/transformers/checkpoints/5-source_oracle-double/"
tokenizer = BartTokenizer.from_pretrained(PATH)
model = BartExtendedForConditionalGeneration.from_pretrained(PATH)

# Generate example
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs. I hope one day they start eating healthier. Maybe a plant-based diet would be enough. <knw> My friends are cool"
summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=20, early_stopping=True, use_cache=False)
print('Predicted text by model:', [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids], sep='\n')

# Add special token
tokenizer.add_tokens(['<knw>'], special_tokens=True)
# Initialize special tokens
knw_token_id = tokenizer.convert_tokens_to_ids(['<knw>'])[0] #50265
pad_id = tokenizer.pad_token

# Tokenize inputs into batch
ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs. I hope one day they start eating healthier. Maybe a plant-based diet would be enough. <knw> My friends are cool"
KNOWLEDGE = "My friends are cool"
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
knowledge_inputs = tokenizer([KNOWLEDGE], max_length=1024, return_tensors='pt')

tokenizer([ARTICLE_TO_SUMMARIZE, KNOWLEDGE], max_length=1024, return_tensors='pt')

# Masking
X = torch.Tensor([[1,2,3,4], [5,6,7,8]])
indexes = ((X == 3) + (X == 6)).nonzero(as_tuple=True)

knw_token_id = tokenizer.convert_tokens_to_ids(['<knw>'])[0] #50265
pad_id = tokenizer.pad_token

for row, ind in zip(X, indexes[1]):
    ind = (row == tokenizer.decode('<knw>')).nonzero()
    print('row', row, ind)
    print(row[ind:])
    row[ind:] = torch.zeros(row[ind:].size())