import sys
print(sys.version_info)


##import transformers.src.transformers.models.bart.modeling_bart_edited as BartExtended
from transformers import BartTokenizer, BartForConditionalGeneration, BartExtendedForConditionalGeneration, BartConfig

pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
tokenizer.add_tokens(['<knw>'], special_tokens=True)

print(tokenizer.encode(['<knw>']))

import torch

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs. I hope one day they start eating healthier. Maybe a plant-based diet would be enough. <knw> My friends are cool"
KNOWLEDGE = "My best friends are cool and healthy"
input_article = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
input_knowledge = tokenizer([KNOWLEDGE], max_length=1024, return_tensors='pt')

# Masking
# X = torch.Tensor([[1,2,3,4], [5,6,7,8]])
# indexes = ((X == 3) + (X == 6)).nonzero(as_tuple=True)
#
# knw_token_id = tokenizer.convert_tokens_to_ids(['<knw>'])[0]) #50265
# pad_id = tokenizer.pad_token
#
# for row, ind in zip(X, indexes[1]):
#     ind = (row == tokenizer.decode('<knw>')).nonzero()
#     print('row', row, ind)
#     print(row[ind:])
#     row[ind:] = torch.zeros(row[ind:].size())
from copy import deepcopy
extended_config = deepcopy(pretrained_bart.config)
extended_config.is_extended = True

bart_extended = BartExtendedForConditionalGeneration(extended_config)

# Adding <knw> token
tokenizer.add_tokens(['<knw>'], special_tokens=True)
pretrained_bart.resize_token_embeddings(len(tokenizer))
bart_extended.resize_token_embeddings(len(tokenizer))

# Generate Summary with bart pretrained
# summary_ids = pretrained_bart.generate(input_article['input_ids'], num_beams=4, max_length=20, early_stopping=True)
# print('Predicted text by pretrained model:', [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

# Generate Summary with bart extended
summary_ids = bart_extended.generate(input_article['input_ids'], input_knowledge['input_ids'], num_beams=4, max_length=20, early_stopping=True, use_cache=False)
print('Predicted text by Extended model:', [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])