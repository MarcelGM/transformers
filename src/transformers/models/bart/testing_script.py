
import transformers.models.bart.modeling_bart_edited as BartExtended
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
tokenizer.add_tokens(['<knw>'], special_tokens=True)

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs. I hope one day they start eating healthier. Maybe a plant-based diet would be enough."
KNOWLEDGE = "My friends are cool"
inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
knowledge_inputs = tokenizer([KNOWLEDGE], max_length=1024, return_tensors='pt')



bart_extended = BartExtended.BartExtendedForConditionalGeneration(pretrained_bart.config)


pretrained_bart.resize_token_embeddings(len(tokenizer))
bart_extended.resize_token_embeddings(len(tokenizer))

# Generate Summary with bart pretrained
summary_ids = pretrained_bart.generate(inputs['input_ids'], num_beams=4, max_length=20, early_stopping=True)
print('Predicted text by pretrained model:', [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

# Generate Summary with bart extended
summary_ids = bart_extended.generate(inputs['input_ids'], num_beams=4, max_length=20, early_stopping=True, use_cache=False)
print('Predicted text by Extended model:', [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])