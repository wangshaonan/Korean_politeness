# bert-base-kor
import torch
import torch.nn.functional as F
import numpy as np
import sys

from transformers import BertTokenizerFast, BertForMaskedLM
bert_tokenizer = BertTokenizerFast.from_pretrained('kykim/bert-kor-base')
bert_model = BertForMaskedLM.from_pretrained('kykim/bert-kor-base').eval()

def convert_logits_to_probs(logits, input_ids):
    """
    input:
        logits: (1, n_word, n_vocab), GPT2 outputed logits of each word
        input_inds: (1, n_word), the word id in vocab
    output: probs: (1, n_word), the softmax probability of each word
    """
    probs = F.softmax(logits[0], dim=1)
    n_word = input_ids.shape[1]

    res = []
    for i in range(n_word):
        res.append(probs[i, input_ids[0][i]].item())
    return np.array(res).reshape(1, n_word)

def encode(tokenizer, text_sentence, add_special_tokens=False):
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    return input_ids

out = open(sys.argv[2], 'w')
for line in open(sys.argv[1]):
    print(line)
    text_sentence = line.strip()
    if not len(text_sentence):
        out.write('\n')
        continue
    input_ids = encode(bert_tokenizer, text_sentence)
    print(input_ids)
    print(bert_tokenizer.convert_ids_to_tokens(input_ids[0]))
    out.write('Prob_Sent '+' '.join(bert_tokenizer.convert_ids_to_tokens(input_ids[0]))+'\n')
    with torch.no_grad():
        logits = bert_model(input_ids)[0]
        #print(logits.shape) #8*42000

    prob = convert_logits_to_probs(logits, input_ids)
    sent_prob = np.prod(prob)
    print(sent_prob, prob)
    out.write(str(sent_prob) + ' ' + ' '.join([str(i) for i in prob[0]]) + '\n')
