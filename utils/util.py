# multi-lingual model
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# mono-lingual model
from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np


def sbert_cossim(multilingual_model, src, tgt, batch_size):
    with torch.no_grad():
        src_emb = multilingual_model.encode(src, convert_to_tensor=True, batch_size=batch_size)
        tgt_emb = multilingual_model.encode(tgt, convert_to_tensor=True, batch_size=batch_size)
    cos_scores = cos_sim(src_emb, tgt_emb).cpu().numpy()
    return np.diag(cos_scores) * 100


def huggingface_cossim(monolingual_model, src, rtt):
    model, tokenizer = monolingual_model
    src_inputs = tokenizer(src, padding=True, truncation=True, return_tensors='pt')
    rtt_inputs = tokenizer(rtt, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        _src_emb = model(**src_inputs, return_dict=True).last_hidden_state
        _rtt_emb = model(**rtt_inputs, return_dict=True).last_hidden_state
    cos_scores = []
    for i in range(len(_src_emb)):
        cos_scores.append(cos_sim(_src_emb[i][0].cpu().numpy(), _rtt_emb[i][0].cpu().numpy()).numpy()[0][0])
    return np.array(cos_scores) * 100


def sbert_model_selector(nickname):
    model_book = {
        'distil': 'distiluse-base-multilingual-cased-v2',
        'mpnet': 'paraphrase-multilingual-mpnet-base-v2',
        'enmini': 'all-MiniLM-L12-v2'
    }
    fullname = model_book[nickname]
    model = SentenceTransformer(fullname)
    return model


def huggingface_model_selector(nickname):

    model_book = {
        "kobert_multi": "BM-K/KoSimCSE-bert-multitask",
        "koroberta_multi": "BM-K/KoSimCSE-roberta-multitask",
        "kobert": "BM-K/KoSimCSE-bert",
        "koroberta": "BM-K/KoSimCSE-roberta",
        "enbert": "princeton-nlp/sup-simcse-bert-base-uncased",
        "enroberta": "princeton-nlp/sup-simcse-roberta-base",
        "enmini": 'sentence-transformers/all-MiniLM-L12-v2',
        'distil': 'sentence-transformers/distiluse-base-multilingual-cased-v2',
        'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

    }
    fullname = model_book[nickname]
    model = AutoModel.from_pretrained(fullname)
    tokenizer = AutoTokenizer.from_pretrained(fullname)
    return model, tokenizer
