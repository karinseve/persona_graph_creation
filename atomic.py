from allennlp.models import basic_classifier
import allennlp
from argparse import ArgumentParser
from allennlp.commands import elmo
from tqdm import tqdm
import numpy as np
import torch
import h5py
import csv
import sys

def create_persona_traits(persona_file):
    persona_traits = []
    with open(persona_file, 'r') as pers_file:
        for line in pers_file:
            if 'persona' in line:
                try:
                    line = line.split(':')
                    persona_traits.append(line[1])
                except:
                    pass
    return persona_traits


def create_and_save(args):
    persona_traits = create_persona_traits(args.persona_file)
    with open('persona_traits.txt', 'w') as out_file:
        writer = csv.writer(out_file)
        for trait in persona_traits:
            writer.writerow(trait)


if __name__ == '__main__':
    ap = ArgumentParser()
    # ap.add_argument('persona_file', help='PersonaChat file')
    args = ap.parse_args()

    # do first time only
    # create_and_save(args)

    # TODO: 300-dim static GloVe embeddings (open file)
    # with open('glove.840B.300d.txt', 'r') as emb:
    #     glove_embeddings = emb.readlines()
    # elm_emb = h5py.File('elmo_embeddings.txt', 'r')
    # TODO: concatenate GloVe with ELMo files


    # tensor of BERT embeddings for atomic dataset
    elm_emb = torch.load('commonsense-kg-completion/bert_model_embeddings/nodes-lm-atomic/atomic_bert_embeddings.pt')
    # k_elm = list(elm_emb.keys())


    # TODO: encoder: Bi-GRU (AllenNLP GRUseq2vec)
    # TODO: decoder: Uni-GRU with softmax  (AllenNLP basic classifier?)

    basic_classifier.BasicClassifier(self,
        vocab: allennlp.data.vocabulary.Vocabulary,
        text_field_embedder: allennlp.modules.text_field_embedders.text_field_embedder.TextFieldEmbedder,
        seq2vec_encoder: allennlp.modules.seq2vec_encoders.seq2vec_encoder.Seq2VecEncoder,
        seq2seq_encoder: allennlp.modules.seq2seq_encoders.seq2seq_encoder.Seq2SeqEncoder=None,
        feedforward: Optional[allennlp.modules.feedforward.FeedForward]=None,
        dropout: float=None,
        num_labels: int=None,
        label_namespace: str='labels',
        initializer: allennlp.nn.initializers.InitializerApplicator=<allennlp.nn.initializers.InitializerApplicator object at 0x7f966c3be0f0>,
        kwargs,
    ) -> None