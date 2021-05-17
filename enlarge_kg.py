from allennlp.predictors.predictor import Predictor
from read_pickle import EntityLinker
from argparse import ArgumentParser
import allennlp_models.syntax.srl
import networkx as nx
import spacy
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


# excluding all entities that are longer than 1 word and do not contain any noun/verb
def contains_noun(entity, pos_tag):
    entity = entity.split()
    if len(entity) > 1:
        pos = []
        for token in pos_tag:
            for ent in entity:
                if token.text == ent:
                    pos.append(token.pos_)
        if ('VERB' not in pos) or ('NOUN' not in pos):
            return False
    return True


# pos tagging of the sentence + entities recognised from the sentence
def remove_tokens(pos_sentence_one, rand_1_entities):
    to_be_removed = ['AUX', 'PRON', 'SCONJ', 'DET', 'ADV', 'PART', 'ADP']
    to_be_removed_verbs = ['like', 'love', 'hate']
    tmp_rem = []
    remove = []
    for ent, val in rand_1_entities.items():
        if not contains_noun(ent, pos_sentence_one):
            remove.append(ent)
    for rem in remove:
        rand_1_entities.pop(rem)
    for token in pos_sentence_one:
        for ent, val in rand_1_entities.items():
            if token.pos_ in to_be_removed and token.text == ent:
                tmp_rem.append(ent)
            elif token.text in to_be_removed_verbs and token.text == ent:
                tmp_rem.append(ent)
    for ent in tmp_rem:
        if ent in rand_1_entities:
            rand_1_entities.pop(ent)
    return rand_1_entities


def find_max_entities(final_ent_1, num_entities=2):
    importance = []
    while num_entities > 0 and bool(final_ent_1):
        max_1 = max(final_ent_1, key=final_ent_1.get)
        importance.append(max_1)
        final_ent_1.pop(max_1)
        num_entities -= 1
    return importance


def find_entities(sentence, linker):
    entities = linker._get_entity_mentions(sentence)
    final_entities = {}
    for ent, val in entities.items():
        final_entities[ent] = val[0]['score']
    return final_entities


def find_in_doc(entities, articles):
    final_sentences = []
    for paragraph in articles:
        # split in sentences
        sentences = paragraph.split('.')
        sent = []
        found = [False for i in range(len(entities))]
        for sentence in sentences:
            for i in range(len(entities)):
                entity = entities[i]
                if entity in sentence:
                    found[i] = True
                    sent.append(sentence)
        if False in found:
            continue
        else:
            final_sentences.append(sent)
    return final_sentences


def enlarge_kg(sentences, linker):
    for sentence in sentences:
        sentence = sentence.split('\n')[0]
        # TODO: number of entities to find in sentence needs to be proportionate to the sentence length
        pos_tag = nlp(sentence)
        print(pos_tag)
        all_entities = find_entities(sentence, linker)
        print(all_entities)


def filter_sentences(articles, entities, predictor):
    if len(articles) == 0:
        return
    filtered_sentences = {}
    for paragraph in articles:
        for sentence in paragraph:
            if '<abstract>' in sentence:  # first sentence of the abstract, probably what we want to keep
                sentence = sentence.split('>')[1]

                # TODO: replace spacy dependency parsing with allennlp semantic role labeling
                sentence_tagging = predictor.predict(sentence=sentence)

                adding = {}
                for token in sentence_tagging:
                    if token.text in entities and (token.dep_ == 'nsubj' or token.dep_ == 'dobj' or token.dep_ == 'pobj'):
                        adding[token.dep_] = sentence
                        filtered_sentences[token.text] = adding
                        # TODO: add sentence to graph (through verb?)
    print(filtered_sentences)
    return filtered_sentences


def adding_to_kg(graph, new_nodes):
    graph_nodes = list(graph.nodes(data='name'))
    count = 0
    for entity in new_nodes:
        try:
            # deal with multiple entities with the same name
            entity_id = [node[0] for node in graph_nodes if node[1] == entity]
        except:
            continue
        for ent in entity:
            graph.add_node(count, name=entity[ent])
            if ent == 'nsubj':
                graph.add_edges_from([count, entity_id, {'name': 'definition'}])
            else:
                graph.add_edges_from([count, entity_id, {'name': 'fact'}])
            count += 1


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('-k', '--knowledge_graph', help='Pickle of the KG')
    ap.add_argument('-p', '--persona_file', help='PersonaChat file')
    ap.add_argument('-d', '--document_file', help='Document file')
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")
    args = ap.parse_args()
    with open(args.document_file, 'r') as in_f:
        articles = in_f.readlines()
    # persona_traits = create_persona_traits(args.persona_file)
    with open(args.persona_file, 'r') as pers_file:
        persona_traits = pers_file.readlines()
    knowledge_graph = nx.read_gpickle(args.knowledge_graph)
    linker = EntityLinker()
    final_write = []
    count = 0
    for trait in persona_traits:
        if count == 10:
            break
        trait = trait.rstrip('\n')
        print(trait)
        entities = find_entities(trait, linker)
        pos_tag = nlp(trait)
        final_entities = remove_tokens(pos_tag, entities)
        ent_import = find_max_entities(final_entities, 2)
        print(ent_import)
        # finding sentences in document
        doc_sentences = find_in_doc(ent_import, articles)
        print(len(doc_sentences))
        filtered_sentences = filter_sentences(doc_sentences, ent_import, predictor)
        final_write.append([trait, ent_import, doc_sentences, len(doc_sentences)])
        adding_to_kg(knowledge_graph, filtered_sentences)
        count += 1
        # enlarge_kg(doc_sentences, linker)
        # be prepared, more than one entity with the same name!
    with open('final_results.txt', 'w') as out_file:
        writer = csv.writer(out_file)
        for trait in final_write:
            writer.writerow(trait)
