from SPARQLWrapper import SPARQLWrapper, JSON
from argparse import ArgumentParser
from tqdm import tqdm
import networkx as nx
import numpy as np
import spacy
import json
import time
import csv
import sys


FIND_FREEBASE_ID = """
    SELECT ?item
    WHERE
    {
      wd:%s wdt:P646 ?item.
      SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
"""

def create_persona_traits(persona_file):
    persona_traits = []
    with open(persona_file, 'r', encoding='utf-8') as pers_file:
        pers_file = pers_file.readlines()
    for line in pers_file:
        if 'persona' in line:
            try:
                line = line.rstrip('\n')
                line = line.split(':')
                persona_traits.append([line[1]])
            except:
                pass
    return persona_traits


# takes in input an array of things to write
def write_file(write_element, name_file, del_='\t'):
    with open(name_file, 'w') as out_file:
        writer = csv.writer(out_file, delimiter=del_)
        for elem in write_element:
            writer.writerow(elem)


def create_triples_fb_id(entities_fb, original_triples, relation_fb):
    with open(relation_fb, 'r') as rel_f:
        relation_fb = rel_f.readlines()
    triples = []
    count = 0
    for orig_trip in tqdm(original_triples):
        try:
            left_entity = orig_trip[0].replace('_', ' ')
            left_entity = entities_fb[left_entity]
            right_entity = orig_trip[2].replace('_', ' ')
            right_entity = entities_fb[right_entity]
            relation_num = np.random.randint(0, len(relation_fb)-1)
            relation = relation_fb[relation_num].split('\t')[0]
            triples.append([left_entity, relation, right_entity])
        except:
            print('Not found: ', count)
            count += 1
            continue
    write_file(triples, 'triples_fb.txt')


# foreach entity, create triple (e_1, r, e_2)
def create_triples(G, entities):
    triples = []
    graph_nodes = list(G.nodes(data='name'))
    for entity in tqdm(entities):
        entity = "".join(entity)
        try:
            # deal with multiple entities with the same name
            entity_id = [node[0] for node in graph_nodes if node[1] == entity]
        except:
            continue
        entity = entity.replace(' ', '_')
        for ent in entity_id:
            for neigh in list(G.neighbors(ent)):
                ent_2 = G.nodes[neigh]['name']
                ent_2 = ent_2.replace(' ', '_')
                relation = G.edges[ent, neigh]['name']
                triple = [entity, relation, ent_2]
                triples.append(triple)
    # write_file(triples, 'kg_triples.txt')
    return triples


# find occurrences of entity in persona traits
def find_node_in_persona(node, persona_traits):
    paragraph = []
    for trait in persona_traits:
        if node in trait:
            trait = trait.replace('\n', ' ')
            if trait not in paragraph:
                paragraph.append(trait)
    paragraph = ' '.join(paragraph)
    return paragraph


# use SpaCy POS tagging to analyse nodes
def analyse_node(node, nlp):
    tagged_node = nlp(node)
    exclude = ['AUX', 'PRON', 'SCONJ', 'DET', 'ADV', 'PART']
    for t_node in tagged_node:
        if t_node.pos_ in exclude:
            return False
    return True


def create_entities(G, persona_traits, nlp, sparql_client):
    # set to keep unique entities only
    und_ents = []
    ent_id = []
    und_ents_to_text = []
    unds_long_t = []
    fb_ents = {}
    count = 0
    for node in tqdm(list(G.nodes)):
        wikidata_id = node
        node = G.nodes[node]['name']
        if not analyse_node(node, nlp):
            continue
        paragraph = find_node_in_persona(node, persona_traits)
        if not paragraph:
            continue
        sparql_client.setQuery(FIND_FREEBASE_ID % wikidata_id)
        results = sparql_client.queryAndConvert()['results']['bindings']
        for res in results:
            fb_id = res['item']['value']
            fb_ents[node] = fb_id
            print(wikidata_id, fb_id, node)
            ent_id.append([fb_id, str(count)])
            write_file(ent_id, 'entity2id.txt')
            time.sleep(3)
        tmp_long_t = [node, paragraph]
        und_ent = node.replace(" ", "_")
        tmp = [und_ent, node]
        und_ents.append([und_ent])
        und_ents_to_text.append(tmp)
        unds_long_t.append(tmp_long_t)
        count += 1
    write_file(und_ents, 'entities.txt')
    # write_file(und_ents_to_text, 'entity2text.txt')
    # write_file(unds_long_t, 'entity2textlong.txt')
    return und_ents, fb_ents


def create_relations(relations):
    write_file(relations, 'relations.txt')
    tuples_r = []
    for relation in relations:
        tmp = [relation]
        tmp.append(relation.replace('_', ' '))
        tuples_r.append(tmp)
    write_file(tuples_r, 'relation2text.txt')


if __name__ == '__main__':
    endpoint_url = "http://query.wikidata.org/sparql"
    sparql_client = SPARQLWrapper(endpoint_url, returnFormat=JSON, agent='User-Agent: Mozilla/5.0')
    sparql_client.setTimeout(604800)
    ap = ArgumentParser()
    ap.add_argument('knowledge_graph', help='Pickle of the KG')
    ap.add_argument('persona_file', help='PersonaChat file')
    ap.add_argument('entities_file', help='Enities file')
    ap.add_argument('freebase_entities_file', help='Freebase Entities file')
    ap.add_argument('-f', '--freebase_relation', help='Relation2id file for Freebase')
    args = ap.parse_args()

    G = nx.read_gpickle(args.knowledge_graph)
    nlp = spacy.load("en_core_web_sm")
    # persona_traits = create_persona_traits(args.persona_file)
    with open(args.persona_file, 'r') as in_f:
        persona_traits = in_f.readlines()
    entities, fb_entities = None, None
    with open(args.entities_file, 'r') as ent:
        entities = ent.readlines()
        print(len(entities))
    with open(args.freebase_entities_file, 'r') as fb:
        fb_entities = json.load(fb)

    count = 0
    for key, val in fb_entities.items():
        count += 1
        if count % 500 == 0:
            print(key, val)
    print(count)
    sys.exit()
    # write_file(persona_traits, 'persona_traits.txt')
    relations = ['subclass_of', 'has_subclass']
    create_relations(relations)
    # entities, fb_entities = create_entities(G, persona_traits, nlp, sparql_client)
    # with open('fb_entities.json', 'w') as out_:
    #     json.dump(fb_entities, out_)

    entities_ = []
    for entity in entities:
        entity = entity.rstrip('\n')
        entity = entity.replace('_', ' ')
        entities_.append([entity])
    triples = create_triples(G, entities_)
    create_triples_fb_id(fb_entities, triples, args.freebase_relation)