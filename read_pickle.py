from SPARQLWrapper import SPARQLWrapper, JSON
from argparse import ArgumentParser
from bert_score import score
from random import choice, randint
from tqdm import tqdm
import networkx as nx
import numpy as np
import operator
import requests
import spacy
import time
import sys

np.random.seed(42)


class EntityLinker():
    def __init__(self):#, conf):
        super(EntityLinker, self)

    def _get_entity_mentions(self, utterance, context=None, ignore_names=[]):

        try:
            annotations = requests.post(
                #self._params["linker_endpoint"],
                "http://localhost:8080",
                json={
                    "text": utterance.replace(".", "").replace("-", ""),
                    "properties": {},
                    "profanity": {
                        "http://www.wikidata.org/prop/direct/P31": [
                            "http://www.wikidata.org/entity/Q184439",
                            "http://www.wikidata.org/entity/Q608"
                        ],
                        "http://www.wikidata.org/prop/direct/P106": [
                            "http://www.wikidata.org/entity/Q488111",
                            "http://www.wikidata.org/entity/Q1027930",
                            "http://www.wikidata.org/entity/Q1141526"
                        ]
                    },
                    "annotationScore": -8.5,
                    "candidateScore": -10
                },
                #timeout=self._params.get("timeout", None)
            )

            if annotations.status_code != 200:
                print(
                    "[Entity Linker]: Status code %d\n"
                    "Something wrong happened. Check the status of the entity linking server and its endpoint." %
                    annotations.status_code)
                annotations = {}
            else:
                annotations = annotations.json()
        except requests.exceptions.RequestException as e:
            print("[Entity Linker]:"
                         "Something wrong happened. Check the status of the entity linking server and its endpoint.")
            annotations = {}
        return annotations


    def __call__(self, *args, **kwargs):

        annotations = kwargs.get("annotations", None)
        if not annotations:
            return None

        p_annotation = annotations.get("processed_text")

        # entity linking for the current user utterance (ignore when the user says their name)
        context = DictQuery(kwargs.get("context", {}))
        user_ents = None
        if p_annotation and not self._name_intent.search(p_annotation):
            user_ents = self._get_entity_mentions(p_annotation, context=context)
        return {
            "entity_linking": user_ents,
        }


FIND_IMPORTANT_NODES = """
    SELECT ?item ?itemLabel ?outcoming ?sitelinks ?incoming {
        wd:%s wdt:P279 ?item .
        ?item wikibase:statements ?outcoming .
        ?item wikibase:sitelinks ?sitelinks .
           {
           SELECT (count(?s) AS ?incoming) ?item WHERE {
               wd:%s wdt:P279 ?item .
               ?s ?p ?item .
               [] wikibase:directClaim ?p
          } GROUP BY ?item
      }
      SERVICE wikibase:label { bd:serviceParam wikibase:language "en" . }.
    } ORDER BY DESC (?incoming)
"""


def find_id(url):
    entity = url.split('/')
    entity = entity[len(entity)-1]
    return entity


def find_entities(value):
    scores = []
    ids = []
    for val in value:
        id_ = val['entityLink']['identifier']
        score = val['score']
        ids.append(id_)
        scores.append(score)
    #scores = np.array(scores)
    min_val = min(ids, key=len)
    main_entity = find_id(min_val)
    return main_entity # returns Q item from Wikidata


def mid_way_step(G, start, destination, k, path):
    entities = [start]
    index_ = 1
    while k > 0:
        tmp_scores = {}
        for ent_ind in range(index_, len(path)-k):
            current_entity = path[ent_ind]
            P, R, F1 = score([start], [current_entity], lang='en', rescale_with_baseline=True)
            tmp_scores[current_entity] = F1.item()
        entity = max(tmp_scores, key=tmp_scores.get)
        entities.append(entity)
        k -= 1
        start = entity
        index_ = path.index(entity)+1
    entities.append(destination)
    return entities


# check all sentences that contain the 2 chosen entities
def check_in_pchat(persona_file, rand_1_n, rand_2_n):
    p_one = []
    p_two = []
    with open(persona_file, 'r') as p_file:
        for line in p_file:
            if 'persona' in line:
                try:
                    line = line.split(':')[1]
                    if rand_1_n in line:
                        p_one.append(line)
                    elif rand_2_n in line:
                        p_two.append(line)
                except:
                    pass
    return p_one, p_two


# find if there's a path between 2 random nodes
def find_paths(G, persona_file, rand_1, rand_2):
    # choosing random nodes + saving their names
    rand_1_n = rand_1
    rand_2_n = rand_2
    temp_search = list(G.nodes(data='name'))
    try:
        rand_1 = [item[0] for item in temp_search if item[1] == rand_1_n][0]
        rand_2 = [item[0] for item in temp_search if item[1] == rand_2_n][0]
    except:
        return False, [], []

    pairs = []
    possible_short_paths = []
    other_paths = []
    try:
        tmp_short = nx.all_shortest_paths(G, source=rand_1, target=rand_2)
        count = 0
        for path in tmp_short:
            if count >= 20:
                break
            possible_short_paths.append(path)
        tmp_others = nx.all_simple_paths(G, source=rand_1, target=rand_2, cutoff=20)
        for path in tmp_others:
            other_paths.append(path)
    except:
        return False, [], []
    if len(possible_short_paths) > 0:
        # print('Path from {} to {}'.format(rand_1_n, rand_2_n))
        short_rand_var = np.random.randint(0, len(possible_short_paths))
        short_rand = [G.nodes[p]['name'] for p in possible_short_paths[short_rand_var]]
        # print('Shortest path: {}'.format(short_rand))
        # 3: number of mid-way entities we want to find
        # 2: start + end entities
        mid_entities = []
        if len(short_rand) > (3+2):
            mid_entities = mid_way_step(G, rand_1_n, rand_2_n, 3, short_rand)
    if len(other_paths) > 0:
        other_rand_var = np.random.randint(0, len(other_paths))
        other_rand = [G.nodes[p]['name'] for p in other_paths[other_rand_var]]
    return True, short_rand, mid_entities


def scan_docs(ent_1, ent_2, articles, check_ent_1, k=2):
    mentions_one = []
    mentions_two = []
    for sentence in tqdm(articles):
        if check_ent_1 and ent_1 in sentence and len(mentions_one) < k:
            mentions_one.append(sentence)
        if ent_2 in sentence and len(mentions_two) < k:
            mentions_two.append(sentence)
    return mentions_one, mentions_two


def find_documents(entities, articles):
    entity_1 = entities[0]
    count = 0
    mentions_one = []
    mentions_two = []
    for entity in entities:
        if entity == entity_1:
            continue

        entity_2 = entity
        # TODO: entity_1 should be searching for mentions if unseen before
        if count == 0:
            temp_one, temp_two = scan_docs(entity_1, entity_2, articles, True)
        else:
            temp_one, temp_two = scan_docs(entity_1, entity_2, articles, False)
        if len(temp_one) > 0:
            mentions_one.append(temp_one)
            mentions_two.append(temp_two)
        # print(mentions_one)
        # print(mentions_two)
        # TODO: find importance score between mentions!
        mentions_one = mentions_two
        entity_1 = entity
        count += 1
    return mentions_one, mentions_two


def add_edges(G):
    for node in tqdm(list(G.nodes)):
        neighbours = G.successors(node)
        for neighbour in neighbours:
            G.add_edges_from([(neighbour, node, {'name': 'has_subclass'})])
        nx.write_gpickle(G, "pickled_graph_complete")


# NOT NEEDED ANYMORE
def remove_useless_links(G):
    # fnid meaningful connections
    sparql_client = SPARQLWrapper("http://query.wikidata.org/sparql", returnFormat=JSON, agent='User-Agent: Mozilla/5.0')
    sparql_client.setTimeout(604800)
    for node in tqdm(list(G.nodes)):
        # deleting root node and all its connections
        if node == 'root':
            G.remove_node(node)

        # find the important nodes, remove connections from current node to every other non-important
        sparql_client.setQuery(FIND_IMPORTANT_NODES % (node, node))
        results = sparql_client.queryAndConvert()['results']['bindings']
        time.sleep(3)
        if len(results) > 0:
            # check for node's neighbours
            neighbours = G.successors(node)
            remove = []
            for neighbour in neighbours:
                if neighbour not in results:
                    remove.append(neighbour)
            for rem in remove:
                G.remove_edge(node, rem)
        nx.write_gpickle(G, "pickled_graph_complete")


def remove_tokens(pos_sentence_one, pos_sentence_two, rand_1_entities, rand_2_entities):
    to_be_removed = ['AUX', 'PRON', 'SCONJ', 'DET', 'ADV', 'PART']
    tmp_rem = []
    for token in pos_sentence_one:
        for ent, val in rand_1_entities.items():
            if token.pos_ in to_be_removed and token.text in ent:
                tmp_rem.append(ent)
    for ent in tmp_rem:
        if ent in rand_1_entities:
            rand_1_entities.pop(ent)
    tmp_rem = []
    for token in pos_sentence_two:
        for ent, val in rand_2_entities.items():
            if token.pos_ in to_be_removed and token.text in ent:
                tmp_rem.append(ent)
    for ent in tmp_rem:
        if ent in rand_2_entities:
            rand_2_entities.pop(ent)
    return rand_1_entities, rand_2_entities


# two entities in each trait, see how paths are
def find_max_entities(final_ent_1, final_ent_2, num_entities=2):
    rand_1 = []
    rand_2 = []
    while num_entities > 0 and bool(final_ent_1) and bool(final_ent_2):
        max_1 = max(final_ent_1, key=final_ent_1.get)
        max_2 = max(final_ent_2, key=final_ent_2.get)
        rand_1.append(max_1)
        rand_2.append(max_2)
        final_ent_1.pop(max_1)
        final_ent_2.pop(max_2)
        num_entities -= 1
    return rand_1, rand_2


def eval_conKB(entities):
    for i in range(len(entities)-1):
        entity_1 = entities[i]
        entity_2 = entities[i+1]

        out_conv = self.convKB(conv_input)
    return


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('persona_file', help='PersonaChat file as input')
    ap.add_argument('article_file', help='Article file as input')
    args = ap.parse_args()
    persona_file = args.persona_file
    article_file = args.article_file

    with open(args.persona_file, 'r') as p_f:
        persona_traits = p_f.readlines()
    # with open(persona_file, 'r') as pers_file:
    #     for line in pers_file:
    #         if 'persona' in line:
    #             try:
    #                 line = line.split(':')
    #                 persona_traits.append(line[1])
    #             except:
    #                 pass

    with open(article_file, 'r') as ar_file:
        articles = ar_file.readlines()

    G = nx.read_gpickle("pickled_graph")
    print('Nodes: {}\tEdges: {}'.format(len(list(G.nodes)), len(list(G.edges))))
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_sm")

    count_found = 0
    count_not_found = 0
    try:
        while(True):
            # choosing random persona traits from the same persona/next one
            rand_1_num = randint(0, len(persona_traits)-1)
            rand_2_num = rand_1_num + 1

            # find entities in the traits
            linker = EntityLinker()
            rand_1_entities = linker._get_entity_mentions(persona_traits[rand_1_num])
            rand_2_entities = linker._get_entity_mentions(persona_traits[rand_2_num])
            pos_sentence_one = nlp(persona_traits[rand_1_num])
            pos_sentence_two = nlp(persona_traits[rand_2_num])
            # removing verbs, pronouns from our entities
            rand_1_entities, rand_2_entities = remove_tokens(pos_sentence_one, pos_sentence_two, rand_1_entities, rand_2_entities)
            # choose entity (HOW?)
            final_ent_1 = {}
            for ent, val in rand_1_entities.items():
                final_ent_1[ent] = val[0]['score']
            final_ent_2 = {}
            for ent, val in rand_2_entities.items():
                final_ent_2[ent] = val[0]['score']
            empty_one = not final_ent_1
            empty_two = not final_ent_2
            if empty_one or empty_two:
                continue
            rand_1, rand_2 = find_max_entities(final_ent_1, final_ent_2, 2)
            if len(rand_1) == 0 or len(rand_2) == 0:
                continue
            printing = False
            paths_storage = {(0, 0): (persona_traits[rand_1_num], persona_traits[rand_2_num])}
            for ent_1 in rand_1:
                for ent_2 in rand_2:
                    paths_storage[(ent_1, ent_2)] = []
                    if ent_1 == ent_2:
                        continue
                    found, shortest_path, entities = find_paths(G, persona_file, ent_1, ent_2)
                    # TODO: evaluate convKB model on our triples from entities
                    # eval_convKB(entities)
                    # TODO: search in documents the mentions of `found' entities (paired)
                    if found:
                        paths_storage[(ent_1, ent_2)] = (shortest_path, entities)
                        count_found += 1
                        # tuple of mentions
                        # mention = find_documents(entities, articles)
                        # print(entities)
                        # print(mention)
                        printing = True
                    else:
                        count_not_found += 1
            if printing:
                print(paths_storage)
    except KeyboardInterrupt:
        print('{}/{} - {}'.format(count_found, count_not_found, count_found/(count_not_found+count_found)))
        sys.exit()
    #remove_useless_links(G)
    # add_edges(G)


