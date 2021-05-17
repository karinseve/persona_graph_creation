"""
    Wikidata requests
"""
from rdflib import ConjunctiveGraph, URIRef, Literal
from SPARQLWrapper import SPARQLWrapper, JSON
from utils.dict_query import DictQuery
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
import numpy as np
import requests
import time
import sys
#import yaml


RETRIEVE_ITEM_PROPERTIES_QUERY = """
        PREFIX wdt: <http://www.wikidata.org/prop/direct/>
        PREFIX wd: <http://www.wikidata.org/entity/>
        SELECT ?s ?p ?o WHERE {
            {
                ?s wdt:P31 ?type.
                ?type wdt:P279 ?t1.
                ?t1 wdt:P279 ?t2.
                ?t2 wdt:P279 ?t3.
                ?t3 wdt:P279 %s.
            }
            UNION
            {
                ?s wdt:P31 ?type.
                ?type wdt:P279 ?t1.
                ?t1 wdt:P279 ?t2.
                ?t2 wdt:P279 %s.
            }
            UNION
            {
                ?s wdt:P31 ?type.
                ?type wdt:P279 ?t.
                ?t wdt:P279 %s.
            }
            UNION
            {
                ?s wdt:P31 ?type.
                ?type wdt:P279 %s.
            }
            UNION
            {
                ?s wdt:P31 %s.
            }
            VALUES ?p {%s}
            ?s ?p ?o.
        }
        """

# def build_item_data_graph(item_type, item_properties, item_data_graph_path, endpoint_url, max_num_objects, create=True):
#     item_data_graph = ConjunctiveGraph("Sleepycat")

#     item_data_graph.open(item_data_graph_path, create=create)

#     for item_property in item_properties:
#         item_data_query = "SELECT ?subj ?prop WHERE { ?subj ?prop ?o. }"
#         # item_data_query = RETRIEVE_ITEM_PROPERTIES_QUERY % (
#         #     item_type, item_type, item_type, item_type, item_type, item_property)
#         sparql_client = SPARQLWrapper(endpoint_url, returnFormat=JSON)
#         sparql_client.setTimeout(604800)
#         sparql_client.setQuery(item_data_query)
#         results = sparql_client.queryAndConvert()
#         num_bindings = len(results["results"]["bindings"])
#         added_triples = defaultdict(lambda: defaultdict(lambda: 0))
#         for i, binding in enumerate(results["results"]["bindings"]):
#             print("[{}/{}]".format(i + 1, num_bindings))
#             subject = URIRef(binding["s"]["value"])
#             predicate = URIRef(binding["p"]["value"])
#             if binding["o"]["type"] == "literal":
#                 object_ = Literal(binding["o"]["value"], datatype=binding["o"]["datatype"])
#             else:
#                 object_ = URIRef(binding["o"]["value"])
#             if max_num_objects is not None:
#                 if added_triples[subject][predicate] < max_num_objects:
#                     triple = (subject, predicate, object_)
#                     added_triples[subject][predicate] += 1
#                     item_data_graph.add(triple)
#             else:
#                 triple = (subject, predicate, object_)
#                 item_data_graph.add(triple)

#     item_data_graph.close()



class EntityLinker():
    def __init__(self):#, conf):
        super(EntityLinker, self)#.__init__(conf)
        #self._params = conf["parameters"]
        #self._wikidata_client = WikidataClient(self._params["wikidata_endpoint"])
        # with open(os.path.join(MERCURY_PATH, self._params["ignored_spans"]), 'r', encoding='UTF-8') as fh:
        #     ignored_spans = yaml.load(fh)
        # self._ignored_spans = set(ignored_spans["ignored_spans"])

        # load the name intent pattern to avoid linking user name
        # with open(os.path.join(MERCURY_PATH, self._params["intents"]), 'r', encoding='UTF-8') as fh:
        #     intents_data = yaml.load(fh)
        # self._name_intent = _compile_patterns(intents_data["name"])
        # self.topic_resolver = EntityTopicResolver()
        # self.gazetteer = LinkerGazetteer(
        #     self._params["gazetteer_file"],
        #     self._wikidata_client,
        #     self._params["gazetteer_mongo_info"],
        #     self._params["gazetteer_mongo_default_keys"]
        # )

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

        # add ready-made entities from the gazetteer
        # gazetteer_ents = self.gazetteer.match_entities(utterance)
        # annotations.update(gazetteer_ents)

        # prev_topic = None
        # if context:
        #     prev_topic = context.get("current_state.last_state.state.nlu.annotations.topics", None)
        #     # the user explicitly asked to talk about something else
        #     # check coherence topic from bot attributes
        #     # ASSUMPTION: most recent mentioned topic
        #     if prev_topic == "change" or prev_topic is None:
        #         prev_topic = context.get("current_state.state.bot_states.coherence_bot.bot_attributes.topic", None)

        return annotations
        #return self._refine_annotations(annotations, prev_topic, ignore_names)

    # def _enhance_annotations(self, span_annotations, curr_topic):
    #     """Replace wikidata prefixes, get person genders for all annotations pertaining to
    #     one span in the text."""
    #     rep_span_annotations = []

    #     for annotation in span_annotations:
    #         annotation["entityLink"].update({
    #             "identifier": get_prefixed_identifier(annotation["entityLink"]["identifier"]),
    #             "types": [get_prefixed_identifier(t) for t in annotation["entityLink"]["types"]],
    #             "properties": {
    #                 get_prefixed_identifier(p):
    #                     [get_prefixed_identifier(pi) for pi in annotation["entityLink"]["properties"][p]]
    #                 for p in annotation["entityLink"]["properties"]
    #             }
    #         })

    #         # ignore entities that don't have any associated type
    #         if not annotation["entityLink"]["types"]:
    #             continue

            # assume that the entity is not related to the current topic
            # is_topic_related_entity = False
            # if curr_topic is not None:
            #     for check_topic, check_foo in self.topic_resolver.topic_mappings:
            #         matched_topic = check_foo(annotation)
            #         if matched_topic and check_topic == curr_topic:
            #             is_topic_related_entity = True
            #             break

            #         # fictional character can be associated to any of topic like books, movies and video games
            #         # try to see if the current topic is among them
            #         if check_topic is None and matched_topic:
            #             if curr_topic in ["movies", "books", "video games"]:
            #                 is_topic_related_entity = True
            #                 break
            # else:
            #     # we don't know the current topic -- no filter active
            #     is_topic_related_entity = True

            # # skip entity if not related and it's associated to an ambiguous mention
            # if not is_topic_related_entity and len(span_annotations) > 1:
            #     continue

    #         rep_span_annotations.append(annotation)

    #         # add gender annotation for persons, if found
    #         if wikidata.human_type in annotation["entityLink"]["types"]:
    #             try:
    #                 gender = get_gender(annotation["entityLink"]["identifier"], self._wikidata_client)
    #             except IOError as e:
    #                 gender = None
    #                 logger.error(
    #                     'Wikidata problem getting gender for %s: %s' % (annotation["entityLink"]["identifier"], e))
    #             if gender:
    #                 annotation["entityLink"]["properties"][wikidata.gender_property] = get_prefixed_identifier(gender)

    #     return rep_span_annotations

    # def _refine_annotations(self, annotations, topic=None, ignore_names=[]):
    #     """Go through all annotations, normalize and enhance all of them,
    #     delete invalid ones, remove list for unambiguous annotations."""
    #     ignore_names = set([name.lower() for name in ignore_names])
    #     refined_annotations = {}

    #     for span_text in annotations:
    #         if (span_text not in self._ignored_spans and span_text.lower() not in ignore_names):
    #             span_annotations = annotations[span_text]
    #             refined_annotations[span_text] = self._enhance_annotations(span_annotations, topic)
    #             # if we don't have any annotations left, remove the annotation
    #             if len(refined_annotations[span_text]) == 0:
    #                 del refined_annotations[span_text]
    #             # if we only have a single annotation for the current span of text, remove the list
    #             elif len(refined_annotations[span_text]) == 1:
    #                 refined_annotations[span_text] = refined_annotations[span_text][0]

    #     return refined_annotations

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

        # link entities from the last bot utterance
        # bot_ents = None
        # if context:
        #     context = DictQuery(context)
        #     last_bot = context.get('current_state.state.last_bot')
        #     last_bot_resp = next(iter(context.get('current_state.last_state.state.response', {'': ''}).values()))
        #     last_bot_resp = re.sub(r'<[^>]*>', '', last_bot_resp)  # remove SSML tags before linking

        #     username = context.get('user_attributes.user_name')
        #     ignored_names = [username] if username else []  # ignore user name

        #     # use entities provided by ontologies bot directly
        #     if last_bot == "ontology_bot":
        #         bot_ents = annotations.get("bot_entities", {})
        #         # check if we added coherence driver -- if so, tag entities from there
        #         driver = context.get('current_state.last_state.state.response_edits.driver_added')
        #         if driver:
        #             driver = re.sub(r'<[^>]*>', '', driver)  # remove SSML tags before linking
        #             driver_ents = self._get_entity_mentions(driver, ignore_names=ignored_names)
        #             for driver_ent in driver_ents.values():  # shift offset for these
        #                 driver_ent['span']['startOffset'] += len(last_bot_resp) - len(driver)
        #                 driver_ent['span']['endOffset'] += len(last_bot_resp) - len(driver)
        #             bot_ents.update(driver_ents)
        #     # tag entities in the bot's response if the last bot was something else
        #     elif last_bot_resp:
        #         bot_ents = self._get_entity_mentions(last_bot_resp, ignore_names=ignored_names)

        return {
            "entity_linking": user_ents,
            #"bot_entities": bot_ents
        }

# P31: instance of
# P279: subclass of
SELECT_QUERY = """ 
    SELECT ?item ?itemLabel
    WHERE {
    ?item wdt:P31/wdt:P279* wd:%s.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
    LIMIT 10

"""
SELECT_PARENT_QUERY = """ 
    SELECT ?item ?itemLabel
    WHERE {
    wd:%s wdt:P279 ?item.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
    }
"""

PROPERTY_QUERY = """
SELECT ?property
WHERE {
wd:%s wdt:P1963 ?property.
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } 
}
"""


CHECK_CONNECTION_QUERY = """
SELECT ?p ?pLabel (count (*) as ?count) {
?s ?pd ?o .
?p wikibase:directClaim ?pd .
?s wdt:P31/wdt:P279* wd:%s .
?o wdt:P31/wdt:P279* wd:%s .
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
} GROUP BY ?p ?pLabel ORDER BY DESC(?count)
LIMIT 10
"""

ALL_PROPERTIES = """
SELECT ?property ?propertyType ?propertyLabel ?propertyDescription ?propertyAltLabel 
WHERE {
  ?property wikibase:propertyType ?propertyType .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
ORDER BY ASC(xsd:integer(STRAFTER(STR(?property), 'P')))
"""

CHECK_PROPERTY = """
SELECT ?item ?itemLabel
WHERE {
    wd:%s wdt:%s* wd:?item.
    SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
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

if __name__ == '__main__':
    endpoint_url = "http://query.wikidata.org/sparql"
    linker = EntityLinker()
    ap = ArgumentParser()
    ap.add_argument('input_file', help='PersonaChat file as input')
    # input KG
    ap.add_argument('-k', '--input_kg', help='Existing KG to be extended')
    args = ap.parse_args()
    persona_file = args.input_file
    kg = args.input_kg

    traits = []
    with open(persona_file, 'r') as p_file:
        for line in p_file:
            if 'persona' in line:
                try:
                    line = line.split(':')[1]
                    traits.append(line)
                except:
                    pass

    print('Found {} persona traits.'.format(len(traits)))
    sparql_client = SPARQLWrapper(endpoint_url, returnFormat=JSON, agent='User-Agent: Mozilla/5.0')
    sparql_client.setTimeout(604800)
    start_point = 0
    output_graph = 'pickled_graph'
    if kg:
        G = nx.read_gpickle(kg)
        start_point = int(input('Starting from... '))
    else:
        user_resp = input('Are you sure you want to write the graph on {}? ( y | n )'.format(output_graph))
        if user_resp == 'n':
            print('Exiting...')
            sys.exit()

        G = nx.DiGraph()
    count = 0
    ent_num = 0
    parent_entities = []
    ent_ = 0
    for utterance in tqdm(traits):
        count += 1
        if count < start_point:
            continue
        entities = linker._get_entity_mentions(utterance)
        ent_num += len(entities)
        for ent, val in entities.items():
            main_ent = find_entities(val)
            G.add_node(main_ent, name=ent)
            sparql_client.setQuery(SELECT_PARENT_QUERY % main_ent)
            # return: list of dict
            results = sparql_client.queryAndConvert()['results']['bindings']
            time.sleep(3)
            parent_entities.append(len(results))
            ent_ += len(results) + 1
            for res in results:
                parent_name = res['itemLabel']['value']
                parent_ent = res['item']['value'] 
                parent_ent = find_id(parent_ent) # entity ID
                G.add_node(parent_ent, name=parent_name) 
                G.add_edges_from([(main_ent, parent_ent, {'name': 'subclass_of'})])
                G.add_edges_from([(parent_ent, main_ent, {'name': 'has_subclass'})])
            nx.write_gpickle(G, output_graph)
        
    parent_entities = np.array(parent_entities)
    print('Nodes: {}\tEdges: {}'.format(len(list(G.nodes)), len(list(G.edges))))
    print('Total #entities: {} - average #parents per entity: {}'.format(ent_num, np.mean(parent_entities)))

    
    #nx.draw(G, with_labels=True, font_weight='bold')
    #plt.save('KG.pdf')

    #plt.show()

    # while (True):
    #     try:
    #         utterance_one = input("Sentence One:\n")
    #         utterance_two = input("Sentence Two:\n")
    #         entities_one = linker._get_entity_mentions(utterance_one)
    #         entities_two = linker._get_entity_mentions(utterance_two)
    #         sparql_client = SPARQLWrapper(endpoint_url, returnFormat=JSON)
    #         sparql_client.setTimeout(604800)
    #         sparql_client.setQuery(ALL_PROPERTIES)
    #         properties = sparql_client.queryAndConvert()
    #         properties = properties['results']['bindings']
    #         len_prop = len(properties)
    #         print('FOUND {} PROPERTIES'.format(len_prop))
    #         # keep only most probable entity
            

    #         # TODO: find main entities in sentences


    #         # checking sentence containing one entity only
    #         # TODO: check more entities in the sentence
    #         for entity_one, value_one in entities_one.items():
    #             for entity_two, value_two in entities_two.items():
    #                 while (True):
    #                     print(entity_one, entity_two)
    #                     ent_one = None
    #                     ent_two = None
    #                     if isinstance(value_one, str):
    #                         ent_one = find_id(value_one)
    #                     else:
    #                         ent_one = find_entities(value_one)
    #                     if isinstance(value_two, str):
    #                         ent_two = find_id(value_two)
    #                     else:
    #                         ent_two = find_entities(value_two)
    #                     print(ent_one, ent_two)
    #                     query = CHECK_CONNECTION_QUERY % (ent_one, ent_two)
    #                     #query = PROPERTY_QUERY % main_entity_one
    #                     sparql_client.setQuery(query)
    #                     results = sparql_client.queryAndConvert()
    #                     res_len = len(results['results']['bindings'])
    #                     if res_len > 0:
    #                         print(results['results']['bindings'])
    #                         break
    #                     else:
    #                         #print(ent_one)
    #                         # for property_ in properties:
    #                         #     property_ = find_id(property_['property']['value'])
    #                         #     print('Checking property {} for entity {}'.format(property_, ent_one))
    #                         #     query = CHECK_PROPERTY % (ent_one, property_)
    #                         #     sparql_client.setQuery(query)
    #                         #     results = sparql_client.queryAndConvert()
    #                         #     print(len(results['results']['bindings']))

    #                         query_rec = SELECT_PARENT_QUERY % ent_one
    #                         sparql_client.setQuery(query_rec)
    #                         results = sparql_client.queryAndConvert()
    #                         # # pick random result from various parents 
    #                         # len_res = len(results['results']['bindings'])
    #                         # ran_num = np.random.randint(0, len_res-1)
    #                         # entity_one = results['results']['bindings'][ran_num]['itemLabel']['value']
    #                         # value_one = results['results']['bindings'][ran_num]['item']['value']
    #                         # print('RESTARTING')
    #                         print(results)
    #                     #print(res_len)
                            
    #                     # wikidata for entity
                    
    #             # find all entities that link to the main_entity (returns an HTML page)
    #             # ress = requests.get("https://www.wikidata.org/w/index.php?title=Special:WhatLinksHere/%s" % main_entity)
    #             # print(ress.text)

    #             # if results.status_code != 200:
    #             #     print("Sad")
                
    #             # results = res.json()
    #             # for res_ in results['search']:
    #             #     print('{}\n'.format(res_))
    #             # print('\n\n')
    #     except KeyboardInterrupt:
    #         sys.exit()




# res = requests.get(
#                     "https://www.wikidata.org/w/index.php?title=Special:WhatLinksHere/",
#                     params={
#                         "action": "wbsearchentities",
#                         "search": entity, 
#                         "language": "en",
#                         "format": "json",
#                     }        
#                 )

#                 print(res.json())