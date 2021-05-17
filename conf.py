import json
import os

ONTOLOGY_CONF_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources/ontology.json")

print("-- Reading ontology manager configuration file from: {}".format(ONTOLOGY_CONF_FILE))
with open(ONTOLOGY_CONF_FILE) as in_file:
    ontology_conf = json.load(in_file)