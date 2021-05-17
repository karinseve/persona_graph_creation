from neo4j import GraphDatabase
from pandas import DataFrame
import json


# FIND_NODES = """
#     MATCH (r1:properties)
#     WHERE (r1.skos__prefLabel = "['(85506) 1997 UU4']")
#     RETURN r1 LIMIT 5
# """WHERE r1.owl:sameAs = 'Q146'




FIND_NODES = """
    MATCH (r1:Resource)
    RETURN r1.owl:sameAs LIMIT 10
"""

if __name__ == '__main__':
    driver = GraphDatabase.driver("bolt://localhost:7687/", auth=('neo4j', 'admin'), encrypted=False)
    session = driver.session()
    # result = session.run("""MATCH (r1:Resource)-[l]->(r2:Resource)
    #     RETURN r1, l, r2 LIMIT 20""")
    result = session.run(FIND_NODES)
    print(result.data())
