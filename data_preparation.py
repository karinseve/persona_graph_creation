import csv


def read_entity_from_id(filename='./entity2id.txt'):
    entity2id = {}
    with open(filename, 'r') as f:
        for line in f:
            if len(line.strip().split()) > 1:
                entity, entity_id = line.strip().split(
                )[0].strip(), line.strip().split()[1].strip()
                entity2id[entity] = int(entity_id)
    return entity2id



def build_data_(path):
    entity2id = read_entity_from_id(path + 'entity2id.txt')
    return entity2id


if __name__ == '__main__':
    with open('./kg_triples.txt', 'r') as f:
        triples = f.readlines()
    entities = set()
    for line in triples:
        line = line.split('\t')
        line_0 = line[0].strip('\n')
        line_0 = line_0.strip(',')
        line_1 = line[2].strip('\n')
        line_1 = line_1.strip(',')
        entities.add(line_0)
        entities.add(line_1)

    with open('./entity2id.txt', 'w') as out_file:
        writer = csv.writer(out_file, delimiter='\t')
        count = 0
        for entity in entities:
            print(entity+'\t'+str(count))
            final_string = entity+'\t'+str(count)
            writer.writerow([entity, str(count)])
            count += 1