from argparse import ArgumentParser


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('entities_persona', help='Entities in persona files')
    ap.add_argument('entities_fb', help='Entities in FreeBase file')
    args = ap.parse_args()

    with open(args.entities_persona, 'r') as p_f:
        persona_entities = p_f.readlines()

    with open(args.entities_fb, 'r') as fb_f:
        freebase_entities = fb_f.readlines()

    freebase_dict = {}
    for line in freebase_entities:
        freebase_dict[line[0]] = line[1]

    not_found = 0
    for line in persona_entities:
        try:
            found = freebase_dict[line[0]]
        except:
            not_found += 1

    print('Not found {}/{} entities from Persona in FreeBase'.format(not_found, len(persona_entities)))