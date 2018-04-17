import sys, os
import pickle

def www2fb(in_str):
    out_str = 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return out_str

def main():
    in_fn = sys.argv[1]
    db = in_fn.split('-')[-1].split('.')[0]

    out_fn = '%s.core.txt' % (db)
    ent_fn = '%s.ent.pkl' % (db)
    rel_fn = '%s.rel.pkl' % (db)

    ent_dict = {}
    rel_dict = {}
    triple_dict = {}

    with open(in_fn) as fi:
        for line in fi:
            fields = line.strip().split('\t')
            sub = www2fb(fields[0])
            rel = www2fb(fields[1])
            objs = fields[2].split()
            if sub in ent_dict:
                ent_dict[sub] += 1
            else:
                ent_dict[sub] = 1
            if rel in rel_dict:
                rel_dict[rel] += 1
            else:
                rel_dict[rel] = 1
            for obj in objs:
                obj = www2fb(obj)
                triple_dict[(sub, rel, obj)] = 1
                if obj in ent_dict:
                    ent_dict[obj] += 1
                else:
                    ent_dict[obj] = 1

    pickle.dump(ent_dict, open(ent_fn, 'wb'))
    with open('%s.ent.txt' % (db), 'w') as fo:
        for k, v in sorted(ent_dict.items(), key = lambda kv: kv[1], reverse = True):
            fo.write(k + '\n')

    pickle.dump(rel_dict, open(rel_fn, 'wb'))
    with open('%s.rel.txt' % (db), 'w') as fo:
        for k, v in sorted(rel_dict.items(), key = lambda kv: kv[1], reverse = True):
            fo.write(k + '\n')

    with open(out_fn, 'w') as fo:
        for (sub, rel, obj) in triple_dict.keys():
            fo.write('<%s>\t<%s>\t<%s>\t.\n' % (sub, rel, obj))
    print(len(triple_dict))

if __name__ == '__main__':
    main()
