nk(graph):
    '''
    A function that gets graph and loads information in it.
    '''
    #get data and put in entity2mention dictionary
    DB2en = {}
    
    entities = graph.subjects(predicate=RDF.type,object=entity_)
    
    for entity in entities:
        entity_id = entity.toPython()
        #print(entity_id)
        link_nodes = list(g.objects(subject=entity,
                                   predicate=URIRef(nist_ont_pref+'link')))
        if len(link_nodes)==0:
            continue
        link_node = link_nodes[0]
        link_string = list(g.objects(subject=link_node,
                       predicate=URIRef(nist_ont_pref+'linkTarget')))[0].toPython()
        
        if 'NIL' in link_string:
            continue
        #print link_string
        if link_string not in DB2en:
            DB2en[link_string.split(':')[-1]] = [entity]
        else:
            DB2en[link_string.split(':')[-1]].append(entity)
    return DB2en

import sys
#sys.path.append("/dvmm-filer2/projects/AIDA/alireza/tools/AIDA-Interchange-Format/python/aida_interchange")
sys.path.append("/home/brian/AIDA-Interchange-Format/python")
import time

start = time.time()
from rdflib import URIRef
from rdflib import URIRef
from rdflib import Graph, plugin, URIRef, Literal, BNode, RDF
import os, sys

import pickle
UIUC_path = argv[1]
UIUC_out = argv[2]
path_pref = UIUC_path+'/PT003_r1.pickle'
RPI_AIF_path = path_pref + ''
#nist_ont_pref = 'https://tac.nist.gov/tracks/SM-KBP/2018/ontologies/InterchangeOntology#'
nist_ont_pref = 'https://tac.nist.gov/tracks/SM-KBP/2019/ontologies/InterchangeOntology#'
entity_ = URIRef(nist_ont_pref+'Entity')

print('Creating id2link dictionary...')
id2link = {}
turtle_files = os.listdir(RPI_AIF_path)
for i,file in enumerate(turtle_files):
    if ".ttl" not in file:
        continue
    turtle_path = os.path.join(RPI_AIF_path, file)
    #loading turtle content
    turtle_content = open(turtle_path).read()
    g = Graph().parse(data=turtle_content, format='n3')
    id_ = file.split('.')[0]
    id2link[id_] = get_DB2link(g)
    sys.stdout.write('File {}/{} \r'.format(i+1,len(turtle_files)))                
    sys.stdout.flush()
    #break
#print id2link
with open(UIUC_out+'.pickle', 'wb') as f:
    pickle.dump(id2link,f,protocol=pickle.HIGHEST_PROTOCOL)
end = time.time()
print(end - start)

