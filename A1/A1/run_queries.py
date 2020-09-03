import argparse
import os

from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity

parser = argparse.ArgumentParser(description='Run all queries on the inverted index.')
parser.add_argument('--new', default=True, help='If True then build a new index from scratch. If False then attempt to'
                                                ' reuse existing index')
parser.add_argument('--sim', default='TFIDF', help='The type of similarity to use. Should be "TF" or "TFIDF"')
args = parser.parse_args()

index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=(not args.new))

sim_name_to_class = {'TF': TF_Similarity,
                     'TFIDF': TFIDF_Similarity}

sim = sim_name_to_class[args.sim]
index.set_similarity(sim)
print(f'Setting similarity to {sim.__name__}')

print()
print('Index ready.')


topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.runs')

# TODO run queries
"""
You will need to:
    1. Read in the topics_file.
    2. For each line in the topics file create a query string (note each line has both a query_id and query_text,
       you just want to search for the text)  and run this query on index with index.run_query().
    3. Write the results of the query to runs_file IN TREC_EVAL FORMAT
        - Trec eval format requires that each retrieval is on a separate line of the form
          query_id Q0 document_id rank similarity_score MY_IR_SYSTEM
"""
file = open(topics_file)
sorted_sim_scores = []
for line in file:
    # print(Preprocessor()(line))
    PreprocessorList = Preprocessor()(line)
    query_id = PreprocessorList[0]
    # print(query_id)
    del (PreprocessorList[0])
    # print(PreprocessorList)
    string = ' '.join(PreprocessorList)
    # print(string)
    sorted_sim_scores = index.run_query(string, 10)
    # print(sorted_sim_scores)
    i = 0
    for kv in sorted_sim_scores:
        # print(kv[0])
        with open(runs_file, 'a') as f:
            f.write('' + str(query_id) + ' Q0' + ' ' + str(kv[0]) + ' ' + str(i) + ' ' + str(kv[1]) + ' MY_IR_SYSTEM\n')
        i += 1
file.close()
