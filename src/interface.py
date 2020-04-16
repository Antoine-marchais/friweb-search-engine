from typing import List
import pickle as pkl

import bool_query as bq
import vectorial_query as vq
import argparse
from preprocess import InvertedIndex, StatCollection
from config import PATH_INDEX

def retrieve_docs_from_bool_query(query: str, inverted_index: InvertedIndex) -> List[str]:
    lemmatized_query = bq.lemmatize_query(query)

    # if no logical operator in the query, we assumes it's a "and"
    # "dog cat" -> "dog and cat"
    if not(any([x in lemmatized_query for x in bq.LOGICAL_TOKENS_VALUES])):
        new_query = []
        for term in lemmatized_query:
            new_query.append(term)
            new_query.append("and")
        new_query.pop()
        lemmatized_query = new_query
        
    postfix_query = bq.query_to_postfix(lemmatized_query)
    relevant_documents_id = bq.process_postfix_query(postfix_query, inverted_index.index)

    return [inverted_index.mapping[doc_id] for doc_id in relevant_documents_id]

def retrieve_docs_from_vectorial_query(query: str, inverted_index: InvertedIndex, n_results: int) -> List[str]:
    lemmatized_query = vq.lemmatize_query(query)
    
    ids_and_scores = vq.get_scores(lemmatized_query, inverted_index)
    best_ids_and_scores = sorted(list(ids_and_scores.items()), key=lambda id_and_score: id_and_score[1], reverse=True)[:n_results]

    return [inverted_index.mapping[id_and_score[0]] for id_and_score in best_ids_and_scores]

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="<boolean|vectorial> model used to process query", default="boolean")
    parser.add_argument("query", help="query to process")
    parser.add_argument("--number", "-n", help="number of results (only for boolean)", type=int, default=10)
    args = parser.parse_args()
    with open(PATH_INDEX, "rb") as f:
        inverted_index = pkl.load(f)
    if args.model == "boolean":
        print("\n".join(retrieve_docs_from_bool_query(args.query, inverted_index)))
    elif args.model == "vectorial":
        print("\n".join(retrieve_docs_from_vectorial_query(args.query, inverted_index, args.number)))


    
    
