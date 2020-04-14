from typing import List

import bool_query as bq
from preprocess import InvertedIndex

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
    
    print(lemmatized_query)
    
    postfix_query = bq.query_to_postfix(lemmatized_query)

    print(postfix_query)

    relevant_documents_id = bq.process_postfix_query(postfix_query, inverted_index.index)

    return [inverted_index.mapping[doc_id] for doc_id in relevant_documents_id]
    
