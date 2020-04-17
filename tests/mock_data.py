from preprocess import InvertedIndex, StatCollection
from collections import OrderedDict

COLLECTION = {
    "test1": "this is a dumb query, but test queries are always dumb",
    "test2": "why did I do information retrieval",
    "test3": "did you know that I will return this paper late ?",
    "test4": "Students nowadays cannot focus, I recently got sidetracked by a fly in my bedroom",
    "test5": "I am a student, I majored in computer science, and here I am writing test collections for information retrieval",
    "test6": "I wish this test will pass, this is why I do very dumb queries",
    "test7": "10 True facts about scientific papers and why we cannot trust the media"
}

INVERTED_INDEX_1 = InvertedIndex(
    itype=1,
    index=OrderedDict([
        ('dumb', [0, 5]), 
        ('query', [0, 5]), 
        ('test', [0, 4, 5]), 
        ('information', [1, 4]), 
        ('retrieval', [1, 4]), 
        ('return', [2]), 
        ('paper', [2, 6]), 
        ('late', [2]), 
        ('student', [3, 4]), 
        ('nowadays', [3]), 
        ('focus', [3]), 
        ('sidetrack', [3]), 
        ('fly', [3]), 
        ('bedroom', [3]), 
        ('major', [4]), 
        ('computer', [4]), 
        ('science', [4]), 
        ('write', [4]), 
        ('collection', [4]), 
        ('wish', [5]), 
        ('pas', [5]), 
        ('true', [6]), 
        ('scientific', [6]), 
        ('trust', [6]), 
        ('medium', [6])
    ]),
    mapping={0:"test1", 1:"test2", 2:"test3", 3:"test4", 4:"test5", 5:"test6", 6:"test7"},
    stats=StatCollection(
        nb_docs=7,
        doc_stats={
            0: OrderedDict([('freq_max', 2), ('moy_freq', 5/3), ('unique', 3)]), 
            1: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 2)]), 
            2: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 3)]), 
            3: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 6)]), 
            4: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 9)]), 
            5: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 5)]), 
            6: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 5)])
        }
    )
)

INVERTED_INDEX_2 = InvertedIndex(
    itype=2,
    index=OrderedDict([
        ('dumb', OrderedDict([(0, 2), (5, 1)])), 
        ('query', OrderedDict([(0, 2), (5, 1)])), 
        ('test', OrderedDict([(0, 1), (4, 1), (5, 1)])), 
        ('information', OrderedDict([(1, 1), (4, 1)])), 
        ('retrieval', OrderedDict([(1, 1), (4, 1)])), 
        ('return', OrderedDict([(2, 1)])), 
        ('paper', OrderedDict([(2, 1), (6, 1)])), 
        ('late', OrderedDict([(2, 1)])), 
        ('student', OrderedDict([(3, 1), (4, 1)])), 
        ('nowadays', OrderedDict([(3, 1)])), 
        ('focus', OrderedDict([(3, 1)])), 
        ('sidetrack', OrderedDict([(3, 1)])), 
        ('fly', OrderedDict([(3, 1)])), 
        ('bedroom', OrderedDict([(3, 1)])), 
        ('major', OrderedDict([(4, 1)])), 
        ('computer', OrderedDict([(4, 1)])), 
        ('science', OrderedDict([(4, 1)])), 
        ('write', OrderedDict([(4, 1)])), 
        ('collection', OrderedDict([(4, 1)])), 
        ('wish', OrderedDict([(5, 1)])), 
        ('pas', OrderedDict([(5, 1)])), 
        ('true', OrderedDict([(6, 1)])), 
        ('scientific', OrderedDict([(6, 1)])), 
        ('trust', OrderedDict([(6, 1)])), 
        ('medium', OrderedDict([(6, 1)]))
    ]),
    mapping={0:"test1", 1:"test2", 2:"test3", 3:"test4", 4:"test5", 5:"test6", 6:"test7"},
    stats=StatCollection(
        nb_docs=7,
        doc_stats={
            0: OrderedDict([('freq_max', 2), ('moy_freq', 5/3), ('unique', 3)]), 
            1: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 2)]), 
            2: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 3)]), 
            3: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 6)]), 
            4: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 9)]), 
            5: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 5)]), 
            6: OrderedDict([('freq_max', 1), ('moy_freq', 1.0), ('unique', 5)])
        }
    )
)

def get_index(index_type):
    if index_type == 1:
        return INVERTED_INDEX_1
    elif index_type == 2:
        return INVERTED_INDEX_2
    else :
        raise NotImplementedError