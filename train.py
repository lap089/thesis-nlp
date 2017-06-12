from os import path
from random import shuffle

import nltk
from gensim.corpora import WikiCorpus
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.corpus import reuters
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer

import logging
import os.path
import six
import sys

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)

logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

google_news_word2vec_model_location = 'data/GoogleNews-vectors-negative300.bin.gz'
enwiki_bin_location = 'training/enwiki-20170420-pages-meta-current.xml.bz2'
enwiki_txt_location = 'training/wiki-documents.txt'
doc2vec_model_location = 'model/doc2vec-model.bin'
word2vec_model_location = 'model/word2vec-model.bin'
doc2vec_vectors_location = 'model/doc2vec-vectors.bin'
doc2vec_dimensions = 300
classifier_model_location = 'model/classifier-model.bin'


# Build the word2vec model from the corpus
# doc2vec.build_vocab(taggedDocuments)


def build_tagged_documents():
    taggedDocuments = []
    for idx, doc in enumerate(open(enwiki_txt_location, 'r', encoding="utf-8")):
        yield TaggedDocument(words=doc.split(), tags=[idx])
    #return taggedDocuments


def build_wiki_text():
    i = 0
    output = open(enwiki_txt_location, 'w+', encoding="utf-8")
    wiki = WikiCorpus(enwiki_bin_location, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        output.write(b' '.join(text).decode('utf-8') + '\n')
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")


if __name__ ==  '__main__':
    #build_wiki_text()

    logger.info("Build TaggedDocuments from training docs")
    it = build_tagged_documents()

    model = Doc2Vec(it, size=doc2vec_dimensions, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.0001,seed=5)  # use fixed learning rate

    #logger.info("Build Vocabulary from docs")
    #model.build_vocab(it)
    #logger.info("Start training")
    #model.train(it, total_examples=model.corpus_count, epochs=1)



    # Load the google news word2vec model
    #if (path.exists(google_news_word2vec_model_location)):
    #    model.load(google_news_word2vec_model_location, binary=True)


    #model = Doc2Vec.load(doc2vec_model_location)
    model.save(doc2vec_model_location)
    model.wv.save_word2vec_format(word2vec_model_location)
