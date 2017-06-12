from gensim.models import Doc2Vec

doc2vec_model_location = 'model/doc2vec-model.bin'

class NewsConverter:

    def __init__(self):
        self.model = Doc2Vec.load(doc2vec_model_location)

    def convert_doc_to_vector(self, doc):
        return self.model.infer_vector(doc.split())


