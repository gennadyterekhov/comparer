# coding: utf-8
from gensim import corpora, models, similarities

def do_all(document, documents):

    # remove common words and tokenize them
    stoplist = set('for a of the and to in'.split())

    texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]


    # remove words those appear only once
    all_tokens = sum(texts, [])


    tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) ==1)

    texts = [[word for word in text if word not in tokens_once]
            for text in texts]


    dictionary = corpora.Dictionary(texts)

    dictionary.save('files/deerwester.dict')  # save as binary file at the dictionary at local directory

    dictionary.save_as_text('files/deerwester_text.dict')  # save as text file at the local directory


    # print(dictionary.token2id) # show pairs of "word : word-ID number" 


    new_doc = "Human computer interaction" # temporary data to see role of below function

    new_vec = dictionary.doc2bow(new_doc.lower().split()) # return "word-ID : Frequency of appearance""
    # print(new_vec)

    corpus = [dictionary.doc2bow(text) for text in texts]

    # print(corpus)

    corpora.MmCorpus.serialize('files/deerwester.mm', corpus) # save corpus at local directory


    corpus = corpora.MmCorpus('files/deerwester.mm') # try to load the saved corpus from local



    dictionary = corpora.Dictionary.load('files/deerwester.dict') # try to load saved dic.from local

    # print(dictionary)





    # print(corpus)





    tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

    # print(tfidf)





    corpus_tfidf = tfidf[corpus]  # map corpus object into tfidf space

    # print(corpus_tfidf)





    for doc in corpus_tfidf: # show tfidf-space mapped words
        # print(doc)
        pass





    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize LSI 
    # print(lsi)





    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus
    # print(corpus_lsi)


    topic = lsi.print_topics(2)


    # print(topic)

    for doc in corpus_lsi:
        # print(doc)
        pass


    lsi.save('files/model.lsi')  # save output model at local directory


    lsi = models.LsiModel.load('files/model.lsi') # try to load above saved model


    #old
    doc = "Human computer interaction"  # give new document to calculate similarity degree with already obtained topics


    doc = document

    vec_bow = dictionary.doc2bow(doc.lower().split())  # put newly obtained document to existing dictionary object


    vec_lsi = lsi[vec_bow] # convert new document (henceforth, call it "query") to LSI space

    index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and indexize it



    index.save('files/deerwester.index') # save index object at local directory


    index = similarities.MatrixSimilarity.load('files/deerwester.index')


    sims = index[vec_lsi] # calculate degree of similarity of the query to existing corpus

    sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort output object as per similarity ( largest similarity document comes first )

    return sims

