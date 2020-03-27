# coding: utf-8

# In[1]:


from gensim import corpora, models, similarities


# In[7]:


documents = ["Human machine interface for lab computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]


# In[8]:


print(documents)


# In[9]:


# remove common words and tokenize them
stoplist = set('for a of the and to in'.split())


# In[10]:


texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]


# In[11]:


print(texts)


# In[12]:


# remove words those appear only once
all_tokens = sum(texts, [])

print(all_tokens)


# In[13]:


tokens_once = set(word for word in set(all_tokens) if all_tokens.count(word) ==1)

print(tokens_once)


# In[14]:


texts = [[word for word in text if word not in tokens_once]
         for text in texts]

print(texts)


# In[15]:


dictionary = corpora.Dictionary(texts)

print(dictionary)


# In[16]:


dictionary.save('deerwester.dict')  # save as binary file at the dictionary at local directory


# In[17]:


dictionary.save_as_text('deerwester_text.dict')  # save as text file at the local directory


# In[18]:


print(dictionary.token2id) # show pairs of "word : word-ID number" 


# In[19]:


new_doc = "Human computer interaction" # temporary data to see role of below function

new_vec = dictionary.doc2bow(new_doc.lower().split()) # return "word-ID : Frequency of appearance""
print(new_vec)


# In[20]:


corpus = [dictionary.doc2bow(text) for text in texts]

print(corpus)


# In[21]:


corpora.MmCorpus.serialize('deerwester.mm', corpus) # save corpus at local directory


# In[22]:


corpus = corpora.MmCorpus('deerwester.mm') # try to load the saved corpus from local

print(list(corpus)) # to show corpus which was read above, need to print(list( )) 


# In[23]:


dictionary = corpora.Dictionary.load('deerwester.dict') # try to load saved dic.from local

print(dictionary)


# In[24]:


print(corpus)


# In[25]:


tfidf = models.TfidfModel(corpus) # step 1 -- initialize a model

print(tfidf)


# In[29]:


corpus_tfidf = tfidf[corpus]  # map corpus object into tfidf space

print(corpus_tfidf)


# In[30]:


for doc in corpus_tfidf: # show tfidf-space mapped words
    print(doc)


# In[31]:


lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2) # initialize LSI 
print(lsi)


# In[32]:


corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus
print(corpus_lsi)


# In[33]:


topic = lsi.print_topics(2)


# In[34]:


print(topic)


# In[35]:


for doc in corpus_lsi:
    print(doc)


# In[36]:


lsi.save('model.lsi')  # save output model at local directory


# In[37]:


lsi = models.LsiModel.load('model.lsi') # try to load above saved model
print(lsi)


# In[38]:


doc = "Human computer interaction"  # give new document to calculate similarity degree with already obtained topics

vec_bow = dictionary.doc2bow(doc.lower().split())  # put newly obtained document to existing dictionary object
print(vec_bow)  # show result of above 


# In[39]:


vec_lsi = lsi[vec_bow] # convert new document (henceforth, call it "query") to LSI space
print(vec_lsi)


# In[40]:


index = similarities.MatrixSimilarity(lsi[corpus]) # transform corpus to LSI space and indexize it
print(index)


# In[41]:


index.save('deerwester.index') # save index object at local directory


# In[42]:


index = similarities.MatrixSimilarity.load('deerwester.index')


# In[43]:


print(index)


# In[44]:


sims = index[vec_lsi] # calculate degree of similarity of the query to existing corpus
print(sims)


# In[45]:


print(list(enumerate(sims))) # output (document_number , document similarity)


# In[46]:


sims = sorted(enumerate(sims), key=lambda item: -item[1])  # sort output object as per similarity ( largest similarity document comes first )
print(sims)

