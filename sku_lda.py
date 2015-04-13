import logging,gensim,jieba
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)



dictionary = gensim.corpora.Dictionary([[]])
corpus=[]
for line in open('sku_name.txt'):
    line=line.strip().lower()
    if len(line) > 0:
        print line
	temp=list(jieba.cut(line))
	print temp
	dictionary.add_documents([temp])
        corpus.append(dictionary.doc2bow(temp))
dictionary.save_as_text('sku_name.dict') 

#doc vector
gensim.corpora.MmCorpus.serialize('sku_name.mm', corpus)

#compute term tfidf for every doc
tfidf = gensim.models.TfidfModel(corpus)
#print tfidf
corpus_tfidf = tfidf[corpus]
#for doc in corpus_tfidf:
#    print doc

lda = gensim.models.LdaModel(corpus_tfidf,id2word=dictionary,num_topics=10000)

lda.save("item_name_lda")

lda.print_topics(10)


