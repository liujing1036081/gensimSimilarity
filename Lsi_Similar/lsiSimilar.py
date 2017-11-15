from gensim import corpora,models,similarities
from gensim.corpora import Dictionary
import jieba
'''
LSI_model
从文本集中检索出与目标文本最相似的文本，并返回
'''
list_all=[]


for line in open('news_corpus.txt',encoding="utf-8"):
    # assume there's one document per line, tokens separated by whitespace
    list1=list(jieba.cut(line,cut_all=False))
    # print(list1)
    list_all.append(list1)

#构建字典
dictionary = corpora.Dictionary(list_all)
#检索源
query=dictionary.doc2bow(list_all[2])
#语料库词袋模型
corpus=[dictionary.doc2bow(text) for text in list_all]

#主题模型训练
lsi_model = models.LsiModel(corpus, id2word=dictionary, num_topics=3)
print(lsi_model.print_topics(3))
documents_vec = lsi_model[corpus]
query_vec = lsi_model[query]
index = similarities.MatrixSimilarity(documents_vec)
sims = index[query_vec]
print(sims)
