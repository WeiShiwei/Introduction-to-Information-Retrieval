#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import numpy as np

import simple_inverted_index
texts = simple_inverted_index.texts # {”文档名称“：“文档内容”，}
words = sorted(simple_inverted_index.words) # 词项集合
invindex = simple_inverted_index.invindex # 倒排索引
# -----------------------------------------------------------------
term_vocabulary = ["螺纹钢","HRB400","石棉","橡胶板","地脚线","铆钉"]

# term_freq = dict()
# inverse_document_freq = dict()
# N = 0
# for term in term_vocabulary:
# 	tf = 0
# 	for document in invindex[term]:
# 		tf += texts[document].count(term)
# 	term_freq[term] = tf
# 	# N += tf

# 	# df = len(invindex[term])
# 	# document_freq[term] = df

# for (term,tf) in document_freq.items():
# 	inverse_document_freq[term] = math.log(N/float(tf))
# print "term_vocabulary:",term_vocabulary
# print "N:",N
# print "document_freq:",document_freq
# print "inverse_document_freq:",inverse_document_freq


# def term_document_weight(document,term_vocabulary=term_vocabulary):
# 	w_t_d = list()
# 	tf_vocabulary = list()
# 	for term in term_vocabulary:
# 		# import pdb;pdb.set_trace()
# 		tf_vocabulary.append( document.count(term) )

# 	module = float(sum([tf**2 for tf in tf_vocabulary]))
# 	w_t_d = [tf/module for tf in tf_vocabulary]
# 	# print w_t_d
# 	return w_t_d

# def init_term_document_matrix(texts):
# 	# for example: term_document_weight(texts['InvertedIndex/T0101.txt'])
# 	term_document_matrix = dict()
# 	document_names = texts.keys()
# 	for doc_name in document_names:
# 		term_document_matrix[doc_name] = term_document_weight( texts[doc_name] )
# 	print 'term_document_matrix:',term_document_matrix
# 	return term_document_matrix

# term_document_matrix = init_term_document_matrix(texts)



class CosineScore(object):
	"""docstring for CosineScore"""

	def __init__(self, arg):
		super(CosineScore, self).__init__()
		self.arg = arg


	@classmethod
	def __term_document_weight(slef, document,term_vocabulary=term_vocabulary):
		w_t_d = list()
		tf_vocabulary = list()
		for term in term_vocabulary:
			# import pdb;pdb.set_trace()
			tf_vocabulary.append( document.count(term) )

		module = float(sum([tf**2 for tf in tf_vocabulary]))
		w_t_d = [tf/module for tf in tf_vocabulary]
		# print w_t_d
		return w_t_d
	@classmethod
	def init_term_document_matrix(slef, texts):
		# for example: term_document_weight(texts['InvertedIndex/T0101.txt'])
		term_document_matrix = dict()
		document_names = texts.keys()
		for doc_name in document_names:
			term_document_matrix[doc_name] = CosineScore.__term_document_weight( texts[doc_name] )
		print 'term_document_matrix:',term_document_matrix
		return term_document_matrix
	# ----------------------------------------------------------------
	@classmethod
	def __calculate_idf(self, invindex, term_vocabulary):
		N = 0
		term_vocabulary_df = list()
		for term in term_vocabulary:
			# import pdb;pdb.set_trace()
			df = len(invindex[term])
			term_vocabulary_df.append(df)
			N += df

		term_vocabulary_idf = list()
		for df in term_vocabulary_df:
			term_vocabulary_idf.append( math.log(N/float(df)) )
		return term_vocabulary_idf
	@classmethod
	def __calculate_tf(self, query_terms, term_vocabulary):
		query_tf = [0]*len(term_vocabulary)
		for term in query_terms:
			# import pdb;pdb.set_trace()
			try:
				i = term_vocabulary.index(term)
				query_tf[i] += 1
			except Exception, e:
				continue
		return query_tf
	@classmethod
	def weight_term_query(self, query_terms):
		query_tf = CosineScore.__calculate_tf(query_terms, term_vocabulary)
		query_idf = CosineScore.__calculate_idf(invindex, term_vocabulary)
		return map((lambda a, b:a*b),query_tf,query_idf)
	# ----------------------------------------------------------------
	@classmethod
	def cosine_score_v1(self, query_terms):
		global texts
		term_document_matrix = CosineScore.init_term_document_matrix(texts)
		
		w_t_q = CosineScore.weight_term_query(query_terms)

		doc_product = (lambda a, b:a*b)
		def doc_product(left,right):
			assert len(left)==len(right)
			res = 0.0
			return reduce( (lambda a, b:a+b),map((lambda a, b:a*b),left,right) )
		
		cosine_score_list = list()
		for doc_name,w_t_d in term_document_matrix.items():
			cosine_score_list.append( doc_product(w_t_q,w_t_d) )

		# import pdb;pdb.set_trace()
		print 'cosine_score:',cosine_score_list
		
		print np.argsort(cosine_score_list) # argsort函数返回的是数组值从小到大的索引值
		Documents = texts.keys()
		for i in np.argsort(cosine_score_list):
			print Documents[i]

	@classmethod
	def cosine_score(self, query_terms):
		CosineScore.cosine_score_v1(query_terms)

	# @classmethod
	# def cosine_score_v2(self, query_terms):
	# 	import pdb;pdb.set_trace()
	# 	query_tf = CosineScore.calculate(query_terms)
	# 	query_idf = None
		
	# 	Scores = [0.0]*N
	# 	Length = [len(doc) for doc in [texts[name] for name in texts.keys()]]

	# 	for term in query_terms:
	# 		w_t_q = CosineScore.calculate_term(query_terms, term)
			
	# 		posting_list = fetch_posting_list(term)
	# 		for d in posting_list:
	# 			tf_t_d = texts[d].count(term)
	# 			Scores += wf_t_d*w_t_q

	# 	# Length = fetch_length()
	# 	for d in xrange(len(Scores)):
	# 		Scores[d] = Scores[d]/Length[d]

	# 	# return Top K components of Scores of Scores[]

def main():
	print "cosine_score"
	print "="*80
	
	# query_terms = ["螺纹钢","HRB400"]
	# query_terms = ["石棉","橡胶板"]
	query_terms = ["地脚线","铆钉"]
	print "Query Terms for:",query_terms

	CosineScore.cosine_score(query_terms)

if __name__ == "__main__":
	main()