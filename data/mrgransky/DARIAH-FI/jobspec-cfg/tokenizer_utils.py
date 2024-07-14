from utils import *
# Define the global MultilingualPipeline object
lemmatizer_multi_lingual_pipeline = None

# with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
with HiddenPrints():
	import nltk

	# nltk_modules = [
	# 	'punkt',
	# 	'wordnet',
	# 	'averaged_perceptron_tagger', 
	# 	'omw-1.4',
	# 	'stopwords',
	# ]

	# nltk.download(
	# 	# 'all',
	# nltk_modules,
	# 'stopwords',
	# 	quiet=True,
	# 	# raise_on_error=True,
	# )

	import trankit
	from trankit import Pipeline

	import stanza
	from stanza.pipeline.multilingual import MultilingualPipeline
	from stanza.pipeline.core import DownloadMethod

	useless_upos_tags = [
		"PUNCT",
		"CCONJ",
		"SCONJ",
		"SYM",
		"AUX",
		"NUM",
		"DET",
		"ADP",
		"PRON",
		"PART",
		"INTJ",
		"ADV", 
		# "X", # foriegn words will be excluded,
	]
	STOPWORDS = nltk.corpus.stopwords.words(nltk.corpus.stopwords.fileids())
	with open('meaningless_lemmas.txt', 'r') as file_:
		my_custom_stopwords=[line.strip() for line in file_]
	STOPWORDS.extend(my_custom_stopwords)

	############################ meaningless_tokens_with_zero_NLF_page ############################
	try:
		# load lemmas_with_zero_nlf_results:
		vb = load_vocab(fname="/scratch/project_2004072/Nationalbiblioteket/dataframes_x732/meaningless_tokens_with_zero_NLF_page.gz")
		lemmas_with_zero_nlf_pages = list(vb.keys())#[:12941]
		STOPWORDS.extend(lemmas_with_zero_nlf_pages)
	except Exception as e:
		print(f"<!> {e}")
		pass
	############################ meaningless_tokens_with_zero_NLF_page ############################

	UNQ_STW = set(STOPWORDS)

# Function to create the MultilingualPipeline object if not already created
def create_stanza_multilingual_pipeline(device: str):
	global lemmatizer_multi_lingual_pipeline
	if lemmatizer_multi_lingual_pipeline is None:
		print(f"Initialize Stanza {stanza.__version__} Multilingual Pipeline using {device}".center(130, "-"))
		lang_id_config={
			"langid_lang_subset": [
				'fi', 
				'sv', 
				'en',
				'da',
				# 'nb', 
				'ru',
				# 'et', # causes wrong lemmas: 
				'de',
				'fr',
			]
		}

		# # without lemma store option: (must be slower)
		# lang_configs = {
		# 	"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True},
		# 	# "fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True}, # TDT
		# 	"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True}, # FTB
		# 	"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# 	# "sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True}, # errors!!!
		# 	"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# 	# "nb": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# 	"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True},
		# 	# "et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True},
		# 	"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True},
		# 	"fr": {"processors":"tokenize,lemma,pos,mwt", "package":'sequoia',"tokenize_no_ssplit":True},
		# }

		# with lemma store option: (must be faster)
		lang_configs = {
			"en": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True, "lemma_store_results":True},
			# "fi": {"processors":"tokenize,lemma,pos,mwt", "package":'tdt',"tokenize_no_ssplit":True, "lemma_store_results":True}, # TDT
			"fi": {"processors":"tokenize,lemma,pos,mwt", "package":'ftb',"tokenize_no_ssplit":True, "lemma_store_results":True}, # FTB
			"sv": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
			# "sv": {"processors":"tokenize,lemma,pos", "package":'lines',"tokenize_no_ssplit":True, "lemma_store_results":True}, # errors!!!
			"da": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
			# "nb": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
			"ru": {"processors":"tokenize,lemma,pos","tokenize_no_ssplit":True, "lemma_store_results":True},
			# "et": {"processors":"tokenize,lemma,pos", "package":'edt',"tokenize_no_ssplit":True, "lemma_store_results":True},
			"de": {"processors":"tokenize,lemma,pos", "package":'hdt',"tokenize_no_ssplit":True, "lemma_store_results":True},
			"fr": {"processors":"tokenize,lemma,pos,mwt", "package":'sequoia',"tokenize_no_ssplit":True, "lemma_store_results":True},
		}

		tt = time.time()
		# Create the MultilingualPipeline object
		lemmatizer_multi_lingual_pipeline = MultilingualPipeline( 
			lang_id_config=lang_id_config,
			lang_configs=lang_configs,
			download_method=DownloadMethod.REUSE_RESOURCES,
			device=device,
		)
		print(f"Elapsed_t: {time.time()-tt:.3f} sec".center(130, "-"))

def create_trankit_multilingual_pipeline(device: str):
	global lemmatizer_multi_lingual_pipeline
	if lemmatizer_multi_lingual_pipeline is None:
		print(f"Initialize Trankit {trankit.__version__} Multilingual Pipeline using {device}".center(130, "-"))
		tt = time.time()
		#lemmatizer_multi_lingual_pipeline = Pipeline('auto', embedding='xlm-roberta-large') # time-consuming and large models (unnecessary languages)
		lemmatizer_multi_lingual_pipeline = Pipeline(
			lang='finnish-ftb',
			gpu=True,
			embedding='xlm-roberta-large', 
			cache_dir=f"{os.environ['HOME']}/datasets/Nationalbiblioteket/trash" if os.environ['USER']!="alijanif" else '/scratch/project_2004072/trashes/',
		)
		lemmatizer_multi_lingual_pipeline.add('english')
		lemmatizer_multi_lingual_pipeline.add('swedish')
		lemmatizer_multi_lingual_pipeline.add('danish')
		lemmatizer_multi_lingual_pipeline.add('russian')
		lemmatizer_multi_lingual_pipeline.add('french')
		lemmatizer_multi_lingual_pipeline.add('german')
		lemmatizer_multi_lingual_pipeline.set_auto(True)
		print(f"Elasped_t: {time.time()-tt:.3f} sec".center(130, "-"))

@cache
def stanza_lemmatizer(docs: str="This is a <NORMAL> document!", device=None):
	# Ensure MultilingualPipeline object is created
	create_stanza_multilingual_pipeline(device=device)
	try:
		print(f'Stanza[{stanza.__version__} device: {device}] Raw Input:\n{docs}\n')
		# docs = docs.title()
		st_t = time.time()
		all_ = lemmatizer_multi_lingual_pipeline(docs)
		lemmas_list = [ 
			re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.sentences)
			for _, vw in enumerate(vsnt.words)
			if (
				(wlm:=vw.lemma)
				and 5 <= len(wlm) <= 43
				# and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<ros>|<eos>|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) # original stanza xx with no problems xx!
				# and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) # does not exclude words containing digits!
				and not re.search(r'\b(?=\d|\w)(?:\w*(?<!\b)(\w)(\1{2,})\w*|\d+\w*|\w*\d\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) # excludes words containing digits!
				and vw.upos not in useless_upos_tags 
				and wlm not in UNQ_STW
			)
		]
	except Exception as e:
		print(f"<!> Stanza Error: {e}")
		return
	##########################################################################################################################
	# # TODO: remove cols with zero results of NLF (Timeout Session): # NOT RECOMMENDED!!
	# print(f"raw {len(lemmas_list)} lemma(s):\n{lemmas_list}")
	# print(f"Asynchronous checking of {len(lemmas_list)} lemma(s) for possible ZERO NLF result pages [might take a while]", end="\t")
	# async_st_time = time.time()
	# lemmas_num_NLF_pages_async = asyncio.run(get_num_NLF_pages_asynchronous_run(TOKENs_list=lemmas_list))
	# lemmas_list_tmp = lemmas_list
	# lemmas_num_NLF_pages_async_tmp = lemmas_num_NLF_pages_async
	# lemmas_list = [word for num, word in zip(lemmas_num_NLF_pages_async_tmp, lemmas_list_tmp) if (num and num != 0) ]
	# print(f"Elapsed_t: {time.time()-async_st_time:.2f} sec")
	##########################################################################################################################
	end_t = time.time()
	print( lemmas_list )
	print(f"Found {len(lemmas_list)} lemma(s) Elapsed_t: {end_t-st_t:.3f} sec".center(160, "-") )
	return lemmas_list

@cache
def trankit_lemmatizer(docs: str="This is a <NORMAL> document!", device=None):
	# print(f'Raw: (len: {len(docs)}) >>{docs}<<')
	# print(f'Raw inp words: { len( docs.split() ) }', end=" ")
	create_trankit_multilingual_pipeline(device=device)
	try:
		print(f'Trankit[{trankit.__version__}] Raw Input:\n{docs}\n')
		# print(f"{f'nW: { len( docs.split() ) }':<10}{str(docs.split()[:7]):<150}", end="")
		docs = docs.title()
		st_t = time.time()
		all_ = lemmatizer_multi_lingual_pipeline( docs )
		# print(all_)
		# print(type(all_))
		# print(json.dumps(all_, indent=2, ensure_ascii=False))
		# print("#"*100)

		# for i, v in enumerate(all_.get("sentences")):
		# 	# print(i, v, type(v)) # shows <class 'dict'> entire dict of id, text, lemma, upos, ...
		# 	for ii, vv in enumerate(v.get("tokens")):
		# 		print(ii, vv.get("text"), vv.get("lemma"), vv.get("upos"))
		# 		# print(f"<>")
		# 	print('-'*50)

		lemmas_list = [
			re.sub(r'[";=&#<>_\-\+\^\.\$\[\]]', '', wlm.lower())
			for _, vsnt in enumerate(all_.get("sentences"))
			for _, vw in enumerate(vsnt.get("tokens"))
			if (
				(wlm:=vw.get("lemma"))
				and 5 <= len(wlm) <= 43
				# and not re.search(r'\b(?:\w*(\w)(\1{2,})\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) # does not exclude words containing digits! 
				and not re.search(r'\b(?=\d|\w)(?:\w*(?<!\b)(\w)(\1{2,})\w*|\d+\w*|\w*\d\w*)\b|<ros>|<eos>|/|<EOS>|<sos>|<SOS>|<UNK>|<unk>|\^|\s+', wlm) # excludes words containing digits! 
				and vw.get("upos") not in useless_upos_tags 
				and wlm not in UNQ_STW
			)
		]
	except Exception as e:
		print(f"<!> trankit Error: {e}")
		return
	end_t = time.time()
	print( lemmas_list )
	print(f"Found {len(lemmas_list)} lemma(s) Elapsed_t: {end_t-st_t:.3f} sec".center(160, "-") )
	# del all_
	# torch.cuda.empty_cache()
	# gc.collect()
	# cuda.select_device(torch.cuda.current_device()) # choosing second GPU 
	# cuda.close()
	return lemmas_list

def spacy_tokenizer(docs, device: str="cuda:0"):
	return None

def nltk_lemmatizer(sentence, device: str="cuda:0"):	
	#print(f'Raw inp ({len(sentence)}): >>{sentence}<<', end='\t')
	if not sentence:
		return
	wnl = nltk.stem.WordNetLemmatizer()

	sentences = sentence.lower()
	sentences = re.sub(r'"|<.*?>|[~|*|^][\d]+', '', sentences)
	sentences = re.sub(r'\b[A-Z](\.| |\:)+|\b[a-z](\.| |\:)+', '', sentences)
	sentences = re.sub(r'["]|[+]|[*]|”|“|\s+|\d', ' ', sentences).strip() # strip() removes leading (spaces at the beginning) & trailing (spaces at the end) characters
	#print(f'preprocessed: {len(sentences)} >>{sentences}<<', end='\t')

	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(sentences)

	filtered_tokens = [w for w in tokens if not w in UNQ_STW and len(w) > 1 and not w.isnumeric() ]
	# nltk.pos_tag: cheatsheet: pg2: https://computingeverywhere.soc.northwestern.edu/wp-content/uploads/2017/07/Text-Analysis-with-NLTK-Cheatsheet.pdf
	lematized_tokens = [wnl.lemmatize(w, t[0].lower()) if t[0].lower() in ['a', 's', 'r', 'n', 'v'] else wnl.lemmatize(w) for w, t in nltk.pos_tag(filtered_tokens)] 
	#print( list( set( lematized_tokens ) ) )

	return lematized_tokens