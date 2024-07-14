from utils import *
from nlp_utils import *

parser = argparse.ArgumentParser(
	description='User-based Recommendation System developed based on National Library of Finland (NLF) dataset', 
	prog='RecSys Concatenated DFs', 
	epilog='Developed by Farid Alijani',
)
parser.add_argument(
	'-d', 
	'--dfsPath', # only available small df_concat: /scratch/project_2004072/Nationalbiblioteket/dataframes_tmp/
	type=str, 
	required=True,
	help='Path to directory of df_concat',
)
parser.add_argument('-q', '--qphrase', default="Suomen Kommunistinen Puolue", type=str)
parser.add_argument('-lm', '--lmMethod', default="stanza", type=str)
parser.add_argument('--normSP', default=False, type=bool)
parser.add_argument('--topTKs', default=5, type=int)
args = parser.parse_args()

fprefix: str = "FILE_PREFIXname_TBR"

# how to run:
# python concat_dfs.py --dfsPath /scratch/project_2004072/Nationalbiblioteket/dataframes_x2 --qphrase "feodalismi" --lmMethod "stanza"

# in Pouta:
# $ nohup python -u concat_dfs.py --dfsPath /media/volume/Nationalbiblioteket/dataframes_yy > concat_x19_check.out & 

def sum_tk_apperance_vb(dframe, qcol, wg, vb):
	updated_vb = dict.fromkeys(vb.keys(), 0.0)
	for tk in dframe[qcol]: # [tk1, tk2, …]
		if updated_vb.get(tk) is not None: # check if this token is available in BoWs
			updated_vb[tk] = updated_vb.get(tk) + wg
			#print(tk, wg, updated_vb[tk])
	#print(f"{dframe.user_ip}".center(50, '-'))
	return updated_vb

def sum_all_tokens_appearance_in_vb(dframe, weights: List[float], vb: Dict[str, int]):
	w_qu, w_hw_sn, w_sn, w_hw_cnt, w_pt_cnt, w_cnt = weights
	updated_vocab = dict.fromkeys(vb.keys(), 0.0)
	print(f"{dframe.user_ip}: "
				f"qu: {len(dframe.qu_tokens)}, "
				f"snHW: {len(dframe.snippets_hw_token)}, "
				f"sn: {len(dframe.snippets_token)}, "
				f"cntHW: {len(dframe.nwp_content_hw_token)}, "
				f"cntPT: {len(dframe.nwp_content_pt_token)}, "
				f"cnt: {len(dframe.nwp_content_lemma_all)}".center(130, " ")
			)
	for i, q_tk in enumerate(dframe.qu_tokens): # [qtk1, qtk2, qtk3, ...]
		#print(f"QU[{i}]: {q_tk:<25}w: {w_qu} | vb_exist? {updated_vocab.get(q_tk) is not None} ")
		if updated_vocab.get(q_tk) is not None:
			prev = updated_vocab.get(q_tk)
			curr = prev + w_qu
			print(f"QU[{i}]: {q_tk:<25}w: {w_qu}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[q_tk] = curr
	print('*'*60)
	for i, sn_hw_tk in enumerate(dframe.snippets_hw_token):
		#print(f"snHW[{i}]: {sn_hw_tk:<25}w: {w_hw_sn} | vb_exist? {updated_vocab.get(sn_hw_tk) is not None} ")
		if updated_vocab.get(sn_hw_tk) is not None:
			prev = updated_vocab.get(sn_hw_tk)
			curr = prev + w_hw_sn
			print(f"snHW[{i}]: {sn_hw_tk:<25}w: {w_hw_sn}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[sn_hw_tk] = curr
	print('*'*60)
	for i, sn_tk in enumerate(dframe.snippets_token):
		#print(f"sn[{i}]: {sn_tk:<25}w: {w_sn} | vb_exist? {updated_vocab.get(sn_tk) is not None} ")
		if updated_vocab.get(sn_tk) is not None:
			prev = updated_vocab.get(sn_tk)
			curr = prev + w_sn
			print(f"sn[{i}]: {sn_tk:<25}w: {w_sn}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[sn_tk] = curr
	print('*'*60)
	for i, c_hw_tk in enumerate(dframe.nwp_content_hw_token):
		#print(f"cntHW[{i}]: {c_hw_tk:<25}w: {w_hw_cnt} | vb_exist? {updated_vocab.get(c_hw_tk) is not None} ")
		if updated_vocab.get(c_hw_tk) is not None:
			prev = updated_vocab.get(c_hw_tk)
			curr = prev + w_hw_cnt
			print(f"cntHW[{i}]: {c_hw_tk:<25}w: {w_hw_cnt}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[c_hw_tk] = curr
	print('*'*60)
	for i, c_pt_tk in enumerate(dframe.nwp_content_pt_token):
		#print(f"cntPT[{i}]: {c_pt_tk:<25}w: {w_pt_cnt} | vb_exist? {updated_vocab.get(c_pt_tk) is not None} ")
		if updated_vocab.get(c_pt_tk) is not None:
			prev = updated_vocab.get(c_pt_tk)
			curr = prev + w_pt_cnt
			print(f"cntPT[{i}]: {c_pt_tk:<25}w: {w_pt_cnt}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[c_pt_tk] = curr
	print('*'*60)
	for i, c_tk in enumerate(dframe.nwp_content_lemma_all):
		#print(f"cnt[{i}]: {c_tk:<25}w: {w_cnt} | vb_exist? {updated_vocab.get(c_tk) is not None} ")
		if updated_vocab.get(c_tk) is not None:
			prev = updated_vocab.get(c_tk)
			curr = prev + w_cnt
			print(f"cnt[{i}]: {c_tk:<25}w: {w_cnt}\tprev: {prev:.3f}\tcurr: {curr:.3f}")
			updated_vocab[c_tk] = curr

	print("#"*150)
	return updated_vocab

def get_newspaper_content(lemmatized_content, vb:Dict[str, int], wg:float=weightContentAppearance):
	updated_vb = dict.fromkeys(vb.keys(), [0, 0])
	# lemmatized_content = [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]
	#print(f"Found {len(lemmatized_content)} content(s) {type(lemmatized_content)} [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]".center(130, "-"))
	for ilm, vlm in enumerate( lemmatized_content ): # [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]
		new_boosts = dict.fromkeys(vb.keys(), 0.0)
		#print(f"cnt[{ilm}] contain(s): {len(vlm)} token(s) {type(vlm)}")
		if vlm: # to ensure list is not empty!
			for iTK, vTK in enumerate( vlm ): # [tk1, tk2, ..., tkN]
				#print(f"tk@idx:{iTK} | {type(vTK)} | {len(vTK)}")
				if vb.get(vTK) is not None:
					new_boosts[vTK] = new_boosts[vTK] + wg				
			new_boosts = {k: v for k, v in new_boosts.items() if v} # get rid of those keys(tokens) with zero values to reduce size
			#print(f"\tcontent[{ilm}] |new_boosts| = {len(new_boosts)}" )
			for k, v in new_boosts.items():
				total_boost = v
				prev_best_boost, prev_best_doc = updated_vb[k]
				if total_boost > prev_best_boost:
					updated_vb[k] = [total_boost, ilm]
	return updated_vb

def get_selected_content(cos_sim, cos_sim_idx, recommended_tokens, df_users_tokens):
	print(f"Selected Content (using top-{len(recommended_tokens)} recommended tokens) INEFFICIENT".center(120, " "))
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, df_users_tokens, N=3)
	df = df_users_tokens[df_users_tokens["user_ip"].isin(nUsers_with_max_cosine)]
	print(df.shape)

	# lemmatized_content = [[tk1, tk2, tk3, ...], [tk1, tk2, ...], [tk1, ...], ...]
	for usr, lemmatized_content, content_text in zip(df["user_ip"], df["nwp_content_lemma_separated"], df["nwp_content_raw_text"]):
		user_selected_content_counter, user_selected_content, user_selected_content_idx = 0, None, None
		print(f"{usr} visited {len(lemmatized_content)} document(s) {type(lemmatized_content)}".center(150, "-"))
		for idx_cnt, tks_list in enumerate(lemmatized_content):
			print(f"<> document[{idx_cnt}] "
						f"contain(s) {len(tks_list)} TOKEN(s) {type(tks_list)}"
					)			
			for iTK, vTK in enumerate(recommended_tokens):
				print(f"recTK[{iTK}]: {vTK:<30}=> previous_counter: {user_selected_content_counter:<15}current_counter: {tks_list.count(vTK)}")
				if tks_list.count(vTK) > user_selected_content_counter:
					print(f"bingoo, found with token[{iTK}]: {vTK}: {tks_list.count(vTK)}".center(80, " "))
					user_selected_content_counter = tks_list.count(vTK)
					user_selected_content = content_text[0]
					user_selected_content_idx = idx_cnt

			#print("-"*100)
		print(f"\n{usr} selected content @idx: {user_selected_content_idx} | num_occurrence(s): {user_selected_content_counter}:\n")
		#print(f"\nSelected content:")
		print(user_selected_content)
		print("+"*180)

def get_cs_faiss(QU, RF, query_phrase: str, query_token, users_tokens_df:pd.DataFrame, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original" # unimportant!
	device = "GPU" if torch.cuda.is_available() else "CPU"
	QU = QU.reshape(1, -1).astype(np.float32) # QU: (nItems, ) => (1, nItems)
	RF = RF.astype(np.float32) # RF: (nUsers, nItems) 
	
	print(f"<Faiss> {device} Cosine Similarity: "
			 	f"QU: {QU.shape} {type(QU)} {QU.dtype}"
				f" vs. "
				f"RF: {RF.shape} {type(RF)} {RF.dtype}".center(160, " ")
			)
	"""
	RF = normalize(RF, norm="l2", axis=1)
	#QU = QU / np.linalg.norm(QU)
	QU = normalize(QU, norm="l2", axis=1)
	"""
	faiss.normalize_L2(RF)
	faiss.normalize_L2(QU)
	k=2048-1 if RF.shape[0]>2048 and device=="GPU" else RF.shape[0] # getting k nearest neighbors

	st_t = time.time()
	if torch.cuda.is_available():
		res = faiss.StandardGpuResources() # use a single GPU
		index = faiss.GpuIndexFlatIP(res, RF.shape[1]) 
	else:
		index = faiss.IndexFlatIP(RF.shape[1])
	index.add(RF)
	sorted_cosine, sorted_cosine_idx = index.search(QU, k=k)
	print(f"Elapsed_t: {time.time()-st_t:.3f} s | {sorted_cosine.shape}".center(100, " "))

	#print(sorted_cosine_idx.flatten()[:17])
	#print(sorted_cosine.flatten()[:17])
	
	# plot_cs(sorted_cosine, sorted_cosine_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp)
	return sorted_cosine, sorted_cosine_idx

def get_cs_sklearn(QU, RF, query_phrase: str, query_token, users_tokens_df:pd.DataFrame, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original"
	QU = QU.reshape(1, -1).astype(np.float32) # QU: (nItems, ) => (1, nItems)
	RF = RF.astype(np.float32) # RF: (nUsers, nItems) 
	print(f"Sklearn Cosine Similarity: "
				f"QU {type(QU)} {QU.shape} {QU.dtype}" # QU: (nItems, ) => (1, nItems)
				f" vs. "
				f"RF {type(RF)} {RF.shape} {RF.dtype}".center(160, " ") # RF: (nUsers, nItems)
			)
	st_t = time.time()
	cos_sim = cosine_similarity(QU, RF) # -> cos: (1, nUsers)
	print(f"Elapsed_t: {time.time()-st_t:.3f} s | {cos_sim.shape}".center(100, " "))

	sorted_cosine = np.flip(np.sort(cos_sim)) # descending
	sorted_cosine_idx = np.flip(cos_sim.argsort()) # descending

	# sorted_cosine = cos_sim
	# sorted_cosine_idx = cos_sim.flatten().argsort()
	# elp_t = time.time()

	# sorted_cosine_idx = cos_sim.flatten().argsort()
	# print(sorted_cosine_idx[-7:])
	# print(sorted_cosine.flatten()[:10])
	# print(sorted_cosine_idx[-1], users_tokens_df.index[sorted_cosine_idx[-1]], sorted_cosine.flatten()[sorted_cosine_idx[-1]])

	# DF = pd.DataFrame(sorted_cosine)
	# DF.to_csv("kantakirjasonni.csv")
	# plot_cs(sorted_cosine, sorted_cosine_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp)
	return sorted_cosine, sorted_cosine_idx

def plot_cs(cos_sim, cos_sim_idx, QU, RF, query_phrase, query_token, users_tokens_df, norm_sp=None):
	sp_type = "Normalized" if norm_sp else "Original"
	print(f"Plotting Cosine Similarity {cos_sim.shape} | Raw Query Phrase: {query_phrase} | Query Lemma(s) : {query_token}")	
	
	alphas = np.ones_like(cos_sim.flatten())
	scales = 100*np.ones_like(cos_sim.flatten())
	for i, v in np.ndenumerate(cos_sim.flatten()):
		if v==0:
			alphas[i] = 0.05
			scales[i] = 5

	f, ax = plt.subplots()
	ax.scatter(	x=cos_sim_idx.flatten(),
							y=cos_sim.flatten(), 
							facecolor="g", 
							s=scales, 
							edgecolors='w',
							alpha=alphas,
							marker=".",
						)
	
	N=3
	if np.count_nonzero(cos_sim.flatten()) < N:
		N = np.count_nonzero(cos_sim.flatten())
	# nUsers_with_max_cosine = list(users_tokens_df.index[cos_sim_idx.flatten()[:N]])
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df, N)

	
	ax.scatter(x=cos_sim_idx.flatten()[:N], y=cos_sim.flatten()[:N], facecolor='none', marker="o", edgecolors="r", s=100)
	#ax.set_xlabel('Users', fontsize=10)
	ax.set_ylabel('Cosine Similarity', fontsize=10.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	plt.xticks(	[i for i in range(len(users_tokens_df.index)) if i%MODULE==0], 
							[f"{users_tokens_df.index.tolist()[i]}" for i in range(len(users_tokens_df.index)) if i%MODULE==0],
							rotation=90,
							fontsize=10.0,
							)

	#ax.grid(linestyle="dashed", linewidth=1.5, alpha=0.5)
	ax.grid(which = "major", linewidth = 1)
	ax.grid(which = "minor", linewidth = 0.2)
	ax.minorticks_on()
	ax.set_axisbelow(True)
	ax.margins(1e-3, 3e-2)
	ax.spines[['top', 'right']].set_visible(False)

	plt.text(	x=0.5, 
						y=0.94, 
						s=f"Query Phrase: {query_phrase}", 
						fontsize=10.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.91,
						s=f"Query (1 x nItems): {QU.shape} {sp_type} Sparse Matrix (nUsers x nItems): {RF.shape} Cosine Similarity (1 x nUsers): {cos_sim.shape}",
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.88,
						s=f"{N}-Max cosine(s): {nUsers_with_max_cosine} : {cos_sim.flatten()[:N]}",
						fontsize=8.5,
						ha="center", 
						color="r",
						transform=f.transFigure,
					)
	
	plt.subplots_adjust(top=0.86, wspace=0.1)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_cosine_{QU.shape[1]}_nItems_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df:pd.DataFrame, N: int=3):
	if np.count_nonzero(cos_sim.flatten()) < N:
		N = np.count_nonzero(cos_sim.flatten())
	print(f"\n< {N} > user(s) with max CS:", end=" ")
	nUsers_with_max_cosine = list(users_tokens_df.index[cos_sim_idx.flatten()[:N]])	
	print(cos_sim_idx.flatten()[:N], nUsers_with_max_cosine, cos_sim.flatten()[:N], )
	return nUsers_with_max_cosine

def get_nwp_cnt_by_nUsers_with_max(cos_sim, cos_sim_idx, sp_mtrx, users_tokens_df, bow, recommended_tokens, norm_sp: bool=False):
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df, N=3)
	print(f"top-{len(recommended_tokens)} recommeded token(s): {recommended_tokens}")
	for recTK_idx, recTK in enumerate(recommended_tokens):
		print(f">> recTK[{recTK_idx}]: {recTK}")
		max_boost = 0.0
		max_boost_idoc = 0
		winner_user = None
		winner_content = None
		for iUSR, vUSR in enumerate(nUsers_with_max_cosine):
			print(vUSR, end="\t")
			tboost, idoc = users_tokens_df[users_tokens_df["user_ip"]==vUSR]["selected_content"].values.tolist()[0].get(recTK)
			print(f"[total_boost, idoc]: [{tboost}, {idoc}]")
			if tboost > max_boost:
				max_boost = tboost
				max_boost_idoc = idoc
				winner_user = vUSR
		if winner_user:
			user_all_doc = users_tokens_df[users_tokens_df["user_ip"]==winner_user]["nwp_content_raw_text"].values.tolist()[0]
			print(f"winner: {winner_user}, idoc: {max_boost_idoc} / ALL:|{len(user_all_doc)}|, {max_boost}".center(120, " "))
			user_best_doc = user_all_doc[max_boost_idoc]
			print(f"\ncontent {type(user_best_doc)} with {len(user_best_doc)} character(s):\n")
			print(user_best_doc)
		print("-"*120)

	#return

def plot_tokens_by_max(cos_sim, cos_sim_idx, sp_mtrx, users_tokens_df, bow, topTKs: int=25, norm_sp: bool=False, N: int=3):
	sp_type = "Normalized" if norm_sp else "Original"
	nUsers_with_max_cosine = get_nUsers_with_max(cos_sim, cos_sim_idx, users_tokens_df, N)
	nTokens = len(users_tokens_df.columns)
	for _, usr in enumerate(nUsers_with_max_cosine):
		f, ax = plt.subplots()
		utDF_s = users_tokens_df.loc[usr, :].sort_values(ascending=False)
		utDF_s_positives = utDF_s[utDF_s > 0]
		topTKs = len(utDF_s_positives) if len(utDF_s_positives) < topTKs else topTKs
		print(f"Plotting top-{topTKs} token(s) / |TKs[ > 0.0] = {len(utDF_s_positives)}| {usr}".center(110, "-"))
		ax.barh(list(users_tokens_df.loc[usr, :].sort_values(ascending=False).index[:topTKs]), 
						users_tokens_df.loc[usr, :].sort_values(ascending=False).values.tolist()[:topTKs],
						color="#0000ff",
						height=0.4,
					)

		ax.tick_params(axis='x', labelrotation=0, labelsize=7.0)
		ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
		ax.set_xlabel(f'{usr} UserInterest {sp_type} Sparse Matrix', fontsize=10.0)
		ax.invert_yaxis()  # labels read top-to-botto
		ax.set_title(f'Top-{topTKs} Tokens / |TKs[ > 0.0] = {len(utDF_s_positives)}| {usr}', fontsize=10)
		ax.margins(1e-2, 5e-3)
		ax.spines[['top', 'right']].set_visible(False)
		for container in ax.containers:
			ax.bar_label(	container, 
										#rotation=45, # no rotation for barh 
										fontsize=6.0,
										padding=1.5,
										# fmt='%.3f', #if norm_sp else '%.2f',
										label_type='edge',
									)
		ax.set_xlim(right=ax.get_xlim()[1]+5.0, auto=True)
		plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_{usr}_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
		plt.clf()
		plt.close(f)
	print(f"DONE".center(100, "-"))

def plot_users_by(token, usrs_name, usrs_value_all, usrs_value_separated, topUSRs, bow, norm_sp: bool=False):
	sp_type = "Normalized" if norm_sp else "Original"
	
	nUsers_orig = len(usrs_name)

	if len(usrs_name) > topUSRs:
		usrs_name = usrs_name[:topUSRs]
		usrs_value_all = usrs_value_all[:topUSRs]
		usrs_value_separated = [elem[:topUSRs] for elem in usrs_value_separated]

	nUsers = len(usrs_name)

	print(f"Plotting top-{nUsers} users out of |ALL_USRs = {nUsers_orig} | token: {token}".center(110, " "))

	f, ax = plt.subplots()
	ax.bar(usrs_name, usrs_value_all, color="#a6111122", width=0.58)
	#ax.set_xlabel('Tokens', rotation=90)
	ax.tick_params(axis='x', labelrotation=90, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)
	ax.set_ylabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)
	ax.set_title(f'Top-{nUsers} User(s) / |ALL_USRs = {nUsers_orig}| for token: {token}', fontsize=11)
	ax.margins(1e-2, 3e-2)
	ax.spines[['top', 'right']].set_visible(False)
	for container in ax.containers:
		ax.bar_label(	container, 
									rotation=0, 
									fontsize=6.0,
									padding=1.5,
									fmt='%.3f',# if norm_sp else '%.2f', 
									label_type='edge',
									)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_topUSRs{nUsers}_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

	#print(f"Separated Values".center(100, " "))
	f, ax = plt.subplots()
	btn = np.zeros(len(usrs_name))
	qcol_list = ["Search PHRs", "Snippet HWs", "Snippet Appr", "Content HWs", "Content PRTs", "Content Appr",]
	vbars = list()
	for i, v in enumerate( usrs_value_separated ):
		#print(i, usrs_name, v)
		vbar = ax.bar(usrs_name, v, color=clrs[i], width=0.6, bottom=btn, edgecolor='w', lw=0.4, label=f"{qcol_list[i]:<20}w: {w_list[i]:<{10}.{3}f}{[f'{val:.3f}' for val in v]}")
		btn += v
		vbars.append(vbar)
	
	ax.tick_params(axis='x', labelrotation=90, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)
	ax.set_ylabel(f'Cell Value in {sp_type} Sparse Matrix', fontsize=10.0)	
	ax.set_title(f'Top-{nUsers} User(s) / |ALL_USRs = {nUsers_orig}| for token: {token}', fontsize=11)
	ax.margins(1e-2, 5e-3)
	ax.spines[['top', 'right']].set_visible(False)
	ax.legend(loc="upper right", fontsize=(110.0/topUSRs) )

	for b in vbars:
		filtered_lbls = [f"{v:.1f}" if v>=7.0 else "" for v in b.datavalues]
		ax.bar_label(	b, 
									labels=filtered_lbls, 
									label_type='center', 
									rotation=0.0, 
									fontsize=6.0,
								)
	ax.set_ylim(top=ax.get_ylim()[1]+1.0, auto=True)
	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_topUSRs{nUsers}_separated_{len(bow)}BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_usersInterest_by(token, sp_mtrx, users_tokens_df, bow, norm_sp: bool=False):
	matrix = sp_mtrx.toarray()
	sp_type = "Normalized" if matrix.max() == 1.0 else "Original" 
	tkIdx = bow.get(token)
	usersInt = matrix[:, tkIdx]
	alphas = np.ones_like(usersInt)
	scales = 100*np.ones_like(usersInt)
	for i, v in np.ndenumerate(usersInt):
		if v==0:
			alphas[i] = 0.05
			scales[i] = 5

	f, ax = plt.subplots()
	ax.scatter(	x=np.arange(len(usersInt)), 
							y=usersInt, 
							facecolor="b", 
							s=scales,
							edgecolors='w',
							alpha=alphas,
							marker=".",
						)
	
	N=3
	if np.count_nonzero(usersInt) < N:
		N = np.count_nonzero(usersInt)

	topN_max_usrInt_user_idx = np.argsort(usersInt)[-N:]
	topN_max_user_interest = usersInt[topN_max_usrInt_user_idx]
	topN_max_usrInt_user_ip = users_tokens_df.loc[topN_max_usrInt_user_idx, 'user_ip'].values.tolist()
	ax.scatter(x=topN_max_usrInt_user_idx, y=topN_max_user_interest, facecolor='none', marker="o", edgecolors="r", s=100)

	#ax.set_xlabel('Users', fontsize=10)
	ax.set_ylabel('UserInterest [Implicit Feedback]', fontsize=10)
	"""
	ax.set_title(	f"{sp_type} Sparse Matrix (nUsers x nItems): {matrix.shape}\n"
								f"Users Interests by (token: {token} idx: {tkIdx})\n"
								f"max(UserInterest): {usersInt.max():.3f}@(userIdx: {np.argmax(usersInt)} userIP: {users_tokens_df.loc[np.argmax(usersInt), 'user_ip']})", 
								fontsize=10,
							)
	"""
	ax.tick_params(axis='y', labelrotation=0, labelsize=7.0)
	plt.xticks(	[i for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0], 
							[f"{users_tokens_df.loc[i, 'user_ip']}" for i in range(len(users_tokens_df["user_ip"])) if i%MODULE==0],
							rotation=90,
							fontsize=10.0,
							)

	#ax.grid(linestyle="dashed", linewidth=1.5, alpha=0.5)
	ax.grid(which = "major", linewidth = 1)
	ax.grid(which = "minor", linewidth = 0.2)
	ax.minorticks_on()
	ax.set_axisbelow(True)
	ax.margins(1e-3, 3e-2)
	ax.spines[['top', 'right']].set_visible(False)

	plt.text(	x=0.5, 
						y=0.94, 
						s=f"Raw Input Query Phrase: {args.qphrase}", 
						fontsize=10.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.text(	x=0.5,
						y=0.91,
						s=f"{sp_type} Sparse Matrix (nUsers x nItems): {matrix.shape} | Users Interests by (token: {token} @ idx: {tkIdx})",
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)

	plt.text(	x=0.5,
						y=0.88,
						s=f"{N}-Max UsersInterest(s): {topN_max_usrInt_user_ip} : {topN_max_user_interest}",
						fontsize=9.0, 
						ha="center", 
						color="r",
						transform=f.transFigure,
					)

	plt.subplots_adjust(top=0.86, wspace=0.1)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_tk_{token}_usersInterest_{len(bow)}BoWs_{sp_type}_SP.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)

def plot_heatmap_sparse(sp_mtrx, df_usr_tk, bow, norm_sp:bool=False, ifb_log10: bool=False):
	name_="sparse_matrix_user_vs_token"
	sp_type = "Normalized" if norm_sp else "Original"
	print(f"{f'{sp_type} Sparse Matrix {sp_mtrx.shape}'.center(100,'-')}")
	print(f"<> Non-zeros vals: {sp_mtrx.data}")# Viewing stored data (not the zero items)
	print(f"<> |Non-zero cells|: {sp_mtrx.count_nonzero()}") # Counting nonzeros
	if df_usr_tk.index.inferred_type == 'string':
		df_usr_tk = df_usr_tk.reset_index()#.rename(columns = {'index':'user_ip'})

	mtrx = sp_mtrx.toarray() # to numpy array

	####################################################
	if ifb_log10:
		mtrx = np.log10(mtrx, out=np.zeros_like(mtrx), where=(mtrx!=0))
		#mtrx = normalize(mtrx, norm="l2", axis=0) # l2 normalize by column -> items
	####################################################

	max_pose = np.unravel_index(mtrx.argmax(), mtrx.shape)
	print(mtrx.max(), 
				max_pose,
				mtrx[max_pose],
				df_usr_tk['user_ip'].iloc[max_pose[0]],
				[k for k, v in bow.items() if v==max_pose[1]],

				#np.argmax(mtrx, axis=0).shape, 
				#np.argmax(mtrx, axis=0)[-10:], 
				#np.argmax(mtrx, axis=1).shape, 
				#np.argmax(mtrx, axis=1)[-10:],
			) # axis=0-> col, axis=1 -> row
	
	f, ax = plt.subplots()
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.05)
	im = ax.imshow(mtrx, 
								#cmap="viridis",#"magma", # https://matplotlib.org/stable/tutorials/colors/colormaps.html
								#cmap="gist_yarg", # 0: white , max -> black
								cmap="GnBu",
								#cmap="gist_gray", # 0: black , max -> white
								aspect="auto",
								)
	cbar = ax.figure.colorbar(im,
														ax=ax,
														orientation="vertical",
														cax=cax,
														#ticks=[0.0, 0.5, 1.0],
														)
	cbar.ax.tick_params(labelsize=7.5)
	cbar.set_label(	label="Implicit Feedback ( $Log_{{10}}$ )" if ifb_log10 else "Implicit Feedback",
									size=9.0, 
									#weight='bold',
									style='italic',
								)
	ax.set_xlabel(f"Token Indices", fontsize=10.0)
	ax.set_ylabel(f"{'user indeces'.capitalize()}\n"
								f"{df_usr_tk['user_ip'].iloc[-1]}$\longleftarrow${df_usr_tk['user_ip'].iloc[0]}", fontsize=10.0)

	#ax.set_yticks([])
	#ax.set_xticks([])
	#ax.xaxis.tick_top() # put ticks on the upper part
	ax.tick_params(axis='x', labelrotation=0, labelsize=8.0)
	ax.tick_params(axis='y', labelrotation=0, labelsize=8.0)

	plt.text(x=0.5, y=0.94, s=f"{sp_type} Sparse Matrix Heatmap (nUsers, nItems): {sp_mtrx.shape}", fontsize=10.0, ha="center", transform=f.transFigure)
	plt.text(	x=0.5, 
						y=0.88, 
						s=f"|non-zeros|: {sp_mtrx.count_nonzero()} / |tot_elem|: {sp_mtrx.shape[0]*sp_mtrx.shape[1]}\n"
							f"max: {mtrx.max():{2}.{1}f}@{max_pose}: (User: {df_usr_tk['user_ip'].iloc[max_pose[0]]}, Token: {[k for k, v in bow.items() if v==max_pose[1]]})", 
						fontsize=9.0, 
						ha="center", 
						transform=f.transFigure,
					)
	plt.subplots_adjust(top=0.8, wspace=0.3)

	plt.savefig(os.path.join( RES_DIR, f"heatmap_{sp_type}_log10_{ifb_log10}_SP_{len(bow)}_BoWs.png" ), bbox_inches='tight')
	plt.clf()
	plt.close(f)
	print(f"Done".center(70, "-"))

def plot_tokens_distribution(sparseMat, users_tokens_df, queryVec, recSysVec, bow, norm_sp: bool=False, topK: int=5):
	sp_type = "Normalized" if norm_sp else "Original"
	print(f"\nPlotting Tokens Distribution in {sp_type} Sparse Matrix (nUsers, nItems): {sparseMat.shape}")
	# sparse_df = pd.DataFrame(sparseMat.toarray(), index=users_tokens_df["user_ip"])
	sparse_df = users_tokens_df

	#sparse_df.columns = sparse_df.columns.map(str) # convert int col -> str col
	sparse_df = sparse_df.replace(0.0, np.nan) # 0.0 -> None: Matplotlib won't plot NaN values.
	#print(f">> queryVec: {queryVec.shape} | recSysVec: {recSysVec.shape}")

	if len(recSysVec.shape) > 1:
		recSysVec = recSysVec.flatten()	

	if len(queryVec.shape) > 1:
		queryVec = queryVec.flatten()
	
	qu_indices = np.nonzero(queryVec)[0]

	all_recommended_tks = [k for idx in recSysVec.flatten().argsort()[-50:] for k, v in bow.items() if (idx not in qu_indices and v==idx)]
	#print(f"ALL (15): {len(all_recommended_tks)} : {all_recommended_tks[-15:]}")
	
	topK_recommended_tokens = all_recommended_tks[-(topK+0):]
	#print(f"top-{topK} recommended Tokens: {len(topK_recommended_tokens)} : {topK_recommended_tokens}")
	topK_recommended_tks_weighted_user_interest = [ recSysVec.flatten()[bow.get(vTKs)] for iTKs, vTKs in enumerate(topK_recommended_tokens)]
	#print(f"top-{topK} recommended Tokens weighted user interests: {len(topK_recommended_tks_weighted_user_interest)} : {topK_recommended_tks_weighted_user_interest}")

	#recSysVec_indices = recSysVec.argsort()[-(topK+0):]
	recSysVec_indices = np.array([bow.get(vTKs) for iTKs, vTKs in enumerate(topK_recommended_tokens)])
	# print(topK_recommended_tokens)
	# print(recSysVec_indices)

	plt.rcParams["figure.subplot.right"] = 0.8
	f, ax = plt.subplots()	
	quTksLegends = []
	for ix, col in np.ndenumerate(qu_indices):
		# print(ix, col)
		sc1 = ax.scatter(	x=sparse_df.index, 
											y=sparse_df.iloc[:, col],
											label=f"{[k for k, v in bow.items() if v==col]} | {col}",
											marker="H",
											s=260,
											facecolor="none", 
											edgecolors=clrs[::-1][int(ix[0])],
										)
		quTksLegends.append(sc1)

	recLegends = []
	for ix, col in np.ndenumerate(np.flip(recSysVec_indices)):
		# print(ix, col)
		sc2 = ax.scatter(	x=sparse_df.index, 
											y=sparse_df.iloc[:, col],
											label=f"{[k for k, v in bow.items() if v==col]} | {col} | {recSysVec[col]:.3f}",
											marker=".",
											s=900*recSysVec[col],
											alpha=1/((2*int(ix[0]))+1),
											#cmap='magma',
											#c=clrs[int(ix[0])],
											edgecolors=clrs[int(ix[0])],
											facecolor="none",
										)
		recLegends.append(sc2)

	leg1 = plt.legend(handles=quTksLegends, loc=(1.03, 0.8), fontsize=9.0, title=f"Lemmatized Query Phrase(s)\nToken | vbIdx", fancybox=True, shadow=True,)
	plt.setp(leg1.get_title(), multialignment='center', fontsize=9.0)
	plt.gca().add_artist(leg1)
	leg2 = plt.legend(handles=recLegends, loc=(1.03, 0.0), fontsize=8.0, title=f"Top-{topK} Recommended Results\nToken | vbIdx | wightedUserInterest", fancybox=True, shadow=True,)
	plt.setp(leg2.get_title(), multialignment='center', fontsize=9.0)
	
	ax.spines[['top', 'right']].set_visible(False)
	ax.margins(1e-3, 5e-2)
	plt.xticks(	[i for i,_ in enumerate( users_tokens_df.index.values.tolist() ) if i%MODULE==0 ], 
							[f"{v}" for i, v in enumerate( users_tokens_df.index.values.tolist() ) if i%MODULE==0 ],
							rotation=90,
							fontsize=9.0,
							)

	plt.yticks(fontsize=10.0)
	plt.ylabel(f"Cell Value in {sp_type} Sparse Matrix", fontsize=10)
	#plt.xlabel(f"Users", fontsize=11)
	plt.title(f"Token(s) Distribution | {sp_type} Sparse Matrix (nUsers, nItems): {sparse_df.shape}\n"
						f"Raw Input Query Phrase: {args.qphrase}\n"
						f"Top-{topK}: {[k for idx in recSysVec_indices for k, v in bow.items() if v==idx][::-1]}", 
						fontsize=9.0,
					)

	plt.savefig(os.path.join( RES_DIR, f"qu_{args.qphrase.replace(' ', '_')}_{args.lmMethod}_top{topK}_recsys_{len(bow)}_BoWs_{sp_type}_SP.png" ), bbox_inches='tight')

	plt.clf()
	plt.close(f)
	print(">> Done!")

def get_users_tokens_df(save_dir: str="saving_directory", prefix_fname: str="prefix_file_name", lm: str=args.lmMethod):
	# inefficient approach! memory error for more than 2 log files...
	print(f"Pandas DataFrame".center(150, " "))
	user_df_files=get_df_files(fpath=save_dir+'/'+'*_user_df_*_BoWs.gz')
	for f in user_df_files:
		print(f)
	print("<>"*80)
	try:
		users_tokens_dfs=load_pickle(fpath=glob.glob(save_dir+'/'+'*_PDFs*_usr_tk_pdfs_list.gz')[0])
	except Exception as e:
		print(f"<!> Load pkl <pDFs list> {e}")
		users_tokens_dfs=list()
		load_time_start=time.time()
		for df_file_idx, df_file in enumerate(user_df_files):
			print(f"[PANDAS] Loading [{df_file_idx+1}/{len(user_df_files)}]: {df_file}")
			user_df=load_pickle(fpath=df_file)
			user_token_df=get_unpacked_user_token_interest(df=user_df)
			# users_tokens_dfs.append(user_token_df) # original PANDAS df
			users_tokens_dfs.append(get_df_spm(df=user_token_df)) # to SparseDtype			
		print(f"Loaded {len(users_tokens_dfs)} users_tokens_pdfs in {time.time()-load_time_start:.1f} s".center(180, "-"))
		usr_tk_pdfs_list_fname = os.path.join(save_dir, f"{prefix_fname}_lemmaMethod_{lm}_usr_tk_pdfs_list.gz")
		save_pickle(pkl=users_tokens_dfs, fname=usr_tk_pdfs_list_fname)
	print(f"[PANDAS] chain concatination of {len(users_tokens_dfs)} user_token_pdfs")
	st_t = time.time()
	# user_token_df_concat=get_concat(pdfs=users_tokens_dfs) # [TIME INEFFICIENT] for sparse pandas dataFrame
	user_token_df_concat=get_optimized_concat(pdfs=users_tokens_dfs)
	print(
		f"Elapsed_t: {time.time()-st_t:.2f} s | {type(user_token_df_concat)} {user_token_df_concat.shape}\n"
		f"Memory: {user_token_df_concat.memory_usage(index=True, deep=True).sum()/1e6:.2f} MB\n"
		f"Sparsity: {user_token_df_concat.sparse.density:.6f}"
	)
	print("<>"*50)
	print(user_token_df_concat.info(memory_usage="deep"))
	print("<>"*50)

	user_token_pdf_fname = os.path.join(save_dir,
																			f"{prefix_fname}_"
																			f"lemmaMethod_{args.lmMethod}_USERs_TOKENs_pdf_"
																			f"{user_token_df_concat.shape[0]}_nUSRs_x_"
																			f"{user_token_df_concat.shape[1]}_nTOKs.gz"
																		)
	save_pickle(pkl=user_token_df_concat, fname=user_token_pdf_fname)
	save_vocab(	vb={ c: i for i, c in enumerate(user_token_df_concat.columns) },
							fname=os.path.join(save_dir, 
																f"{prefix_fname}_lemmaMethod_{args.lmMethod}_{len(user_token_df_concat.columns)}_concatVBs.json",
																),
					)
	return user_token_df_concat 

def get_users_tokens_ddf():
	print(f"Dask DataFrame".center(150, " "))

	# BoWs_files = natsorted( glob.glob( args.dfsPath + "/nike" + "*.json" ) )
	# print(f"<> Loading and Merging {len(BoWs_files)} BoWs:")
	# nTOTAL_BoWs: int = 0
	# for file_ in BoWs_files:
	# 	print(file_)
	# 	m=re.search(r'_(\d+)_vocabs\.json', file_)
	# 	nTOTAL_BoWs += int( m.group(1) ) if m else 0
	# st_t = time.time()
	# BoWs_merged = {k: i for i, k in enumerate(sorted(set().union(*(set(d) for d in [load_vocab(fname=fn) for fn in BoWs_files ] ) ) ) ) }
	# print(f"Loaded all {len(BoWs_files)} BoWs in {time.time()-st_t:.2f} s |nUNQ_vocab(s)|: {len(BoWs_merged)} |nTOTAL_vocab(s)|: {nTOTAL_BoWs}".center(150, " "))
	# # print("-"*80)
	# # gc.collect()

	user_df_files = natsorted( glob.glob( args.dfsPath+'/'+'*_user_df_*_BoWs.gz' ) )
	print(f"Found {len(user_df_files)} user_df [{type(user_df_files)}] files:")
	for f in user_df_files:
		print(f)
	print("<>"*80)
	# gc.collect() # TODO: check the impact on memory!

	try:
		users_tokens_dfs = load_pickle(fpath=glob.glob( args.dfsPath+'/'+'*_usr_tk_pdfs_list.gz' )[0]  )
	except Exception as e:
		print(f"<!> Load pkl <pDFs list> {e}")
		users_tokens_dfs = list()
		load_time_start = time.time()
		for df_file_idx, df_file in enumerate(user_df_files):
			print(f"[PANDAS] Loading {df_file_idx+1}/{len(user_df_files)}: {df_file}")
			user_df = load_pickle(fpath=df_file)
			print(f"[PANDAS] Unpacking nested dict of tokens & reindex cols (A, B, C, ..., Ö)", end="\t")
			st_t = time.time()
			# user_token_df = user_df.set_index("user_ip")["user_token_interest"].apply(pd.Series).astype("float32") # future warning
			# user_token_df = user_df.set_index("user_ip")["user_token_interest"].apply(lambda x: pd.Series(x, dtype="object")).astype("float32") # future warning
			user_token_df = pd.json_normalize(user_df["user_token_interest"]).set_index(user_df["user_ip"]).astype("float32")
			user_token_df = user_token_df.reindex(columns=sorted(user_token_df.columns), index=user_df["user_ip"])
			print(f"Elapsed_t: {time.time()-st_t:.2f} s  | {type(user_token_df)} | {user_token_df.shape}")
			users_tokens_dfs.append(user_token_df)
		print(f"Loaded {len(users_tokens_dfs)} users_tokens_dfs in {time.time()-load_time_start:.1f} sec".center(180, "-"))
		# gc.collect()
		usr_tk_pdfs_list_fname = os.path.join(args.dfsPath, f"{fprefix}_lemmaMethod_{args.lmMethod}_usr_tk_pdfs_list.gz")
		save_pickle(pkl=users_tokens_dfs, fname=usr_tk_pdfs_list_fname)

	print(f"[DASK] Concatinating {len(users_tokens_dfs)} users_tokens_dfs & GroupBy user_ip column", end="\t")
	st_t = time.time()
	user_token_ddf_concat = dd.concat(users_tokens_dfs, axis=0).groupby("user_ip").sum().reset_index()
	user_token_ddf_concat = user_token_ddf_concat.assign(n=user_token_ddf_concat['user_ip'].str[2:].astype(int)).set_index('user_ip').sort_values(['n']).drop(columns=['n'])
	user_token_ddf_concat = user_token_ddf_concat[sorted(user_token_ddf_concat.columns)]
	print(f"Elapsed_t: {time.time()-st_t:.2f} s | {type(user_token_ddf_concat)} | {user_token_ddf_concat.shape}")	
	# gc.collect()

	# TODO: investigate with user_token_ddf_concat.index.size.compute() for nUSRs
	user_token_ddf_concat_fname = os.path.join(args.dfsPath, 
																						f"{fprefix}_"
																						f"lemmaMethod_{args.lmMethod}_USERs_TOKENs_ddf_"
																						f"XXX_nUSRs_x_"
																						f"{len(user_token_ddf_concat.columns)}_nTOKs.parquet"
																					)
	print(f"Saving {type(user_token_ddf_concat)} might take a while...\n{user_token_ddf_concat_fname}")
	st_time = time.time()
	user_token_ddf_concat.to_parquet(path=user_token_ddf_concat_fname, engine='fastparquet')
	print(f"Elapsed_t: {time.time()-st_time:.2f} sec".center(110, " "))
	# gc.collect()
	return user_token_ddf_concat 

def run():
	print(f"Running {__file__} with {args.lmMethod.upper()} lemmatizer & {nb.get_num_threads()} CPU core(s)")
	make_folder(folder_name=args.dfsPath)

	sp_mtx_concat_BoWs = get_files(pth=args.dfsPath+'/'+'nike*_'+args.lmMethod+'_*_vocabs.json')
	sp_mtx_files = get_files(pth=args.dfsPath+'/'+'nike*_'+args.lmMethod+'*_SPM_UxT.gz')
	sp_mtx_rows_files = get_files(pth=args.dfsPath+'/'+'nike*_'+args.lmMethod+'*_SPM_rows.gz')
	sp_mtx_cols_files = get_files(pth=args.dfsPath+'/'+'nike*_'+args.lmMethod+'*_SPM_cols.gz')

	print(
		f"{len(sp_mtx_files)} spMtx files | " 
		f"{len(sp_mtx_rows_files)} spMtx rows(users) | "
		f"{len(sp_mtx_cols_files)} spMtx columns(tokens) | "
		f"{len(sp_mtx_concat_BoWs)} spMtx vocabs (json)"
		.center(190, "*")
	)
	# assert len(sp_mtx_files)==len(sp_mtx_rows_files)==len(sp_mtx_cols_files)==len(sp_mtx_concat_BoWs), f"<!> Error: 4 SPMs files (+1 BoWs) have different length!"
	assert len(sp_mtx_files)==len(sp_mtx_rows_files)==len(sp_mtx_cols_files), f"<!> Error: 3 SPMs files have different length!"
	if (
		len(sp_mtx_files)==0 
		or len(sp_mtx_rows_files)==0 
		or len(sp_mtx_cols_files)==0
	):
		print(f"<!> Error! no SPMs found!")
		return

	global fprefix, RES_DIR
	# fprefix=f"concatinated_{len(sp_mtx_files)}_SPMs_lm_{args.lmMethod}"
	fprefix=f"concatinated_732_SPMs_lm_{args.lmMethod}" # temporary!
	RES_DIR = make_result_dir(infile=fprefix)	
	try:
		concat_spm_U_x_T = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz')[0]
		)
		concat_spm_usrNames = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_spMtx_rows_*_nUSRs.gz')[0]
		)
		concat_spm_tokNames = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_spMtx_cols_*_nTOKs.gz')[0]
		)
	except Exception as e:
		print(f"<!> No SPM concat files found! {e} Generating in progress... [might take a while]")
		concat_spm_U_x_T, concat_spm_usrNames, concat_spm_tokNames = get_user_token_spm_concat(
			SPMs=[(load_pickle(fpath=spm_fpath), load_pickle(fpath=spm_usr_fpath), load_pickle(fpath=spm_tk_fpath)) for spm_fpath, spm_usr_fpath, spm_tk_fpath in zip(sp_mtx_files, sp_mtx_rows_files, sp_mtx_cols_files)],
			save_dir=args.dfsPath,
			prefix_fname=fprefix,
		)
	
	try:
		idf_vec = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_idf_vec_1_x_*_nTOKs.gz')[0]
		)
	except Exception as e:
		print(f"<!> idf file not available! {e}")
		idf_vec = get_idf(
			spMtx=concat_spm_U_x_T,
			save_dir=args.dfsPath,
			prefix_fname=fprefix,
		)

	try:
		usrNorms = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_users_norm_1_x_*_nUSRs.gz')[0]
		)
	except Exception as e:
		print(f"<!> usrNorm file not found! {e}")
		usrNorms = get_idfed_users_norm(
			spMtx=concat_spm_U_x_T,
			idf_vec=idf_vec,
			save_dir=args.dfsPath,
			prefix_fname=fprefix,
		) # (nUsers,) 

	################################################# SHRINKED #################################################
	try:
		# load shrinked spMtx
		concat_shrinked_spm_U_x_T = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_shrinked_spMtx_USERs_vs_TOKENs_*_nUSRs_x_*_nTOKs.gz')[0]
		)
		concat_shrinked_spm_usrNames = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_shrinked_spMtx_rows_*_nUSRs.gz')[0]
		)
		concat_shrinked_spm_tokNames=load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_shrinked_spMtx_cols_*_nTOKs.gz')[0]
		)
	except Exception as e:
		print(f"<!> No Shrinked_SPM concat files found {e} [might take a while]...")
		concat_shrinked_spm_U_x_T, concat_shrinked_spm_usrNames, concat_shrinked_spm_tokNames = get_shrinked_spMtx(
			spMtx=concat_spm_U_x_T,
			spMtx_rows=concat_spm_usrNames,
			spMtx_cols=concat_spm_tokNames,
			save_dir=args.dfsPath,
			prefix_fname=fprefix+f"_shrinked",
		)

	try:
		idf_vec_shrinked=load_pickle(fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_shrinked_idf_vec_1_x_*_nTOKs.gz')[0])
	except Exception as e:
		print(f"<!> SHRINKED idf file not available! {e}")
		idf_vec_shrinked=get_idf(
			spMtx=concat_shrinked_spm_U_x_T,
			save_dir=args.dfsPath,
			prefix_fname=fprefix+f"_shrinked",
		)

	try:
		usrNorms_shrinked = load_pickle(
			fpath=glob.glob( args.dfsPath+'/'+f'{fprefix}'+'_shrinked_users_norm_1_x_*_nUSRs.gz')[0]
		)
	except Exception as e:
		print(f"<!> SHRINKED usrNorm file not found! {e}")
		usrNorms_shrinked = get_idfed_users_norm(
			spMtx=concat_shrinked_spm_U_x_T,
			idf_vec=idf_vec_shrinked,
			save_dir=args.dfsPath,
			prefix_fname=fprefix+f"_shrinked",
		) # (nUsers,) 

	print(f"Concatenated < Original BIG > Sparse Matrix".center(120, "-"))
	print(
		f"spMtx {type(concat_spm_U_x_T)} {concat_spm_U_x_T.shape} {concat_spm_U_x_T.dtype}"
		f"byte size[memory footage]: {sum([sys.getsizeof(i) for i in concat_spm_U_x_T.data])/1e9:.3f} GB\n"
		f"spMtx_rows {type(concat_spm_usrNames)} {concat_spm_usrNames.shape} "	# <class 'numpy.ndarray'> (nUsers,)
		f"spMtx_cols {type(concat_spm_tokNames)} {concat_spm_tokNames.shape}\n"	# <class 'numpy.ndarray'> (nTokens,)
		f"IDF {type(idf_vec)} {idf_vec.shape} {idf_vec.dtype} {idf_vec.nbytes/1e6:.2f} MB\n"
		f"Customized Users Norm (IDFed): {type(usrNorms)} {usrNorms.dtype} {usrNorms.shape}"
	) # lil_matrix (nUsers, nTokens)
	print("-"*120)

	print(f"Concatenated < SHRINKED > Sparse Matrix".center(120, "-"))
	print(
		f"spMtx {type(concat_shrinked_spm_U_x_T)} {concat_shrinked_spm_U_x_T.shape} {concat_shrinked_spm_U_x_T.dtype}"
		f"byte size[memory footage]: {sum([sys.getsizeof(i) for i in concat_shrinked_spm_U_x_T.data])/1e9:.3f} GB\n"
		f"spMtx_rows {type(concat_shrinked_spm_usrNames)} {concat_shrinked_spm_usrNames.shape} "	# <class 'numpy.ndarray'> (nUsers,)
		f"spMtx_cols {type(concat_shrinked_spm_tokNames)} {concat_shrinked_spm_tokNames.shape}\n"	# <class 'numpy.ndarray'> (nTokens,)
		f"IDF {type(idf_vec_shrinked)} {idf_vec_shrinked.shape} {idf_vec_shrinked.dtype} {idf_vec_shrinked.nbytes/1e6:.2f} MB\n"
		f"Customized Users Norm (IDFed): {type(usrNorms_shrinked)} {usrNorms_shrinked.dtype} {usrNorms_shrinked.shape}"
	) # lil_matrix (nUsers, nTokens)
	print("-"*120)

	# save ALL concat files in dir as a tar archive file
	# comp_dir: str= "/scratch/project_2004072/Nationalbiblioteket/compressed_concatenated_SPMs"
	comp_dir = get_compressed_concatenated_path(base_path=args.dfsPath)
	concat_compressed_fname = f"concat_x{len(sp_mtx_files)}_lm_{args.lmMethod}.tar.gz"
	
	if not os.path.isfile(os.path.join(args.dfsPath, concat_compressed_fname)):
		print(f'>> fpath: {os.path.join(args.dfsPath, concat_compressed_fname)} does not exist, creating...')
		print(f">> comp_dir: {comp_dir}")
		get_compressed_archive(
			save_dir=args.dfsPath, 
			compressed_fname=concat_compressed_fname,
			upload_2_gdrive=False,
			compressed_dir=comp_dir,
		)
	#########################################################################################################################
	print(f"Input Query Phrase(s): < {args.qphrase} > ".center(150, " "))
	tokenized_query_phrase = get_lemmatized_sqp(qu_list=[args.qphrase], lm=args.lmMethod)
	print(f"Raw < {args.qphrase} > lemmatized into {len(tokenized_query_phrase)} lemma(s): {tokenized_query_phrase}")
	query_vector=get_query_vec(
		mat=concat_spm_U_x_T,
		mat_row=concat_spm_usrNames,
		mat_col=concat_spm_tokNames,
		tokenized_qu_phrases=tokenized_query_phrase,
	)
	print(
		f"quVec {type(query_vector)} {query_vector.dtype} {query_vector.shape}\n"
		f"Allzero? {np.all(query_vector==0.0)}\n"
		f"NonZeros: {np.count_nonzero(query_vector)}\n"
		f"@ idx(s): {np.where(query_vector.flatten()!=0)[0]}\n"
		f"{[f'idx[{qidx}] {concat_spm_tokNames[qidx]}' for _, qidx in enumerate(np.where(query_vector.flatten()!=0)[0])]}"
	)
	if np.all( query_vector==0.0 ):
		print(f"We couldn't find lemmas similar to {Fore.RED+Back.WHITE}{args.qphrase}{Style.RESET_ALL} in our BoWs! Search again!")
		return

	ccs=get_optimized_cs(
		spMtx=concat_spm_U_x_T,
		query_vec=query_vector,
		idf_vec=idf_vec,
		spMtx_norm=usrNorms, # must be adjusted, accordingly!
	)
	avgRecSys=get_avg_rec(
		spMtx=concat_spm_U_x_T,
		cosine_sim=ccs**5,
		idf_vec=idf_vec,
		spMtx_norm=usrNorms,
	)
	print("<>"*25)
	print(
		f"Recommendation Result:\n"
		f"Raw Query Phrase: < {args.qphrase} >\n"
		f"Lemmatized Query: {tokenized_query_phrase}\n"
	)
	topKtokens=get_topK_tokens(
		mat_cols=concat_spm_tokNames,
		avgrec=avgRecSys,
		raw_query=args.qphrase,
		tok_query=tokenized_query_phrase,
		meaningless_lemmas_list=UNQ_STW, # generated in tokenizer_utils.py
		K=200,
	)
	print(len(topKtokens), topKtokens)

	################################################# SHRINKED ################################################# 
	print(f"Shrinked Implementation [Token(s) seen by more than 1 User]".center(150, " "))
	query_vector_shrinked=get_query_vec(
		mat=concat_shrinked_spm_U_x_T,
		mat_row=concat_shrinked_spm_usrNames,
		mat_col=concat_shrinked_spm_tokNames,
		tokenized_qu_phrases=tokenized_query_phrase,
	)
	print(
		f"quVec [SHRINKED] {type(query_vector_shrinked)} {query_vector_shrinked.dtype} {query_vector_shrinked.shape}\n"
		f"Allzero? {np.all(query_vector_shrinked==0.0)}\n"
		f"NonZeros: {np.count_nonzero(query_vector_shrinked)}\n"
		f"@ idx(s): {np.where(query_vector_shrinked.flatten()!=0)[0]}\n"
		f"{[f'idx[{qidx}] {concat_shrinked_spm_tokNames[qidx]}' for _, qidx in enumerate(np.where(query_vector_shrinked.flatten()!=0)[0])]}"
	)
	if np.all( query_vector_shrinked==0.0 ):
		print(f"Sorry, We couldn't find tokenized words similar to {Fore.RED+Back.WHITE}{args.qphrase}{Style.RESET_ALL} in our BoWs! Search other phrases!")
		return

	ccs_shrinked=get_optimized_cs(
		spMtx=concat_shrinked_spm_U_x_T,
		query_vec=query_vector_shrinked,
		idf_vec=idf_vec_shrinked,
		spMtx_norm=usrNorms_shrinked, # must be adjusted, accordingly!
	)

	avgRecSys_shrinked=get_avg_rec(
		spMtx=concat_shrinked_spm_U_x_T,
		cosine_sim=ccs_shrinked**5,
		idf_vec=idf_vec_shrinked,
		spMtx_norm=usrNorms_shrinked,
	)
	print(
		f"Recommendation Result[SHRINKED]:\n"
		f"Raw Query Phrase: < {args.qphrase} >\n"
		f"Lemmatized Query: {tokenized_query_phrase}\n"
	)
	topKtokens_shrinked=get_topK_tokens(
		mat_cols=concat_shrinked_spm_tokNames,
		avgrec=avgRecSys_shrinked,
		raw_query=args.qphrase,
		tok_query=tokenized_query_phrase,
		meaningless_lemmas_list=UNQ_STW, # generated in tokenizer_utils.py
		K=200,
	)
	print(len(topKtokens_shrinked), topKtokens_shrinked)

	# print("<>"*100)
	# # Serialize the list into a string and print it
	# serialized_result = json.dumps(topKtokens)
	# print('Serialized Result:', serialized_result)

if __name__ == '__main__':
	# os.system("clear")
	print(
		f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
		.center(160, " ")
	)
	START_EXECUTION_TIME = time.time()
	run()
	END_EXECUTION_TIME = time.time()
	print(
		f"Finished: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
		f"TOTAL_ELAPSED_TIME: {END_EXECUTION_TIME-START_EXECUTION_TIME:.1f} sec"
		.center(160, " ")
	)