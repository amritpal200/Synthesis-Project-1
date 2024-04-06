import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from tokenizers import CharBPETokenizer
from collections import Counter
import numpy as np
import random
import os
from typing import Union, Tuple, Iterable, Optional, List, Dict, Iterator, Literal
from IPython.display import clear_output
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import json
import re

def train_tokenizer(
		corpus: Union[str, Iterable[str]],
		vocab_size: int=1000,
		min_frequency: int=10,
		suffix: str="</w>",
		bert_normalizer: bool=True,
		special_tokens: Iterable[str]=["<unk>"],
		initial_alphabet: Optional[Iterable[str]]=None,
		save_prefix: Optional[str]=None,
		save_dir: Optional[str]=None
) -> CharBPETokenizer:
	"""
	Trains a Character-level Byte-Pair Encoding tokenizer on a given corpus.
		Returns the trained tokenizer instance.
	"""
	if initial_alphabet is None:
		initial_alphabet = list(set("".join(corpus)))
	tokenizer = CharBPETokenizer(
		suffix=suffix,
		bert_normalizer=bert_normalizer)
	tokenizer.train_from_iterator(
		corpus,
		vocab_size=vocab_size,
		min_frequency=min_frequency,
		special_tokens=special_tokens,
		initial_alphabet=initial_alphabet)
	if save_prefix and save_dir:
		tokenizer.save_model(save_dir, prefix=save_prefix)
	return tokenizer

def refine_tokenizer(
		tokenizer: CharBPETokenizer,
		corpus: Union[str, Iterable[str]],
		extended_vocab_size: int=1200,
		min_frequency: int=10,
		special_tokens: Iterable[str]=["<unk>"],
		save_prefix: Optional[str]=None,
		save_dir: Optional[str]=None
) -> CharBPETokenizer:
	"""
	Refines a pre-trained tokenizer on a given corpus.
		Returns the refined tokenizer instance.
	"""
	initial_alphabet = list(set((list(set("".join(corpus)))+list(tokenizer.get_vocab().keys()))))
	tokenizer.train_from_iterator(
		corpus,
		vocab_size=extended_vocab_size,
		min_frequency=min_frequency,
		special_tokens=special_tokens,
		initial_alphabet=initial_alphabet)
	if save_prefix and save_dir:
		tokenizer.save_model(save_dir, prefix=save_prefix)
	return tokenizer

def load_tokenizer(
		model_dir: str,
		model_prefix: str,
		suffix: str="</w>",
		bert_normalizer: bool=True
) -> CharBPETokenizer:
	"""
	Loads a pre-trained tokenizer from a given directory.
		Returns the tokenizer instance.
	"""
	return CharBPETokenizer(
		vocab=os.path.join(model_dir, model_prefix + "-vocab.json"),
		merges=os.path.join(model_dir, model_prefix + "-merges.txt"),
		suffix=suffix,
		bert_normalizer=bert_normalizer
	)

def tokenize_corpus(
		corpus: Iterable[str],
		tokenizer: CharBPETokenizer
) -> List[List[str]]:
	"""
	Tokenizes a given corpus using a pre-trained tokenizer.
		Returns a list of tokenized sentences.
	"""
	tokenized_corpus = tokenizer.encode_batch(corpus)
	print("Corpus tokenized. Extracting tokens...")
	tokenized_corpus = [s.tokens for s in tqdm(tokenized_corpus)]
	return tokenized_corpus

def trim_corpus(
		tokenized_corpus: List[List[str]],
		min_freq: int=5,
		trim: Optional[str]="<unk>"
) -> List[List[str]]:
	"""
	Trims a tokenized corpus by removing tokens that occur less than a given frequency.
	Trimmed tokens can be replaced with a special token specified in `trim`.
		Returns the trimmed corpus.
	"""
	token_counts = Counter([token for sentence in tokenized_corpus for token in sentence])
	trimmed = []
	for url in tqdm(tokenized_corpus):
		trimmed_url = []
		for token in url:
			if token_counts[token] >= min_freq:
				trimmed_url.append(token)
			elif trim is not None:
				trimmed_url.append(trim)
		trimmed.append(trimmed_url)
	return trimmed

def create_lookup_tables(
		tokenized_corpus: Union[List[str], List[List[str]]]
) -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
	"""
	Creates lookup tables for converting words to indices and vice versa.
		Returns the word-to-index and index-to-word dictionaries and the vocabulary list.
	"""
	if isinstance(tokenized_corpus[0], list):
		tokenized_corpus = [word for sentence in tokenized_corpus for word in sentence]
	word_counts = Counter(tokenized_corpus)
	vocab = sorted(word_counts, key=word_counts.get, reverse=True)
	idx2word = {idx: word for idx, word in enumerate(vocab)}
	word2idx = {word: idx for idx, word in idx2word.items()}
	return word2idx, idx2word, vocab

def subsample(
		tokenized_corpus: List[List[str]],
		word2idx: Dict[str, int],
		threshold: float=1e-2
) -> List[List[int]]:
	"""
	Subsamples a tokenized corpus by removing tokens with a probability proportional to their frequency.
		Returns the subsampled corpus.
	"""
	tokens_idx = [word2idx[word] for word in [token for sentence in tokenized_corpus for token in sentence]]
	corpus_idx = [[word2idx[word] for word in url] for url in tokenized_corpus]
	token_counts = Counter(tokens_idx)
	total_tokens = len(tokens_idx)
	freq_ratios = {token: count / total_tokens for token, count in token_counts.items()}
	p_drop = {token: 1 - np.sqrt(threshold / freq_ratios[token]) for token in token_counts}
	subsampled_corpus = [[token for token in url if random.random() < (1 - p_drop[token])] for url in tqdm(corpus_idx)]
	l1 = len(tokens_idx)
	l2 = len([token for sentence in subsampled_corpus for token in sentence])
	print(f"Subsampled {l1 - l2} tokens from the corpus. {l2/l1:.2%} of the corpus remaining.")
	return subsampled_corpus

def _get_context(
		sentence: List[int],
		idx: int,
		max_window_size: int=5
) -> List[int]:
	"""
	Generates a random-lenght list of context words for a given target word.
		Returns the context words.
	"""
	r = random.randint(1, max_window_size)
	start = max(0, idx - r)
	end = min(idx + r, len(sentence))
	context = sentence[start:idx] + sentence[idx+1:end+1] # +1 because we exclude idx word
	return context

def _get_subbatches(
		words: List[int],
		batch_size: int,
		max_window_size: int=5,
		full: bool=False
) -> Iterator[Tuple[List[int], List[int], List[int]]]:
	"""
	Generates subbatches of center and context words for a given list of words.
	`batch_size` is the number of center words in each batch, but note that these are repeated
	for each context word in order to form pairs, so `len(batch_x) == len(batch_y)`, which might
	be bigger than `batch_size`, and of random size.
		Yields the batch of center words, batch of context words, and the number of times each center
		word has been repeated.
	"""
	if full:
		batch_x, batch_y, ls = [], [], []
		for i, word in enumerate(words):
			x = [word]
			y = _get_context(words, i, max_window_size)
			batch_x.extend(x * len(y))
			batch_y.extend(y)
			ls.append(len(y))
		yield batch_x, batch_y, ls
	else:
		n_batches = len(words) // batch_size
		remainder = len(words) % batch_size

		for i in range(0, n_batches*batch_size, batch_size):
			if (i == (n_batches-1) * batch_size) and (remainder != 0):
				# last batch includes remainder
				batch_of_center_words = words[i:]
			else:
				batch_of_center_words = words[i:i+batch_size]
			batch_x, batch_y, ls = [], [], []
			for ii in range(len(batch_of_center_words)):
				x = [batch_of_center_words[ii]]
				y = _get_context(batch_of_center_words, ii, max_window_size)
				batch_x.extend(x * len(y))
				batch_y.extend(y)
				ls.append(len(y))
			yield batch_x, batch_y, ls

def _get_batches(
		sentences: List[List[int]],
		batch_size: int,
		max_window_size: int=5
) -> Iterator[Tuple[List[int], List[int]]]:
	"""
	Generates batches of center and context words for a given list of sentences.
		Yields the batch of center words and batch of context words.
	"""
	batch_x, batch_y = [], []
	l = 0
	for sentence in sentences:
		for batch in _get_subbatches(sentence, batch_size, max_window_size, full=len(sentence) <= batch_size):
			batch_x.extend(batch[0])
			batch_y.extend(batch[1])
			l += len(batch[2])
			while l >= batch_size:
				rem = l - batch_size
				if rem == 0:
					yield batch_x, batch_y
					batch_x, batch_y = [], []
					l = 0
				else:
					b_sum = sum(batch[2][-rem:])
					yield batch_x[:-b_sum], batch_y[:-b_sum]
					batch_x, batch_y = batch_x[-b_sum:], batch_y[-b_sum:]
					l = rem
	if l > 0:
		yield batch_x, batch_y

def cosine_similarity(
		embedding: nn.Embedding,
		device: torch.device,
		n_valid_words: int=10,
		valid_window: int=30
) -> Tuple[torch.Tensor, torch.Tensor]:
	"""
	Calculates the cosine similarity of validation words with words in the embedding matrix.
	n_valid_words: # of validation words (recommended to have even numbers)
		Returns the validation words and their similarities.
	"""
	all_embeddings = embedding.weight  # (n_vocab, n_embed) 
	# sim = (a . b) / |a||b|
	magnitudes = all_embeddings.pow(2).sum(dim=1).sqrt().unsqueeze(0) # (1, n_vocab)
	# Pick validation words from 2 ranges: (0, window): common words & (1000, 1000+window): uncommon words 
	valid_words = 	random.sample(range(valid_window), n_valid_words//2) + \
					random.sample(range(100, 100+valid_window), n_valid_words//2)
	valid_words = torch.LongTensor(np.array(valid_words)).to(device) # (n_valid_words, 1)
	valid_embeddings = embedding(valid_words) # (n_valid_words, n_embed)
	# (n_valid_words, n_embed) * (n_embed, n_vocab) --> (n_valid_words, n_vocab) / 1, n_vocab)
	similarities = torch.mm(valid_embeddings, all_embeddings.t()) / magnitudes  # (n_valid_words, n_vocab)
	return valid_words, similarities

class SkipGram(nn.Module):
	# implemented as in: https://github.com/lukysummer/SkipGram_with_NegativeSampling_Pytorch
	def __init__(
			self,
			n_vocab: int,
			n_embed: int,
			noise_dist: Optional[torch.Tensor]=None
	):
		super(SkipGram, self).__init__()
		self.n_vocab = n_vocab
		self.n_embed = n_embed
		self.noise_dist = noise_dist
		self.device = torch.device("cpu")

		self.in_embed = nn.Embedding(n_vocab, n_embed)
		self.out_embed = nn.Embedding(n_vocab, n_embed)

		# Initialize embedding tables with uniform distribution
		# no need for Kaiming or other complex methods, skip-gram is simple not deep
		self.in_embed.weight.data.uniform_(-1, 1)
		self.out_embed.weight.data.uniform_(-1, 1)
	
	def forward_input(
			self,
			input_words: torch.Tensor
	):
		input_vectors = self.in_embed(input_words)
		return input_vectors
	
	def forward_target(
			self,
			output_words: torch.Tensor
	):
		output_vectors = self.out_embed(output_words)
		return output_vectors
	
	def forward_noise(
			self,
			batch_size: int,
			n_samples: int=5
	):
		""" Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
		# If no Noise Distribution specified, sample noise words uniformly from vocabulary
		if self.noise_dist is None:
			noise_dist = torch.ones(self.n_vocab)
		else:
			noise_dist = self.noise_dist
			
		noise_words = torch.multinomial(input       = noise_dist,
										num_samples = batch_size*n_samples,
										replacement = True)
		noise_words = noise_words.to(self.device)
		
		# use context matrix for embedding noise samples
		noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
		
		return noise_vectors
	
	def _apply(self, fn, recurse: bool=True):
		super()._apply(fn, recurse)
		self.device = next(self.parameters()).device
		return self
	
class NegativeSamplingLoss(nn.Module):
	# negative sampling: https://www.baeldung.com/cs/nlps-word2vec-negative-sampling
	def __init__(self):
		super().__init__()

	def forward(self, 
				input_vectors: torch.Tensor, 
				output_vectors: torch.Tensor, 
				noise_vectors: torch.Tensor
	):
		
		batch_size, embed_size = input_vectors.shape
		
		input_vectors = input_vectors.view(batch_size, embed_size, 1)   # batch of column vectors
		output_vectors = output_vectors.view(batch_size, 1, embed_size) # batch of row vectors
		
		# log-sigmoid loss for correct pairs
		out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log().squeeze()
		
		# log-sigmoid loss for incorrect pairs
		noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
		noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

		return -(out_loss + noise_loss).mean()  # average batch loss
	
def generate_noise_dist(
		subsampled_corpus: List[List[int]],
		vocab_size: int,
):
	"""
	Generates a noise distribution for negative sampling based on the frequency of tokens in the corpus.
		Returns the noise distribution.
	"""
	# # As defined in the paper by Mikolov et al.
	token_counts = Counter([token for sentence in subsampled_corpus for token in sentence])
	freq_ratio = {token: count / vocab_size for token, count in token_counts.items()}        
	freq_ratio = np.array(sorted(freq_ratio.values(), reverse=True))
	unigram_dist = freq_ratio / freq_ratio.sum()
	noise_dist = torch.from_numpy(unigram_dist**0.75 / np.sum(unigram_dist**0.75))
	return noise_dist

def train_skipgram(
		model: SkipGram,
		criterion: nn.Module,
		optimizer: optim.Optimizer,
		subsampled_corpus: List[List[int]],
		device: torch.device,
		batch_size: int=512,
		context_size: int=5,
		n_epochs: int=1,
		n_neg_samples: int=5,
		print_every: Optional[Union[int, Tuple[int, int]]]=(50, 1000),
		idx2word: Optional[Dict[int, str]] = None,
) -> None:
	"""
	Trains a SkipGram model on a given corpus.
	"""
	if isinstance(print_every, int):
		print_every = (print_every, print_every)
	l = len([t for u in subsampled_corpus for t in u])
	s = int(np.ceil(l/batch_size))
	bs = len(str(s))
	for epoch in range(n_epochs):
		str2 = ""
		for i, (inputs, targets) in enumerate(_get_batches(subsampled_corpus, batch_size, context_size)):
			inputs = torch.LongTensor(inputs).to(device)    # [b*n_context_words]
			targets = torch.LongTensor(targets).to(device)  # [b*n_context_words]
			
			optimizer.zero_grad()
			embedded_input_words = model.forward_input(inputs)
			embedded_target_words = model.forward_target(targets)
			embedded_noise_words = model.forward_noise(batch_size=inputs.shape[0], 
														n_samples=n_neg_samples)

			loss = criterion(embedded_input_words, embedded_target_words, embedded_noise_words)
			loss.backward()
			optimizer.step()
			
			if print_every and (i % print_every[1]) == 0 and idx2word is not None:
				str2 = ""
				valid_idxs, similarities = cosine_similarity(model.in_embed, device)
				_, closest_idxs = similarities.topk(6)
				valid_idxs, closest_idxs = valid_idxs.to('cpu'), closest_idxs.to('cpu')
				
				for ii, v_idx in enumerate(valid_idxs):
					closest_words = [idx2word[idx.item()] for idx in closest_idxs[ii]][1:]
					str2 += idx2word[v_idx.item()] + " | "+ ", ".join(closest_words) + "\n"

			if print_every and (i % print_every[0]) == 0:
				clear_output(wait=True)
				print(f"Epoch {epoch+1:>2}/{n_epochs:>2} | Batch {i+1:>{bs}}/{s:>{bs}} | Loss: {loss.item():.4f}")
				print(str2)

def save_embeddings(
		model: SkipGram,
		save_path: str
) -> None:
	"""
	Saves the embeddings from a trained SkipGram model to a given path.
	Note: file extension should be .pt
	"""
	embeddings = model.in_embed.weight.to('cpu')
	torch.save(embeddings, save_path)

def load_embeddings(
		load_path: str
) -> torch.Tensor:
	"""
	Loads embeddings from a given path.
	Note: file extension should be .pt
	"""
	embeddings = torch.load(load_path)
	embeddings.requires_grad = False
	return embeddings

def save_idx2word(
		idx2word: Dict[int, str],
		save_path: str
) -> None:
	"""
	Saves the index-to-word dictionary to a given path.
	Note: file extension should be .json
	"""
	with open(save_path, "w") as f:
		json.dump(idx2word, f)

def load_idx2word(
		load_path: str
) -> Dict[int, str]:
	"""
	Loads the index-to-word dictionary from a given path.
	Note: file extension should be .json
	"""
	with open(load_path, "r") as f:
		idx2word = json.load(f)
	return {int(k): v for k, v in idx2word.items()}

def visualize_embeddings(
		embeddings: torch.Tensor,
		idx2word: Dict[int, str],
		n: int = 200,
		method: Literal["pca", "tsne", "both"] = "tsne",
		figsize: Tuple[int, int] = (10, 10)
) -> None:
	"""
	Visualizes embeddings using PCA, t-SNE, or both.
	"""
	if method in ["pca", "both"]:
		pca = PCA(n_components=2)
		embed_pca = pca.fit_transform(embeddings[:n, :])
	if method in ["tsne", "both"]:
		tsne = TSNE()
		embed_tsne = tsne.fit_transform(embeddings[:n, :])

	if method == "both":
		fig, axes = plt.subplots(1, 2, figsize=figsize)
		ax_pca, ax_tsne = axes
	else:
		fig, ax = plt.subplots(figsize=figsize)
		if method == "pca":
			ax_pca = ax
		else:
			ax_tsne = ax

	if method in ["pca", "both"]:
		for i in range(n):
			ax_pca.scatter(embed_pca[i, 0], embed_pca[i, 1], color="steelblue")
			ax_pca.annotate(idx2word[i], (embed_pca[i, 0], embed_pca[i, 1]), alpha=0.7)
		ax_pca.set_title('PCA')
	if method in ["tsne", "both"]:
		for i in range(n):
			ax_tsne.scatter(embed_tsne[i, 0], embed_tsne[i, 1], color="steelblue")
			ax_tsne.annotate(idx2word[i], (embed_tsne[i, 0], embed_tsne[i, 1]), alpha=0.7)
		ax_tsne.set_title('t-SNE')

	plt.show()

def closest(
		word: Union[str, torch.Tensor],
		embeddings: torch.Tensor,
		idx2word: Dict[int, str],
		n: int=10,
) -> List[Tuple[str, float]]:
	"""
	Returns the closest words to a given word in the embedding space.
	"""
	if isinstance(word, str):
		word2idx = {v: k for k, v in idx2word.items()}
		word = embeddings[word2idx[word]]
	all_dists = [(w, torch.dist(word, embeddings[w])) for w in idx2word.keys()]
	return [(idx2word[idx], v.item()) for idx, v in sorted(all_dists, key=lambda t: t[1])[:n]]

def analogy(
		w1: str,
		w2: str,
		w3: str,
		embeddings: torch.Tensor,
		idx2word: Dict[int, str],
		n: int=5,
		filter_given: bool=True
) -> None:
	"""
	Performs an analogy task on a given set of words: w1 is to w2 as w3 is to ?.
	"""
	print('%s is to %s as %s is to ?]' % (w1, w2, w3))
   
	# w2 - w1 + w3 = w4
	word2idx = {v: k for k, v in idx2word.items()}
	w1, w2, w3 = word2idx[w1], word2idx[w2], word2idx[w3]
	closest_words = closest(embeddings[w2] - embeddings[w1] + embeddings[w3],
						   embeddings, idx2word, n)
	
	# Optionally filter out given words
	if filter_given:
		closest_words = [t for t in closest_words if t[0] not in [w1, w2, w3]]
		
	def print_tuples(tuples):
		for tuple in tuples:
			print('(%.4f) %s' % (tuple[1], tuple[0]))

	print_tuples(closest_words[:n])

def get_included(
		token: str,
		vocab: List[str]
) -> str:
	"""
	Returns the first token in vocab that includes token, or "<unk>" if none found.
	"""
	if not isinstance(vocab, list):
		vocab = list(vocab)
	# assuming vocab is sorted by frequency, candidates are too
	for token2 in vocab:
		t1 = re.sub(r"<.*>", "", token)
		t2 = re.sub(r"<.*>", "", token2)
		if t1 and t2 and t1 in t2:
			return token2
	return "<unk>"

def get_idx(
		tokens: Iterable[str],
		word2idx: Dict[str, int]
) -> List[int]:
	tokens_idx = []
	for token in tokens:
		try: tokens_idx.append(word2idx[token])
		except KeyError: tokens_idx.append(word2idx[get_included(token, word2idx.keys())])
	return torch.tensor(tokens_idx, dtype=torch.int)