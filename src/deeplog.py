import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from skipgram import *
import os
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.abspath(""))

class PVModel(nn.Module):
	def __init__(
			self,
			vector_dim: int,
			hidden_dim: int,
			n_layers: int=1,
			dropout_prob: float=0.5
	):
		"""
		Parameter Value Model to predict the next log vector in a sequence.
		"""
		super(PVModel, self).__init__()
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		
		self.lstm = nn.LSTM(vector_dim, hidden_dim, n_layers, batch_first=True)
		self.dropout = nn.Dropout(dropout_prob)
		self.fc1 = nn.Linear(hidden_dim, 256)
		self.bn1 = nn.BatchNorm1d(256)
		self.fc2 = nn.Linear(256, vector_dim)
		
	def forward(self, x, hc):
		"""
		Predict the next log vector in the sequence
		"""
		out, hc = self.lstm(x, hc)
		out = self.dropout(out[:, -1, :])
		out = self.fc1(out)
		
		if out.size(0) > 1:
			out = self.bn1(out)
		
		out = torch.relu(out)
		out = self.fc2(out)
		return out, hc
	
	def init_hidden(self, batch_size, device='cpu'):
		"""
		Initialize hidden state and cell state
		"""
		hidden_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
		cell_state = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(device)
		return hidden_state, cell_state
	
	def apply_weight_init(self):
		"""
		Apply weight initialization to the model
		"""
		for m in self.modules():
			if isinstance(m, nn.LSTM):
				for name, param in m.named_parameters():
					if 'weight_ih' in name:
						nn.init.xavier_uniform_(param.data)
					elif 'weight_hh' in name:
						nn.init.orthogonal_(param.data)
					elif 'bias' in name:
						param.data.fill_(0)
			elif isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0)

def sort_by_group(data: pd.DataFrame) -> pd.DataFrame:
	"""
	Sort the dataframe by group and index
	"""
	df_sorted = data.copy()
	df_sorted["index"] = df_sorted.index
	df_sorted = df_sorted.sort_values(["group", "index"])
	df_sorted = df_sorted.drop(columns=["index"])
	df_sorted = df_sorted.reset_index(drop=True)
	# Also normalize the elapsed time column
	df_sorted["elapsed"] = (df_sorted["elapsed"] - df_sorted["elapsed"].mean()) / df_sorted["elapsed"].std()
	return df_sorted

class GroupSequenceDataset(Dataset):
	
	def __init__(
			self,
			data: pd.DataFrame,
			seq_len: int=15
	):
		"""
		Grouped sequence dataset for training the PV model.
		Takes a dataset sorted by group and index, and creates a new dataset of parameter
		value vectors, where text columns are embedded. When indexing the dataset, you get
		a tuple of the previous vectors in the group and the current vector.
		"""
		self.df = data
		# Prepare the groups
		self.seq_len = seq_len
		self.gs = self.df["group"].unique()
		self.groups_idx = {g: self.df[self.df["group"] == g].index.to_numpy() for g in self.gs}
		self.groups = torch.tensor(self.df["group"].to_numpy(), dtype=torch.int32)
		self.pos_in_group = torch.empty(len(self.df), dtype=torch.int32)
		for group in self.groups_idx.values():
			self.pos_in_group[group] = torch.arange(len(group), dtype=torch.int32)

		# extract embeddings
		print("Loading embeddings...")
		embeddings_url = load_embeddings(os.path.join(ROOT_DIR, "models", "embeddings-url.pt"))
		idx2word_url = load_idx2word(os.path.join(ROOT_DIR, "models", "idx2word-url.json"))
		tokenizer_url = load_tokenizer(os.path.join(ROOT_DIR, "models"), "charbpe-url")
		embeddings_referer = load_embeddings(os.path.join(ROOT_DIR, "models/embeddings-referer.pt"))
		idx2word_referer = load_idx2word(os.path.join(ROOT_DIR, "models/idx2word-referer.json"))
		tokenizer_referer = load_tokenizer(os.path.join(ROOT_DIR, "models"), "charbpe-referer")
		embeddings_useragent = load_embeddings(os.path.join(ROOT_DIR, "models/embeddings-useragent.pt"))
		idx2word_useragent = load_idx2word(os.path.join(ROOT_DIR, "models/idx2word-useragent.json"))
		tokenizer_useragent = load_tokenizer(os.path.join(ROOT_DIR, "models"), "charbpe-useragent")

		print("Extracting embeddings...")
		url_embeddings = extract_embeddings(
			sequence = self.df["URL"],
			embeddings = embeddings_url,
			idx2word = idx2word_url,
			tokenizer = tokenizer_url
		)
		referers_embeddings = extract_embeddings(
			sequence = self.df["referer"],
			embeddings = embeddings_referer,
			idx2word = idx2word_referer,
			tokenizer = tokenizer_referer
		)
		useragents_embeddings = extract_embeddings(
			sequence = self.df["user-agent"],
			embeddings = embeddings_useragent,
			idx2word = idx2word_useragent,
			tokenizer = tokenizer_useragent
		)

		print("Preparing dataset...")
		self.df = self.df.drop(columns=["URL", "referer", "user-agent", "level", "group"])
		self.df = self.df.reindex(columns=[
			"bytes", "elapsed", "IP_oct0", "IP_oct1", "IP_oct2", "IP_oct3", "month_sin",
			"month_cos", "day_sin", "day_cos", "weekday_sin", "weekday_cos",
			"hour_sin", "hour_cos", "minute_sin", "minute_cos", "petition_-",
			"petition_GET", "petition_HEAD", "petition_POST", "petition_other",
			"status_1", "status_2", "status_3", "status_4", "status_5"
		])
		# Create the parameter value vectors
		self.df = self.df.astype(np.float32)
		self.pv_vectors = torch.empty(
			(len(self.df), self.df.shape[1] + url_embeddings[0].shape[1] + referers_embeddings[0].shape[1] + useragents_embeddings[0].shape[1]), dtype=torch.float32)
		for i in tqdm(range(len(self.df))):
			self.pv_vectors[i, :self.df.shape[1]] = torch.from_numpy(self.df.iloc[i].values)
			self.pv_vectors[i, self.df.shape[1]:] = torch.cat((url_embeddings[i].mean(0), referers_embeddings[i].mean(0), useragents_embeddings[i].mean(0)))

		# Create the distribution for later padding
		means = torch.mean(self.pv_vectors, dim=0)
		stds = torch.std(self.pv_vectors, dim=0)+1e-6
		self.dist = torch.distributions.normal.Normal(means, stds)

		# Free up memory
		del self.df, url_embeddings, referers_embeddings, useragents_embeddings
		del embeddings_url, embeddings_referer, embeddings_useragent

	def __len__(self) -> int:
		return len(self.pv_vectors)
	
	def get_group(self, idx: int) -> torch.Tensor:
		"""
		Get the group of vectors of the current index
		"""
		return self.pv_vectors[self.groups_idx[self.groups[idx].item()]]

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Get the previous vectors in the group and the current vector
		"""
		group = self.get_group(idx)
		pos_in_group = self.pos_in_group[idx]

		# Truncate or pad the sequence to seq_len
		if pos_in_group >= self.seq_len:
			x = group[pos_in_group - self.seq_len:pos_in_group]
		else:
			padding = torch.zeros((self.seq_len - pos_in_group, group.shape[1]), dtype=torch.float32)
			# padding = self.dist.sample((self.seq_len - pos_in_group,)).squeeze(1)
			x = torch.cat((padding, group[:pos_in_group]), dim=0)
			
		y = group[pos_in_group]
		return x, y

class PVAnomalyDetectionModel():
	def __init__(
			self,
			vector_dim: int,
			hidden_dim: int,
			n_layers: int,
			device: torch.device,
			lr: float=1e-3
		):
		"""
		Parameter Value Anomaly Detection Model to detect anomalies in sequences of parameter value vectors.
		"""
		# Initialize the model, loss function and optimizer
		self.model = PVModel(vector_dim, hidden_dim, n_layers).to(device)
		self.model.apply_weight_init()
		self.loss_fn = nn.MSELoss()
		self._loss_fn = nn.MSELoss(reduction='none')
		self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

		self.vector_dim = vector_dim
		self.hidden_dim = hidden_dim
		self.n_layers = n_layers
		self.device = device
		self.iqr_interval = (0, 0)
	
	def forward(
			self,
			x: torch.Tensor,
			seq_norm: bool=False
	) -> torch.Tensor:
		"""
		Forward pass of the model to predict the next log vector in the sequence.
		- If seq_norm is True, the sequence is normalized using the mean and std of itself.
		"""
		if seq_norm:
			mean = x.mean(dim=1, keepdim=True)
			std = x.std(dim=1, keepdim=True)
			x = (x - mean) / (std + 1e-6)
		x = x.to(self.device)
		hc = self.model.init_hidden(x.size(0))
		hc = (hc[0].to(self.device), hc[1].to(self.device))
		out, hc = self.model(x, hc)
		return out

	def train(
			self,
			train_loader: DataLoader,
			n_epochs: int,
			seq_norm: bool=False
	) -> List[float]:
		"""
		Train the model on the training data for n_epochs.
		- If seq_norm is True, the sequence is normalized using the mean and std of itself.
		"""
		losses = []
		self.model.train()
		for epoch in range(n_epochs):
			for i, (x, y) in enumerate(train_loader):
				self.optimizer.zero_grad()
				out = self.forward(x, seq_norm)
				loss = self.loss_fn(out, y.to(self.device))
				loss.backward()
				self.optimizer.step()
				losses.append(loss.item())
				print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}", end="\r")
		return losses
	
	def predict(
			self,
			x: torch.Tensor,
			seq_norm: bool=False
	) -> torch.Tensor:
		"""
		Predict the next log vector in the sequence (without updating the model weights).
		- If seq_norm is True, the sequence is normalized using the mean and std of itself.
		"""
		self.model.eval()
		with torch.no_grad():
			out = self.forward(x, seq_norm)
			out = out.detach()
		return out
	
	def compute_iqr_interval(
			self,
			val_loader: DataLoader,
			seq_norm: bool=False,
			q1: float=0.25,
			q2: float=0.75,
			k: float=1.5
	):
		"""
		Compute the IQR interval for anomaly detection using the validation data.
		- If seq_norm is True, the sequence is normalized using the mean and std of itself.
		"""
		self.model.eval()
		# Compute the errors on the validation set
		with torch.no_grad():
			all_errors = []
			for i, (x, y) in tqdm(enumerate(val_loader), total=len(val_loader)):
				pred = self.predict(x, seq_norm).to(self.device)
				y = y.to(self.device)
				loss = self.loss_fn(pred, y)
				all_errors.append(loss.item())
			all_errors = torch.tensor(all_errors)
		# Compute the IQR interval
		q1 = torch.quantile(all_errors, q1)
		q2 = torch.quantile(all_errors, q2)
		iqr = q2 - q1
		self.iqr_interval = (q1 - k * iqr, q2 + k * iqr)

	def detect(
			self,
			sequences: torch.Tensor,
			vecs: torch.Tensor,
			seq_norm: bool=False,
			return_probs: bool=False
		) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
		"""
		Detect anomalies in the sequences of parameter value vectors.
		- If seq_norm is True, the sequence is normalized using the mean and std of itself.
		- If return_probs is True, the function also returns the anomaly probabilities.
		"""
		self.model.eval()
		with torch.no_grad():
			# Compute the errors on the validation set
			preds = self.predict(sequences, seq_norm).to(self.device)
			losses = self._loss_fn(preds, vecs.to(self.device))
			losses = losses.mean(dim=1)
			iqr_interval = torch.tensor(self.iqr_interval).to(self.device)
			# Detect anomalies
			anomalies = losses > iqr_interval[1]
			if return_probs:
				# Compute the anomaly probabilities
				iqr_width = iqr_interval[1] - iqr_interval[0]
				distances = (losses - iqr_interval[1]) / iqr_width
				probs = torch.sigmoid(distances * 10)
				return losses, anomalies, probs
		return losses, anomalies
	
	def evaluate(
			self,
			test_loader: DataLoader,
			device: torch.device,
			seq_norm: bool=False,
			return_probs: bool=False
	) -> torch.Tensor:
		"""
		Evaluate the model on the test data.
		- If seq_norm is True, the sequence is normalized using the mean and std of itself.
		- If return_probs is True, the function also returns the anomaly probabilities.
		"""
		self.model.eval()
		losses, anomalies, probs = [], [], []
		with torch.no_grad():
			for i, (x, y) in enumerate(test_loader):
				if return_probs:
					loss, anomaly, prob = self.detect(x, y, seq_norm, return_probs)
					probs.append(prob)
				else:
					loss, anomaly = self.detect(x, y, seq_norm)
				losses.append(loss)
				anomalies.append(anomaly)
				print(f"Batch {i+1}, Loss: {loss.mean().item()}", end="\r")
		losses = torch.cat(losses).cpu()
		anomalies = torch.cat(anomalies).cpu()
		if return_probs:
			probs = torch.cat(probs).cpu()
			return losses, anomalies, probs
		return losses, anomalies
	
	def save(self, path: str):
		"""
		Save the model and its parameters to a file.
		"""
		torch.save({
			"model_state_dict": self.model.state_dict(),
			"iqr_interval": self.iqr_interval,
			"vector_dim": self.model.lstm.input_size,
			"hidden_dim": self.model.lstm.hidden_size,
			"n_layers": self.model.lstm.num_layers
		}, path)

def load_model(path, device: torch.device):
	"""
	Load a model and its parameters from a file.
	"""
	checkpoint = torch.load(path)
	model = PVAnomalyDetectionModel(
		vector_dim=checkpoint["vector_dim"],
		hidden_dim=checkpoint["hidden_dim"],
		n_layers=checkpoint["n_layers"],
		device=device
	)
	model.model.load_state_dict(checkpoint["model_state_dict"])
	model.iqr_interval = checkpoint["iqr_interval"]
	return model
