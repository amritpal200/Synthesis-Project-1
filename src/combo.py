import pandas as pd
import os
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from skipgram import *
import deeplog

class AutoEncoder(nn.Module):
	def __init__(
			self,
			input_size: int,
			hidden_size: Tuple[int],
	):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(input_size, hidden_size[0]),
			nn.ReLU(),
			nn.Linear(hidden_size[0], hidden_size[1]),
			nn.ReLU(),
			nn.Linear(hidden_size[1], hidden_size[2])
		)
		self.decoder = nn.Sequential(
			nn.Linear(hidden_size[2], hidden_size[1]),
			nn.ReLU(),
			nn.Linear(hidden_size[1], hidden_size[0]),
			nn.ReLU(),
			nn.Linear(hidden_size[0], input_size)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		emb = self.encoder(x)
		x = self.decoder(emb)
		return x
	
	def apply_weight_init(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0)
	
class AEModel(nn.Module):
	def __init__(
			self,
			vector_dim: int,
			hidden_dim: Tuple[int],
			lr: float,
	):
		super(AEModel, self).__init__()
		self.model = AutoEncoder(vector_dim, hidden_dim)
		self.model.apply_weight_init()
		self.loss_fn = nn.MSELoss()
		self._loss_fn = nn.MSELoss(reduction="none")
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

		self.vector_dim = vector_dim
		self.hidden_dim = hidden_dim

	def train(
			self,
			train_loader: DataLoader,
			n_epochs: int,
			lr: float,
	) -> List[float]:
		losses = []
		self.model.train()
		for epoch in range(n_epochs):
			for i, (_, pv) in enumerate(train_loader):
				pv = pv.to(self.model.device)
				self.optimizer.zero_grad()
				pv_pred = self.model(pv)
				loss = self.loss_fn(pv_pred, pv)
				loss.backward()
				self.optimizer.step()
				losses.append(loss.item())
				print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item()}", end="\r")
		return losses
	
	def evaluate(
			self,
			test_loader: DataLoader,
			device: torch.device,
	) -> torch.Tensor:
		self.model.eval()
		losses = []
		with torch.no_grad():
			for i, (_, pv) in enumerate(test_loader):
				pv = pv.to(device)
				pv_pred = self.model(pv)
				loss = self.loss_fn(pv_pred, pv)
				losses.extend(loss.mean(dim=1).tolist())
				print(f"Batch {i+1}, Loss: {loss.mean().item()}", end="\r")
		return torch.tensor(losses)
	
class Combo:
	def __init__(
			self,
			vector_dim: int,
			dl_hidden_dim: int,
			dl_n_layers: int,
			ae_hidden_dim: Tuple[int],
			device: torch.device,
			lr: Tuple[float] = (1e-3, 1e-3)
	):
		self.deeplog = deeplog.PVAnomalyDetectionModel(vector_dim, dl_hidden_dim, dl_n_layers, device, lr[0])
		self.autoencoder = AEModel(vector_dim, ae_hidden_dim, lr[1])

		self.vector_dim = vector_dim
		self.dl_hidden_dim = dl_hidden_dim
		self.dl_n_layers = dl_n_layers
		self.ae_hidden_dim = ae_hidden_dim
		self.device = device
		self.lr = lr

	def train_deeplog(
			self,
			train_loader: DataLoader,
			n_epochs: int,
			seq_norm: bool = False,
	) -> List[float]:
		return self.deeplog.train(train_loader, n_epochs, seq_norm)
	
	def train_autoencoder(
			self,
			train_loader: DataLoader,
			n_epochs: int,
	) -> List[float]:
		return self.autoencoder.train(train_loader, n_epochs, self.lr[1])
	
			

	def save(self, path: str):
		torch.save({
			"deeplog": self.deeplog.model.state_dict(),
			"autoencoder": self.autoencoder.state_dict(),
			"vector_dim": self.vector_dim,
			"dl_hidden_dim": self.dl_hidden_dim,
			"dl_n_layers": self.dl_n_layers,
			"dl_iqr_interval": self.deeplog.iqr_interval,
			"ae_hidden_dim": self.ae_hidden_dim
		}, path)

def load_combo(path: str, device: torch.device) -> Combo:
	checkpoint = torch.load(path)
	combo = Combo(
		checkpoint["vector_dim"],
		checkpoint["dl_hidden_dim"],
		checkpoint["dl_n_layers"],
		checkpoint["ae_hidden_dim"],
		device
	)
	combo.deeplog.model.load_state_dict(checkpoint["deeplog"])
	combo.deeplog.iqr_interval = checkpoint["dl_iqr_interval"]
	combo.autoencoder.load_state_dict(checkpoint["autoencoder"])
	return combo
