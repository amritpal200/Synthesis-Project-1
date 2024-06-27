
# Importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pandas as pd

import torch.nn.functional as F
from typing import List, Literal
import math
import matplotlib.pyplot as plt

# Check if GPU is available
assert torch.cuda.is_available(), "GPU is not enabled"

#  use gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create Dataset
class DatasetFromCsvFile(Dataset):

    """
    A PyTorch Dataset class to load sequences and their corresponding labels from a CSV file.

    Parameters:
    - filename (str): The path to the CSV file containing the data.
    - max_sequence_length (int): The maximum length for each sequence. Sequences longer than this will be truncated, and shorter ones will be padded.

    Attributes:
    - data (pd.DataFrame): The raw data loaded from the CSV file.
    - max_sequence_length (int): The maximum length for each sequence.
    - sequences (List[torch.Tensor]): List of preprocessed sequences.
    - outputs (List): List of output labels corresponding to each sequence.
    """

    def __init__(self, filename: str, max_sequence_length: int):
        self.data = pd.read_csv(filename)
        self.max_sequence_length = max_sequence_length
        self.sequences, self.outputs = self.preprocess_sequences()


    def preprocess_sequences(self) :

        """
        Preprocesses the sequences from the raw data, including padding or truncating and converting characters to their ASCII values.

        Returns:
        - final_sequences (List[torch.Tensor]): The preprocessed and padded/truncated sequences.
        - outputs (List): The output labels for each sequence.
        """

        final_sequences = []
        outputs = []

        for row in self.data.itertuples():
            sequences = []

            # Process server_name
            # sequ = [ord(char) for char in row.server_name]
            # sequ = self.pad_or_truncate(sequ)
            # sequences.extend(sequ)

            # Process URL
            sequ = [ord(char) for char in str(row.URL)]
            sequ = self.pad_or_truncate(sequ)
            sequences.extend(sequ)

            # Append status and bytess
            sequences.append(row.status)
            sequences.append(row.bytess)

            # Process user_agent
            sequ = [ord(char) for char in row.user_agent]
            sequ = self.pad_or_truncate(sequ)
            sequences.extend(sequ)

            # Append petition columns
            sequences.append(row.petition__)
            sequences.append(row.petition_CONNECT)
            sequences.append(row.petition_GET)
            sequences.append(row.petition_HEAD)
            sequences.append(row.petition_OPTIONS)
            sequences.append(row.petition_POST)
            sequences.append(row.petition_USER)
            sequences.append(row.petition_PUT)

            # Convert to tensor
            sequences = torch.tensor(sequences, dtype=torch.float).unsqueeze(1)
            final_sequences.append(sequences)
            outputs.append(row.level)

        return final_sequences, outputs


    def pad_or_truncate(self, sequ: List[int]) -> List[int]:
        """
        Pads or truncates the sequence to the maximum sequence length.

        Parameters:
        - sequ (List[int]): The sequence to be padded or truncated.

        Returns:
        - sequ (List[int]): The padded or truncated sequence.
        """


        if len(sequ) < self.max_sequence_length:
            sequ.extend([0] * (self.max_sequence_length - len(sequ)))
        elif len(sequ) > self.max_sequence_length:
            sequ = sequ[:self.max_sequence_length]
        return sequ

    def __len__(self) -> int:

        """
        Returns the total number of sequences in the dataset.

        Returns:
        - length (int): The number of sequences.
        """

        return len(self.sequences)

    def __getitem__(self, idx: int):

        """
        Retrieves the sequence and output label at the specified index.

        Parameters:
        - idx (int): The index of the sequence to retrieve.

        Returns:
        - sequence (torch.Tensor): The preprocessed sequence at the specified index.
        - output (int): The output label corresponding to the sequence.
        """

        assert 0 <= idx < len(self.sequences), "Index out of range"
        return self.sequences[idx], self.outputs[idx]



class MultiScaleAttention(nn.Module):
	"""
	MultiScaleAttention module that combines local and global multi-head attention mechanisms
	and includes a feed-forward network with layer normalization.

	Parameters:
	- embed_dim: int, dimensionality of the embedding space.
	- num_heads: int, number of attention heads.
	- local_window_size: int, size of the local attention window.
	- ffn_dim: int, dimensionality of the feed-forward network's hidden layer.
	"""

	def __init__(self, embed_dim, num_heads, local_window_size, ffn_dim):
		super(MultiScaleAttention, self).__init__()
		self.embed_dim = embed_dim
		self.num_heads = num_heads
		self.local_window_size = local_window_size
		self.ffn_dim = ffn_dim

		# Local attention
		self.local_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
		# Global attention
		self.global_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

		# Feed-forward network
		self.ffn = nn.Sequential(
			nn.Linear(embed_dim, ffn_dim),
			nn.ReLU(),
			nn.Linear(ffn_dim, embed_dim)
		)

		# Layer normalization
		self.norm1 = nn.LayerNorm(embed_dim)
		self.norm2 = nn.LayerNorm(embed_dim)

	def forward(self, x):

		"""
		Forward pass through the MultiScaleAttention module.

		Parameters:
		- x: torch.Tensor

		Returns:
		- x: torch.Tensor, output tensor of the same shape as input.
		"""

		# Extract dimensions from the input tensor
		batch_size, seq_length, embed_dim = x.size()
		assert embed_dim == self.embed_dim

		# Local attention
		local_attention_mask = self.create_local_mask(seq_length, self.local_window_size, x.device)
		local_output, _ = self.local_attention(x, x, x, attn_mask=local_attention_mask)

		# Global attention
		global_output, _ = self.global_attention(x, x, x)

		# Combine local and global attention
		combined_output = local_output + global_output

		# Add & Norm after attention
		x = x + combined_output # Residual connection
		x = self.norm1(x)       # Layer normalization

		# Feed-forward network
		ffn_output = self.ffn(x)

		# Add & Norm after feed-forward network
		x = x + ffn_output # Residual connection
		x = self.norm2(x)  # Layer normalization

		return x

	def create_local_mask(self, seq_length, window_size, device):

		"""
		Create a local attention mask that restricts attention to a local window around each position.

		Parameters:
		- seq_length: int, the length of the sequence.
		- window_size: int, the size of the local window.
		- device: torch.device, the device on which the mask will be created.

		Returns:
		- mask: torch.Tensor, boolean mask of shape (seq_length, seq_length).
		"""
		# Initialize mask with all True values
		mask = torch.ones(seq_length, seq_length, dtype=torch.bool, device=device)
		for i in range(seq_length):
			# Set False for positions outside the local window
			for j in range(max(0, i - window_size), min(seq_length, i + window_size + 1)):
				mask[i, j] = False
		return mask



class TransformerAttention(nn.Module):
  """
    Transformer-based attention model with multi-scale attention mechanisms and regression head.

    Parameters:
    - dim (int): The dimension of the embeddings.
    - depth (int): The number of attention layers.
    - heads (int): The number of attention heads.
    - mlp_dim (int): The dimension of the MLP layer in the regression head.
    - max_sequence_length (int): The maximum length of the input sequences.
    - local_window_size (int, optional): The size of the local window for local attention. Default is 5.
    - device (torch.device): The device on which the model will be run.
    """


  def __init__(self, *, dim: int, depth: int, heads: int, mlp_dim: int, max_sequence_length: int, local_window_size: int = 5, device: torch.device):
    super().__init__()

    self.device = device
    self.max_sequence_length = max_sequence_length

    # Linear layer to convert ASCII codes to embeddings
    self.codiascii_to_embedding = nn.Linear(1, dim)
    # Positional embeddings
    self.pos_embedding = nn.Parameter(torch.randn(1,self.max_sequence_length+1, dim))
    # Class token
    self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    # Multi-scale attention layers
    self.multi_scale_attention_layers = nn.ModuleList([
            MultiScaleAttention(dim, heads, local_window_size, mlp_dim) for _ in range(depth)
        ])

    # Identity layer for CLS token extraction
    # self.to_cls_token = nn.Identity()
    # Dropout layer
    self.dropout = nn.Dropout(p=0.1)

    # Regression head with two linear layers and activation
    self.regression_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, 1),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    """
        Forward pass for the TransformerAttention model.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, seq_length, 1).

        Returns:
        - output (torch.Tensor): Output tensor of shape (batch_size, 1).
    """

    # Convert ASCII codes to embeddings
    x = self.codiascii_to_embedding(x)

    # Expand CLS token and concatenate with input
    cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # print(f"x shape after cls_token is {x.shape}")

    # Add positional embeddings
    x += self.pos_embedding

    # Apply multi-scale attention and normalization layers
    for layer in self.multi_scale_attention_layers:
      x = layer(x)

    # Extract CLS token, apply dropout, and pass through regression head
    x = x.mean(dim=1)
    x = self.dropout(x)
    return self.regression_head(x)

# Define the validation function
def validate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    # Set model to evaluation mode
    model.eval()
    # Initialize validation loss
    val_loss = 0.0
    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the validation data
        for data, target in data_loader:
            data = data.to(device) # Move data to device
            target = target.to(device).float() # Move target to device
            output = model(data).squeeze() # Forward pass
            loss = F.mse_loss(output, target) # Compute loss
            val_loss += loss.item() # Accumulate loss

    val_loss /= len(data_loader) # Compute average loss
    return val_loss # Return average loss

# Define the training function
def train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    val_loader: DataLoader,
    loss_history: List[float],
    device: torch.device,
    grad_clip_value: float = 1.0,
    model_save_path: str = 'best_model.pt',
    best_loss: float = 100.0
) -> None:
  #
  total_samples = len(data_loader.dataset) # Get total number of samples
  model.train() # training mode

  total_batches = len(data_loader) # Get total number of batches
  print_interval = math.ceil(total_batches / 10) # Define printing interval

  for i, (data, target) in enumerate(data_loader): # Iterate over the data
    data = data.to(device) # Move data to device
    target = target.to(device).float() # Move target to device
    optimizer.zero_grad() # Zero the gradients


    output = model(data).squeeze()  # Squeeze to match target shape
    loss = F.mse_loss(output, target)  # Use Mean Squared Error loss

    loss.backward() # Backward pass

    # Apply gradient clipping
    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

    optimizer.step() # Update weights

    # Print training loss
    if i % print_interval == 0 or i == total_batches - 1:
      print(f'[{i * len(data):5}/{total_samples:5} ({100 * i / len(data_loader):3.0f}%)]  Loss: {loss.item():.4f}')
      loss_history.append(loss.item())


  # Validate the model
  val_loss = validate(model, val_loader, device)

  print(f'Validation Loss: {val_loss:.4f}')

  # Save the best model
  if val_loss < best_loss:
      print(f'Validation loss improved from {best_loss:.4f} to {val_loss:.4f}. Saving model...')
      best_loss = val_loss
      torch.save(model.state_dict(), model_save_path)

  return best_loss

# Set hyperparameters
max_sequence_length_dataset = 150
max_sequence_length = 310
batch_size = 150
dim = 200
depth = 2
heads = 4
mlp_dim = 512
local_window_size = 5
LR = 0.0005
SDLR_MILESTONES = [1,2]
GAMMA = 0.1
N_EPOCHS = 2
# Initialize best_loss and loss_history
best_loss = float('inf')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Set device

train_filename = '/content/train.csv' # Path to the training data
train_dataset = DatasetFromCsvFile(train_filename, max_sequence_length_dataset) # Create training dataset
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # Create data loader

val_filename = '/content/validate.csv' # Path to the validation data
val_dataset = DatasetFromCsvFile(val_filename, max_sequence_length_dataset) # Create validation dataset
val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True) # Create data loader

# Initialize the model, optimizer, scheduler, and loss history
model = TransformerAttention(dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, max_sequence_length=max_sequence_length, local_window_size=local_window_size, device=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=SDLR_MILESTONES, gamma=GAMMA)
loss_history = []

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
# Count parameters that require gradients
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Training loop
for epoch in range(N_EPOCHS):
  print(f'Epoch {epoch+1}/{N_EPOCHS}') # Print epoch
  # Train the model
  best_loss = train_epoch(model, optimizer, train_data_loader, val_data_loader, loss_history, device)
  # Plotting the training loss after each epoch
  plt.figure(figsize=(10, 5))
  plt.plot(loss_history, label='Training Loss')
  plt.xlabel('Batch')
  plt.ylabel('Loss')
  plt.title('Training Loss Over Time')
  plt.legend()
  plt.show()


def evaluate(
    model: torch.nn.Module,
    data_loader: DataLoader,
    loss_history: List[float],
    device: torch.device
) -> float:
  model.eval() # evaluation mode


  with torch.no_grad(): # Disable gradient computation
    for i, (data, target) in enumerate(data_loader): # Iterate over the data
      data = data.to(device)
      target = target.to(device).float()
      output = model(data).squeeze()  # Squeeze to match target shape
      loss = F.mse_loss(output, target)  # Use Mean Squared Error loss
      print(f"loss is {loss}")
      print(f"correct output is {target}")
      print(f"predicted output is output {output}")
      print("-------")
      for i in range(len(target)):
        print(f"correct is {target[i]} and predicted is {output[i]}")

dataset = DatasetFromCsvFile("/content/test_sampled.csv", max_sequence_length_dataset) # Create test dataset
val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True) # Create data loader
evaluate(model, val_loader, [], device) # Evaluate the model

