import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import autocast, GradScaler
from ..model.decoder import Decoder
from ..model.encoder import Encoder
from ..model.transformer import EncoderDecoder
from ..utils.dataset import MyDataset, collate_fn

def train_one_epoch(data_loader, epoch_index, writer, model, optimizer, criterion, scaler, device):
    total_loss = 0.0
    model.train()

    for batch_index, item in enumerate(data_loader):
        src_input = item['src'].to(device)
        tgt_input = item['tgt'][:, :-1].to(device)
        labels = item['tgt'][:, 1:].to(device)

        optimizer.zero_grad()
        with autocast():
            outputs = model(src_input, tgt_input)
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
        total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if (batch_index + 1) % 10000 == 0:
            avg_loss_10000 = total_loss / (batch_index + 1)
            print(f"Epoch {epoch_index + 1}, Batch {batch_index + 1}, Avg Loss Last 10000 Batches: {avg_loss_10000:.2f}")

    epoch_avg_loss = total_loss / len(data_loader)
    return epoch_avg_loss

def validate(data_loader, model, criterion, device):
    running_vloss = 0.0
    model.eval()
    with torch.no_grad():
        for item in data_loader:
            src_input = item['src'].to(device)
            tgt_input = item['tgt'][:, :-1].to(device)
            labels = item['tgt'][:, 1:].to(device)
            outputs = model(src_input, tgt_input)
            outputs = outputs.reshape(-1, outputs.shape[-1])
            labels = labels.reshape(-1)

            vloss = criterion(outputs, labels)
            running_vloss += vloss.item()

    avg_vloss = running_vloss / len(data_loader)
    return avg_vloss

def main():
    # Set environment variable for memory management
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

    # Load input data from JSON file
    with open('../data/encoded_data.json', 'r') as file:
        data = json.load(file)

    # Split data into training and validation sets
    train_data = data['train']
    validation_data = data['validation']

    # Create instances of MyDataset
    train_dataset = MyDataset(train_data)
    validation_dataset = MyDataset(validation_data)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Setup TensorBoard writer and device
    writer = SummaryWriter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model components
    num_layers = 4
    num_heads = 4
    d_model = 256
    d_ff = 1024
    encoder = Encoder(vocab_size=10000, n_layer=num_layers, n_head=num_heads, d_model=d_model, d_ff=d_ff)
    decoder = Decoder(vocab_size=10000, n_layer=num_layers, n_head=num_heads, d_model=d_model, d_ff=d_ff)
    model = EncoderDecoder(encoder, decoder, device).to(device)

    # Setup training essentials
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    scaler = GradScaler()

    # Training loop
    EPOCHS = 30
    checkpoints_dir = '../checkpoints'
    os.makedirs(checkpoints_dir, exist_ok=True)
    best_vloss = float('inf')

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch + 1}:')
        epoch_avg_loss = train_one_epoch(train_loader, epoch, writer, model, optimizer, criterion, scaler, device)
        print(f'Average Training Loss: {epoch_avg_loss:.2f}')

        avg_vloss = validate(validation_loader, model, criterion, device)
        print(f'Validation Loss: {avg_vloss:.2f}')

        writer.add_scalars('Training vs. Validation Loss', {'Training': epoch_avg_loss, 'Validation': avg_vloss}, epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_filename = os.path.join(checkpoints_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(model.state_dict(), model_filename)
            print(f'Model saved: {model_filename}')

        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()}")
        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    main()