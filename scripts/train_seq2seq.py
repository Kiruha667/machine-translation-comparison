"""
Training script for Seq2Seq model with Attention
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import math
import argparse

from src.config import *
from src.preprocessing import DataPreprocessor, print_data_statistics
from src.dataset import create_dataloaders
from src.models.seq2seq import create_seq2seq_model


def train_epoch(model, iterator, optimizer, criterion, clip, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(iterator, desc="Training")
    
    for src, tgt in progress_bar:
        src, tgt = src.to(device), tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt)
        
        # Reshape for loss calculation
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        tgt = tgt[:, 1:].reshape(-1)
        
        loss = criterion(output, tgt)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion, device):
    """Evaluate model"""
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(iterator, desc="Evaluating"):
            src, tgt = src.to(device), tgt.to(device)
            
            output = model(src, tgt, teacher_forcing_ratio=0)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)


def main(args):
    """Main training function"""
    # Print GPU info
    print_gpu_info()
    
    # Load and preprocess data
    print("\n" + "="*80)
    print("LOADING AND PREPROCESSING DATA")
    print("="*80)
    
    data_path = os.path.join(DATA_DIR, DATASET_FILE)
    
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        print("Please run: python data/download_data.py")
        return
    
    preprocessor = DataPreprocessor(max_length=MAX_LENGTH, min_freq=MIN_FREQ)
    pairs = preprocessor.load_and_preprocess(data_path)
    
    print_data_statistics(pairs)
    
    # Split data
    train_pairs, val_pairs, test_pairs = preprocessor.split_data(
        pairs, TRAIN_SPLIT, VAL_SPLIT
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_pairs, val_pairs, test_pairs,
        preprocessor.src_vocab, preprocessor.tgt_vocab, preprocessor,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    # Create model
    print("\n" + "="*80)
    print("CREATING SEQ2SEQ MODEL")
    print("="*80)
    
    src_vocab_size, tgt_vocab_size = preprocessor.get_vocab_sizes()
    
    model = create_seq2seq_model(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        embedding_dim=SEQ2SEQ_CONFIG['embedding_dim'],
        hidden_dim=SEQ2SEQ_CONFIG['hidden_dim'],
        dropout=SEQ2SEQ_CONFIG['dropout'],
        device=DEVICE
    )
    
    # Setup training
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Training loop
    print("\n" + "="*80)
    print("TRAINING SEQ2SEQ MODEL")
    print("="*80)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    total_train_time = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, 
                                CLIP_GRAD, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        
        epoch_time = time.time() - start_time
        total_train_time += epoch_time
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Print epoch results
        print(f"\nEpoch: {epoch+1:02}/{args.epochs}")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}")
        print(f"  Val Loss: {val_loss:.3f} | Val PPL: {math.exp(val_loss):7.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': SEQ2SEQ_CONFIG,
                'preprocessor': {
                    'src_vocab': preprocessor.src_vocab,
                    'tgt_vocab': preprocessor.tgt_vocab,
                    'src_idx2word': preprocessor.src_idx2word,
                    'tgt_idx2word': preprocessor.tgt_idx2word
                }
            }
            save_path = os.path.join(SAVE_DIR, 'seq2seq_best.pt')
            torch.save(checkpoint, save_path)
            print(f"  ✓ Model saved to {save_path}")
        
        # Print GPU memory usage
        if USE_CUDA:
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"  GPU Memory: {allocated:.2f} GB")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(f"Total training time: {total_train_time:.2f}s")
    print(f"Best validation loss: {best_val_loss:.3f}")
    print(f"Best validation PPL: {math.exp(best_val_loss):.3f}")
    
    # Save final model
    final_path = os.path.join(SAVE_DIR, 'seq2seq_final.pt')
    torch.save(checkpoint, final_path)
    print(f"\nFinal model saved to {final_path}")
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'total_time': total_train_time,
        'best_val_loss': best_val_loss
    }
    history_path = os.path.join(SAVE_DIR, 'seq2seq_history.pt')
    torch.save(history, history_path)
    print(f"Training history saved to {history_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Seq2Seq model')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                       help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE_SEQ2SEQ,
                       help=f'Learning rate (default: {LEARNING_RATE_SEQ2SEQ})')
    
    args = parser.parse_args()
    main(args)
