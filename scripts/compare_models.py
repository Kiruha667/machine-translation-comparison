"""
Script to compare Seq2Seq and Transformer models
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import sacrebleu

from src.config import *
from src.models.seq2seq import Seq2Seq, Encoder, Decoder
from src.models.transformer import TransformerModel
from src.preprocessing import DataPreprocessor


def load_model_and_preprocessor(model_path, model_type='seq2seq'):
    """Load trained model and preprocessor"""
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    # Load preprocessor
    preprocessor = DataPreprocessor()
    preprocessor.src_vocab = checkpoint['preprocessor']['src_vocab']
    preprocessor.tgt_vocab = checkpoint['preprocessor']['tgt_vocab']
    preprocessor.src_idx2word = checkpoint['preprocessor']['src_idx2word']
    preprocessor.tgt_idx2word = checkpoint['preprocessor']['tgt_idx2word']
    
    src_vocab_size = len(preprocessor.src_vocab)
    tgt_vocab_size = len(preprocessor.tgt_vocab)
    
    # Create model
    if model_type == 'seq2seq':
        encoder = Encoder(src_vocab_size, 256, 512, 0.1)
        decoder = Decoder(tgt_vocab_size, 256, 512, 0.1)
        model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    else:
        model = TransformerModel(src_vocab_size, tgt_vocab_size).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, preprocessor, checkpoint


def translate_seq2seq(sentence, model, preprocessor, max_len=20):
    """Translate using Seq2Seq model"""
    model.eval()
    tokens = preprocessor.tokenize(preprocessor.clean_text(sentence))
    indices = preprocessor.sentence_to_indices(tokens, preprocessor.src_vocab)
    src_tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)
        
        input_token = torch.tensor([preprocessor.tgt_vocab['<sos>']]).to(DEVICE)
        outputs = [input_token.item()]
        
        for _ in range(max_len):
            output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs)
            top1 = output.argmax(1)
            outputs.append(top1.item())
            
            if top1.item() == preprocessor.tgt_vocab['<eos>']:
                break
            
            input_token = top1
    
    words = [preprocessor.tgt_idx2word.get(idx, '<unk>') for idx in outputs]
    words = [w for w in words if w not in ['<sos>', '<eos>', '<pad>']]
    return ' '.join(words)


def translate_transformer(sentence, model, preprocessor, max_len=20):
    """Translate using Transformer model"""
    model.eval()
    tokens = preprocessor.tokenize(preprocessor.clean_text(sentence))
    indices = preprocessor.sentence_to_indices(tokens, preprocessor.src_vocab)
    src_tensor = torch.tensor(indices).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        tgt_indices = [preprocessor.tgt_vocab['<sos>']]
        
        for _ in range(max_len):
            tgt_tensor = torch.tensor(tgt_indices).unsqueeze(0).to(DEVICE)
            output = model(src_tensor, tgt_tensor)
            
            next_token = output[0, -1].argmax().item()
            tgt_indices.append(next_token)
            
            if next_token == preprocessor.tgt_vocab['<eos>']:
                break
    
    words = [preprocessor.tgt_idx2word.get(idx, '<unk>') for idx in tgt_indices]
    words = [w for w in words if w not in ['<sos>', '<eos>', '<pad>']]
    return ' '.join(words)


def calculate_bleu(model, test_pairs, model_type, preprocessor, num_samples=500):
    """Calculate BLEU score"""
    references = []
    hypotheses = []
    
    samples = min(num_samples, len(test_pairs))
    
    for i in tqdm(range(samples), desc=f"Calculating BLEU ({model_type})"):
        src_tokens, tgt_tokens = test_pairs[i]
        src_sentence = ' '.join(src_tokens)
        tgt_sentence = ' '.join(tgt_tokens)
        
        if model_type == 'seq2seq':
            pred_sentence = translate_seq2seq(src_sentence, model, preprocessor)
        else:
            pred_sentence = translate_transformer(src_sentence, model, preprocessor)
        
        references.append([tgt_sentence])
        hypotheses.append(pred_sentence)
    
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    return bleu.score


def plot_training_curves(seq2seq_history, transformer_history):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Seq2Seq
    epochs = range(1, len(seq2seq_history['train_losses']) + 1)
    axes[0].plot(epochs, seq2seq_history['train_losses'], 'o-', label='Train')
    axes[0].plot(epochs, seq2seq_history['val_losses'], 's-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Seq2Seq Training Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Transformer
    epochs = range(1, len(transformer_history['train_losses']) + 1)
    axes[1].plot(epochs, transformer_history['train_losses'], 'o-', label='Train')
    axes[1].plot(epochs, transformer_history['val_losses'], 's-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Transformer Training Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=PLOT_DPI)
    print(f"✓ Training curves saved to {OUTPUT_DIR}/training_curves.png")
    plt.show()


def plot_comparison(seq2seq_bleu, transformer_bleu, seq2seq_time, transformer_time):
    """Plot comparison bar charts"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # BLEU scores
    models = ['Seq2Seq', 'Transformer']
    bleu_scores = [seq2seq_bleu, transformer_bleu]
    colors = ['#3498db', '#e74c3c']
    
    bars1 = axes[0].bar(models, bleu_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[0].set_ylabel('BLEU Score', fontsize=12)
    axes[0].set_title('BLEU Score Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, max(bleu_scores) * 1.2])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar, score in zip(bars1, bleu_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{score:.2f}', ha='center', fontweight='bold', fontsize=11)
    
    # Training time
    times = [seq2seq_time, transformer_time]
    bars2 = axes[1].bar(models, times, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_ylabel('Training Time (seconds)', fontsize=12)
    axes[1].set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar, time in zip(bars2, times):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{time:.1f}s', ha='center', fontweight='bold', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_comparison.png'), dpi=PLOT_DPI)
    print(f"✓ Comparison chart saved to {OUTPUT_DIR}/model_comparison.png")
    plt.show()


def main():
    """Main comparison function"""
    print("="*80)
    print("MODEL COMPARISON")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    seq2seq_path = os.path.join(SAVE_DIR, 'seq2seq_best.pt')
    transformer_path = os.path.join(SAVE_DIR, 'transformer_best.pt')
    
    if not os.path.exists(seq2seq_path):
        print(f"Error: Seq2Seq model not found at {seq2seq_path}")
        print("Please train the model first: python scripts/train_seq2seq.py")
        return
    
    if not os.path.exists(transformer_path):
        print(f"Error: Transformer model not found at {transformer_path}")
        print("Please train the model first: python scripts/train_transformer.py")
        return
    
    seq2seq_model, seq2seq_preprocessor, seq2seq_checkpoint = load_model_and_preprocessor(
        seq2seq_path, 'seq2seq'
    )
    transformer_model, transformer_preprocessor, transformer_checkpoint = load_model_and_preprocessor(
        transformer_path, 'transformer'
    )
    
    print("✓ Models loaded successfully")
    
    # Load training histories
    seq2seq_history = torch.load(os.path.join(SAVE_DIR, 'seq2seq_history.pt'))
    transformer_history = torch.load(os.path.join(SAVE_DIR, 'transformer_history.pt'))
    
    # Load test data
    print("\nLoading test data...")
    data_path = os.path.join(DATA_DIR, DATASET_FILE)
    preprocessor = DataPreprocessor(max_length=MAX_LENGTH)
    pairs = preprocessor.load_and_preprocess(data_path)
    _, _, test_pairs = preprocessor.split_data(pairs, TRAIN_SPLIT, VAL_SPLIT)
    
    # Sample translations
    print("\n" + "="*80)
    print("SAMPLE TRANSLATIONS")
    print("="*80)
    
    sample_sentences = [
        "i love you",
        "good morning",
        "how are you",
        "thank you",
        "see you tomorrow",
        "i am learning french",
        "the weather is nice today",
        "where is the station",
        "i dont understand",
        "can you help me"
    ]
    
    for sent in sample_sentences:
        seq2seq_trans = translate_seq2seq(sent, seq2seq_model, seq2seq_preprocessor)
        transformer_trans = translate_transformer(sent, transformer_model, transformer_preprocessor)
        
        print(f"\n{'Source:':<15} {sent}")
        print(f"{'Seq2Seq:':<15} {seq2seq_trans}")
        print(f"{'Transformer:':<15} {transformer_trans}")
    
    # Calculate BLEU scores
    print("\n" + "="*80)
    print("CALCULATING BLEU SCORES")
    print("="*80)
    
    seq2seq_bleu = calculate_bleu(seq2seq_model, test_pairs, 'seq2seq', 
                                  seq2seq_preprocessor, BLEU_SAMPLES)
    transformer_bleu = calculate_bleu(transformer_model, test_pairs, 'transformer',
                                     transformer_preprocessor, BLEU_SAMPLES)
    
    # Print results
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    
    print(f"\n{'Metric':<30} {'Seq2Seq':<20} {'Transformer':<20}")
    print("-" * 70)
    print(f"{'BLEU Score':<30} {seq2seq_bleu:<20.2f} {transformer_bleu:<20.2f}")
    print(f"{'Training Time (s)':<30} {seq2seq_history['total_time']:<20.2f} {transformer_history['total_time']:<20.2f}")
    print(f"{'Parameters':<30} {seq2seq_model.count_parameters():>19,} {transformer_model.count_parameters():>19,}")
    print(f"{'Best Val Loss':<30} {seq2seq_history['best_val_loss']:<20.3f} {transformer_history['best_val_loss']:<20.3f}")
    
    # Plot results
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_training_curves(seq2seq_history, transformer_history)
    plot_comparison(seq2seq_bleu, transformer_bleu,
                   seq2seq_history['total_time'], transformer_history['total_time'])
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETED")
    print("="*80)
    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Check the plots for detailed analysis!")


if __name__ == '__main__':
    main()
