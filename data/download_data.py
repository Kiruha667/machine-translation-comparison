"""
Script to download and prepare the English-French dataset
"""

import os
import urllib.request
import zipfile
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file from URL with progress bar"""
    # Add headers to avoid 406 error
    opener = urllib.request.build_opener()
    opener.addheaders = [
        ('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    ]
    urllib.request.install_opener(opener)
    
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def main():
    """Main download function"""
    # Configuration
    data_dir = os.path.dirname(os.path.abspath(__file__))
    url = 'https://www.manythings.org/anki/fra-eng.zip'
    zip_path = os.path.join(data_dir, 'fra-eng.zip')
    extract_path = data_dir
    
    print("="*80)
    print("DOWNLOADING ENGLISH-FRENCH DATASET")
    print("="*80)
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset
    if not os.path.exists(zip_path):
        print(f"\nDownloading dataset from {url}...")
        try:
            download_url(url, zip_path)
            print("✓ Download completed!")
        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            return
    else:
        print(f"\n✓ Dataset already downloaded: {zip_path}")
    
    # Extract dataset
    fra_txt_path = os.path.join(extract_path, 'fra.txt')
    if not os.path.exists(fra_txt_path):
        print("\nExtracting dataset...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("✓ Extraction completed!")
        except Exception as e:
            print(f"✗ Error extracting dataset: {e}")
            return
    else:
        print(f"\n✓ Dataset already extracted: {fra_txt_path}")
    
    # Verify file
    if os.path.exists(fra_txt_path):
        # Count lines
        with open(fra_txt_path, 'r', encoding='utf-8') as f:
            num_lines = sum(1 for _ in f)
        
        file_size = os.path.getsize(fra_txt_path) / (1024 * 1024)  # MB
        
        print("\n" + "="*80)
        print("DATASET INFORMATION")
        print("="*80)
        print(f"File: {fra_txt_path}")
        print(f"Size: {file_size:.2f} MB")
        print(f"Number of sentence pairs: {num_lines:,}")
        print("="*80)
        
        # Show sample lines
        print("\nSample sentence pairs:")
        print("-"*80)
        with open(fra_txt_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    print(f"{i+1}. EN: {parts[0]}")
                    print(f"   FR: {parts[1]}")
                    print()
        print("-"*80)
        
        print("\n✓ Dataset ready for use!")
        print(f"\nYou can now run:")
        print(f"  python scripts/train_seq2seq.py")
        print(f"  python scripts/train_transformer.py")
        
    else:
        print("\n✗ Error: Dataset file not found after extraction")
        print("Please check your internet connection and try again")


if __name__ == '__main__':
    main()
