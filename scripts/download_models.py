
import os
import sys
from sentence_transformers import SentenceTransformer, CrossEncoder
import torch

def download_models():
    # downloads and caches the required models for the demo.
    # Run this inside the container before the demo to ensure no internet is needed later.
    # """
    print("="*60)
    print("PRE-DOWNLOADING MODELS FOR DEMO")
    print("="*60)

    # 1. Bi-Encoder: e5-base-v2
    # This is the base model we fine-tuned. 
    # Even if we load a fine-tuned version from disk later, it's good to have the base cached.
    model_name = "intfloat/e5-base-v2"
    print(f"\nDownloading Bi-Encoder base: {model_name}...")
    try:
        model = SentenceTransformer(model_name)
        print("✅ Bi-Encoder downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading Bi-Encoder: {e}")

    # 2. Cross-Encoder: ms-marco-MiniLM-L12-v2
    # We use this directly from HuggingFace cache
    cross_model_name = "cross-encoder/ms-marco-MiniLM-L12-v2"
    print(f"\nDownloading Cross-Encoder: {cross_model_name}...")
    try:
        model = CrossEncoder(cross_model_name)
        print("✅ Cross-Encoder downloaded successfully.")
    except Exception as e:
        print(f"❌ Error downloading Cross-Encoder: {e}")

    # 3. Check GPU
    print("\nChecking GPU availability...")
    if torch.cuda.is_available():
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ No GPU detected. Models will run on CPU (slower but functional).")

    print("\n" + "="*60)
    print("DOWNLOAD COMPLETE")
    print("Models are now cached in ~/.cache/huggingface (or container equivalent)")
    print("="*60)

if __name__ == "__main__":
    download_models()
