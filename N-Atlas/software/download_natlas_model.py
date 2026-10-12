"""
N-ATLaS Model Download and Linker Utility
Model: NCAIR1/N-ATLaS (National Centre for AI and Robotics, Nigeria)
Hugging Face: https://huggingface.co/NCAIR1/N-ATLaS

This script provides utilities to:
1. Authenticate with Hugging Face (required because N-ATLaS is a Llama-3 derivative).
2. Test connectivity and cloud inference via Hugging Face Inference API (0 GB download).
3. Download the full model snapshot (~16 GB safetensor weights) locally to data/models/natlas.
4. Verify translation into Yoruba, Igbo, and Hausa.
"""
import os
import sys
import time
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "software"))

import config

MODEL_ID = "NCAIR1/N-ATLaS"
LOCAL_MODEL_DIR = config.MODELS_DIR / "natlas"

def check_hf_auth():
    """Verify Hugging Face authentication token."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    
    # Check default huggingface-cli login cache
    cache_token_path = Path.home() / ".cache" / "huggingface" / "token"
    if not token and cache_token_path.exists():
        try:
            token = cache_token_path.read_text().strip()
        except Exception:
            pass

    if token:
        print(f"[✓] Hugging Face token detected (starts with: {token[:6]}...)")
        return token
    else:
        print("[!] No Hugging Face token found in environment or cache.")
        print("    Because N-ATLaS is an 8B Llama-3 derivative, Hugging Face requires authentication.")
        print("    1. Visit: https://huggingface.co/NCAIR1/N-ATLaS and accept the model terms.")
        print("    2. Get your free token at: https://huggingface.co/settings/tokens")
        token_input = input("    Paste your Hugging Face Token (or press Enter to skip): ").strip()
        if token_input:
            os.environ["HF_TOKEN"] = token_input
            return token_input
        return None

def test_cloud_inference(token: str, english_text: str = "Hello, how are you doing?"):
    """
    Test N-ATLaS translation via Hugging Face Serverless Inference API.
    Does NOT require downloading 16 GB of weights.
    """
    print("\n" + "="*60)
    print(f"Testing Cloud Inference on {MODEL_ID} (Zero GB Download)...")
    print("="*60)
    
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient(model=MODEL_ID, token=token)
        
        languages = ["Yoruba", "Igbo", "Hausa"]
        for lang in languages:
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are N-ATLaS, an Afro-centric AI language assistant built by NCAIR Nigeria. "
                f"Translate the user's sentence into {lang}. Return ONLY the direct translation "
                f"with accurate tonal diacritics, without conversational filler.<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"Translate to {lang}: \"{english_text}\"<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            print(f"\n[Prompting N-ATLaS -> {lang}]...")
            t0 = time.time()
            try:
                response = client.text_generation(
                    prompt,
                    max_new_tokens=64,
                    temperature=0.1,
                    repetition_penalty=1.12,
                    stop=["<|eot_id|>"]
                )
                output = response.strip().replace("<|eot_id|>", "")
                dt = time.time() - t0
                print(f"[✓] {lang}: {output} (Time: {dt:.2f}s)")
            except Exception as e:
                print(f"[!] API call for {lang} returned: {e}")
                print("    (Note: If the serverless endpoint is cold or loading, try again in 1-2 minutes).")

    except ImportError:
        print("[!] huggingface_hub is not installed. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"[!] Error during inference test: {e}")

def download_model_weights(token: str):
    """
    Download full N-ATLaS model weights (~16 GB) locally to data/models/natlas.
    """
    print("\n" + "="*60)
    print(f"Downloading Full Model Weights: {MODEL_ID}")
    print(f"Destination: {LOCAL_MODEL_DIR}")
    print("="*60)
    print("Note: The model size is ~16 GB across 4 safetensor shards.")
    print("Ensure you have at least 20 GB of free disk space and a stable internet connection.")
    
    confirm = input("Proceed with 16 GB download? (y/n): ").strip().lower()
    if confirm != "y":
        print("[*] Download cancelled by user.")
        return False

    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
        print("\nStarting download from Hugging Face...")
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(LOCAL_MODEL_DIR),
            token=token,
            ignore_patterns=["*.msgpack", "*.h5"]
        )
        print(f"\n[✓] Successfully downloaded {MODEL_ID} to: {path}")
        return True
    except Exception as e:
        print(f"[!] Download failed: {e}")
        print("\nTroubleshooting:")
        print("1. Did you accept the license agreement at https://huggingface.co/NCAIR1/N-ATLaS?")
        print("2. Is your Hugging Face token valid and has 'read' permissions?")
        return False

def test_local_model():
    """
    Test loading local weights if already downloaded.
    """
    if not (LOCAL_MODEL_DIR / "config.json").exists():
        print(f"[!] Local model files not found in {LOCAL_MODEL_DIR}")
        print("    Run this script with --download to download them first.")
        return

    print("\n[Loading Local N-ATLaS Model]...")
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained(str(LOCAL_MODEL_DIR))
        print("[✓] Tokenizer loaded.")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        print(f"[*] Loading model on {device.upper()} (dtype: {dtype})...")
        
        model = AutoModelForCausalLM.from_pretrained(
            str(LOCAL_MODEL_DIR),
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        print("[✓] Model weights loaded successfully!")
        
        # Test generation
        prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "Translate to Yoruba: \"Hello, thank you!\"<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        outputs = model.generate(**inputs, max_new_tokens=32, temperature=0.1)
        res = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[✓] Output: {res}")
        
    except Exception as e:
        print(f"[!] Local loading error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="N-ATLaS Model Downloader & Linker")
    parser.add_argument("--test-cloud", action="store_true", help="Test N-ATLaS cloud inference via Hugging Face API")
    parser.add_argument("--download", action="store_true", help="Download full 16 GB weights to data/models/natlas")
    parser.add_argument("--test-local", action="store_true", help="Test local model weights from data/models/natlas")
    args = parser.parse_args()

    print("="*60)
    print("N-ATLaS MULTILINGUAL MODEL LINKER")
    print(f"Target: {MODEL_ID} (NCAIR / Awarri AI)")
    print("="*60)

    token = check_hf_auth()

    if args.download:
        download_model_weights(token)
    elif args.test_local:
        test_local_model()
    elif args.test_cloud:
        test_cloud_inference(token)
    else:
        print("\nChoose an option:")
        print("1. Test N-ATLaS Cloud Inference (Fast, 0 GB download, requires HF token)")
        print("2. Download Full 16 GB Model Weights locally to data/models/natlas")
        print("3. Test Local Model (if already downloaded)")
        print("4. Exit")
        choice = input("\nEnter choice [1-4]: ").strip()
        
        if choice == "1":
            test_cloud_inference(token)
        elif choice == "2":
            download_model_weights(token)
        elif choice == "3":
            test_local_model()
        else:
            print("Done.")
