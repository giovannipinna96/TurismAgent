# from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
# import argparse
# import os

# def download_model(model_name: str, save_dir: str, include_tokenizer=True, include_processor=True):
#     os.makedirs(save_dir, exist_ok=True)

#     # Scarica e salva il modello
#     print(f"🔄 Downloading model: {model_name}")
#     try:
#         model = AutoModel.from_pretrained(model_name)
#     except Exception as e:
#         print(f"⚠️  Failed to download model trying AutoModelForCausalLM")
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#     model.save_pretrained(save_dir)
#     print(f"✅ Model saved to: {save_dir}")

#     # Scarica e salva tokenizer (se esiste)
#     if include_tokenizer:
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(model_name)
#             tokenizer.save_pretrained(save_dir)
#             print(f"✅ Tokenizer saved to: {save_dir}")
#         except Exception as e:
#             print(f"⚠️  No tokenizer found: {e}")

#     # Scarica e salva processor (se esiste)
#     if include_processor:
#         try:
#             processor = AutoProcessor.from_pretrained(model_name)
#             processor.save_pretrained(save_dir)
#             print(f"✅ Processor saved to: {save_dir}")
#         except Exception as e:
#             print(f"⚠️  No processor found: {e}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Download and save a Hugging Face model locally.")
#     parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
#     parser.add_argument("--save_dir", type=str, required=True, help="Directory where to save the model")
#     args = parser.parse_args()

#     download_model(args.model_name, args.save_dir)


from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import argparse
import os

def download_model(model_name: str, save_dir: str, include_tokenizer=True, include_processor=True, only_download=False):
    os.makedirs(save_dir, exist_ok=True)

    if only_download:
        print(f"📦 Using snapshot_download to download model files (no RAM load)")
        snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Files downloaded to: {save_dir}")
        return

    # Scarica e salva il modello (con caricamento in RAM)
    print(f"🔄 Downloading and loading model: {model_name}")
    try:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️  Failed to load with AutoModel, trying AutoModelForCausalLM...")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    model.save_pretrained(save_dir)
    print(f"✅ Model saved to: {save_dir}")

    # Scarica e salva tokenizer (se richiesto)
    if include_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trist_remote_code=True)
            tokenizer.save_pretrained(save_dir)
            print(f"✅ Tokenizer saved to: {save_dir}")
        except Exception as e:
            print(f"⚠️  No tokenizer found: {e}")

    # Scarica e salva processor (se richiesto)
    if include_processor:
        try:
            processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            processor.save_pretrained(save_dir)
            print(f"✅ Processor saved to: {save_dir}")
        except Exception as e:
            print(f"⚠️  No processor found: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and save a Hugging Face model locally.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model on Hugging Face")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory where to save the model")
    parser.add_argument("--only_download", action="store_true", help="Only download model files without loading into memory")
    args = parser.parse_args()

    download_model(
        model_name=args.model_name,
        save_dir=args.save_dir,
        only_download=args.only_download
    )
