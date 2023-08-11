'''
DEF CON 31 AI Village - LLMs: Loose Lips Multipliers
Kyle Easterly & Mitch Kitter
https://www.purpleaerospace.com
download-base-model.py: Download the openllama-7b base model
'''

from huggingface_hub import snapshot_download

print("Downloading openllama-7b...")
snapshot_download(repo_id = "openlm-research/open_llama_7b", repo_type = 'model')
