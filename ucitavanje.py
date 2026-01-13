from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
import os

def ucitaj_model():
    print(" Učitavam Llama-3.2")
    
    model_path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    )
    
    print(f" path: {model_path}")

    llm = Llama(
        model_path=model_path,
        n_ctx=32000, ##ogranicavanje kratkorocne memorije
        n_gpu_layers=-1,  ##koristimo graficku maksimalno koliko mozemo
        verbose=False ## ispis lagova nam ne treba pa stavljamo false
    )
    
    return llm

def ucitaj_tokenizer():
    print("Učitavam Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-3B-Instruct")
    return tokenizer


def llm_wrapper(llm, prompt):
    output = llm(
        prompt,
        max_tokens=1000, 
        stop=["<|eot_id|>", "<|start_header_id|>"], 
        echo=False 
    )
    
    return output['choices'][0]['text'].strip()