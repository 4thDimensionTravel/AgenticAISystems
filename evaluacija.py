import json
import ucitavanje
from agenti import Agent

ULAZNI_TEKST = """
Title: Federated Learning: Strategies for Improving Communication Efficiency.
Abstract: Federated Learning (FL) enables model training on decentralized data. However, communication costs remain a bottleneck. We propose a new algorithm, 'FedCom', which uses gradient compression and quantization to reduce data transfer by 40%. The method was tested on MNIST and CIFAR-10 datasets. Results show 99% accuracy retention compared to full-precision transmission. However, non-IID data distribution remains a challenge for convergence speed.
"""

PROMPT_NAIVE = """
Analyze the text. I need a JSON output with keys 'algo', 'analysis', 'code'.
Find the trade-off. Write a Python function for the algorithm.
Don't use the word 'problem'. Translate analysis to Serbian.
Make sure JSON is valid. The algorithm is FedCom.
"""

PROMPT_FINETUNED = """
[ROLE]
You are an expert AI Analyst specializing in Federated Learning.

[TASK]
Analyze the provided text and structure the output into a strict JSON format.

[CHAIN OF THOUGHT PROCESS]
1. First, identify the core algorithm and datasets.
2. Second, reason about the trade-off mentioned (Accuracy vs Speed).
3. Third, draft the Python signature.
4. Fourth, CHECK for forbidden words ("problem", "challenge"). If found, replace them.

[OUTPUT SPECIFICATION]
Return ONLY a JSON object:
{
  "algorithm_name": "String",
  "trade_off_analysis": "String (Serbian)",
  "python_signature": "String"
}

[NEGATIVE CONSTRAINTS]
- NO introductory text like "Here is the JSON".
- NO usage of word "problem".
- Do not invent code if not present.
"""

BROJ_ITERACIJA = 10  
generator = ucitavanje.ucitaj_model()
token = ucitavanje.ucitaj_tokenizer()

def moj_llm_poziv(prompt):
    return ucitavanje.llm_wrapper(generator, prompt)

def testiraj_prompt(naziv_testa, sistemski_prompt):
    uspesni = 0

    print(f"POKREĆEM TEST: {naziv_testa}")
    
    for i in range(BROJ_ITERACIJA):
        print(f"  Iteracija {i+1}/{BROJ_ITERACIJA}...", end="\r")
        

        test_agent = Agent(
            ime="TestAnaliticar",
            sistemski_prompt=sistemski_prompt,
            model_pipeline=moj_llm_poziv,
            tokenajzer=token,
            alati={} 
        )
        

        try:
            odgovor = test_agent.run(ULAZNI_TEKST)
        except Exception as e:
            print(f"\n  Greška u modelu: {e}")
            odgovor = ""

        is_json = False
        no_forbidden = True
        no_hallucination = True
        
        try:
            clean_odg = odgovor.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_odg)
            
            if "algorithm_name" in parsed or "algo" in parsed:
                is_json = True
            else:
                is_json = False 
        except:
            is_json = False
            
        if "problem" in odgovor.lower():
            no_forbidden = False
            
        if "import torch" in odgovor or "class FedCom" in odgovor: ###Ako model napiše "import torch", 
            ##on je uveo informaciju koja ne postoji u ulazu.
            no_hallucination = False

        if is_json and no_forbidden and no_hallucination:
            uspesni += 1
        else:

            pass

    stopa_uspeha = (uspesni / BROJ_ITERACIJA) * 100
    print(f"\nREZULTAT {naziv_testa}: {stopa_uspeha:.1f}% Uspešnosti")
    return stopa_uspeha

if __name__ == "__main__":
    rez_naive = testiraj_prompt("L3 NAIVE (Loš Prompt)", PROMPT_NAIVE)
    
    rez_finetuned = testiraj_prompt("L3 FINE-TUNED (Dobar Prompt)", PROMPT_FINETUNED)
    
    print("\n\n")
    print(f"1. L3 NAIVE: {rez_naive}%")
    print(f"2. L3 FINE-TUNED: {rez_finetuned}%")

    
