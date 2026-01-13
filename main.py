from agenti import Agent
from pretraga import alat_pretrazi_arxiv
import ucitavanje

prompt_istrazivac = """
You are the RESEARCHER AGENT.
Your goal is to find scientific papers on Arxiv.

RULES:
1. You must convert the user's request into specific ENGLISH keywords (Arxiv is in English).
2. You must use the tool 'alat_pretrazi_arxiv'.
3. DO NOT answer the user directly.
4. DO NOT hallucinate or make up paper titles.

WORKFLOW:
1. User asks a question.
2. You generate: TOOL_CALL: alat_pretrazi_arxiv|english keywords
3. System gives you results.
4. You IMMEDIATELY pass those results to the Analyst using:
   TOOL_CALL: alat_pozovi_analiticara|raw_text_from_arxiv

Example:
User: "Nadji mi nesto o AI"
You: TOOL_CALL: alat_pretrazi_arxiv|Artificial Intelligence agents
"""

prompt_analiticar = """
You are the ANALYST AGENT.
Your goal is to extract key information from raw data.

RULES:
1. You will receive raw text from the Researcher.
2. Summarize it briefly (Key findings, Methods).
3. DO NOT answer the user directly.

WORKFLOW:
1. Receive raw text.
2. Analyze it.
3. Pass the analysis to the Writer using:
   TOOL_CALL: alat_pozovi_pisca|your_analysis_summary
"""

prompt_pisac = """
You are the WRITER AGENT.
You are the ONLY agent allowed to speak to the user.

RULES:
1. You will receive an analysis from the Analyst.
2. Based on that analysis, write a helpful, professional response.


Your output should be the final answer to the user.
"""

generator = ucitavanje.ucitaj_model()
token = ucitavanje.ucitaj_tokenizer()

def moj_llm_poziv(prompt):
    return ucitavanje.llm_wrapper(generator, prompt)

def pozovi_analiticara(sirovi_tekst): #zbog prompta gore ne generise odgovor nego generise tool call za f-ju koji ima alat_pretrazi
    print(f"\n[SISTEM] Istraživač predaje podatke Analitičaru...\n")
    odgovor_analiticara = agent_analiticar.run(f"Evo sirovih podataka: {sirovi_tekst}")
    return odgovor_analiticara

def pozovi_pisca(analiza):
    print(f"\n[SISTEM] Analitičar predaje analizu Piscu...\n")
    konacni_tekst = agent_pisac.run(f"Evo analize: {analiza}")
    return konacni_tekst


agent_istrazivac = Agent(
    ime="Istrazivac",
    sistemski_prompt=prompt_istrazivac,
    model_pipeline=moj_llm_poziv,
    tokenajzer=token,
    alati={"alat_pretrazi_arxiv": alat_pretrazi_arxiv, "alat_pozovi_analiticara": pozovi_analiticara} # Dajemo mu alat
)

agent_analiticar = Agent(
    ime="Analiticar",
    sistemski_prompt=prompt_analiticar,
    model_pipeline=moj_llm_poziv,
    tokenajzer=token,
    alati={"alat_pozovi_pisca": pozovi_pisca}
)

agent_pisac = Agent(
    ime="Pisac",
    sistemski_prompt=prompt_pisac,
    model_pipeline=moj_llm_poziv,
    tokenajzer=token,
    alati={}
)


def pokreni_sistem(korisnicki_upit):
    print(f"\n--- KORISNIK: {korisnicki_upit} ---\n")
    rez = agent_istrazivac.run(korisnicki_upit)
    return rez

if __name__ == "__main__":
    pitanje = """
                 microbes in deep ocean
            """
    odgovor = pokreni_sistem(pitanje)
    print("\n================ KONACAN ODGOVOR ================\n")
    print(odgovor)