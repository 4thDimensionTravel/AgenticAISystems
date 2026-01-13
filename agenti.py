import json

class Agent:
    def __init__(self, ime, sistemski_prompt, model_pipeline, tokenajzer, alati=None):

        self.ime = ime
        self.sistemski_prompt = sistemski_prompt
        self.llm = model_pipeline
        self.tokenajzer = tokenajzer
        self.alati = alati if alati else {}
        
        self.memorija = [
            {"role": "system", "content": sistemski_prompt}
        ]

    def dodaj_u_memoriju(self, uloga, sadrzaj):
        self.memorija.append({"role": uloga, "content": sadrzaj})
        #print(f"radi: {self.memorija{-1}}")
   
    def _pozovi_llm(self):
            prompt_text = self.tokenajzer.apply_chat_template(
                self.memorija, 
                tokenize=False,  
                add_generation_prompt=True 
            )
            prompt_text = prompt_text.replace("<|begin_of_text|>", "") 
            response = self.llm(prompt_text) 
            return response
    """ 
    def _pozovi_llm(self):
        prompt_text = ""
        for poruka in self.memorija:
            uloga = poruka['role']
            sadrzaj = poruka['content']
            
            if uloga == "user":
                prompt_text += f"[INST] {sadrzaj} [/INST]\n"
            elif uloga == "system":
                prompt_text += f"<<SYS>> {sadrzaj} <</SYS>>\n"
            else:
                prompt_text += f"{sadrzaj}\n"
        
        response = self.llm(prompt_text) 
        return response
     """ 

    def izvrsi_alat(self, ime_alata, argumenti):
        try:
            if ime_alata in self.alati:
                #print(f" Agent {self.ime} pokreće alat: {ime_alata} sa argumentima {argumenti}")
                rezultat = self.alati[ime_alata](argumenti) ##alati = { "alat_pretrazi_arxiv": alat_pretrazi_arxiv }
                                                                                ##kljuc         ##string
               
                if "pozovi_" in ime_alata:
                    return rezultat
               
                return f"Rezultat alata {ime_alata}: {rezultat}"
            else:
                return f"Greška: Alat {ime_alata} nije pronađen."
        except Exception as e:
            return f"Greška {ime_alata}: {str(e)}"

    def run(self, korisnicki_ulaz):
            
            self.dodaj_u_memoriju("user", korisnicki_ulaz)
            
            # Vrtimo petlju SVE DOK model traži alate
            while True:
                odgovor = self._pozovi_llm()
                
                #Ako model ne traži alat, završio je
                if "TOOL_CALL:" not in odgovor:
                    self.dodaj_u_memoriju("assistant", odgovor)
                    return odgovor

                #OBRADA ALATA
                try:
                    self.dodaj_u_memoriju("assistant", odgovor)
                    
                    # Parsiranje
                    deo_za_alat = odgovor.split("TOOL_CALL:")[1].strip()

                    if "|" in deo_za_alat:
                        ime_alata, argument = deo_za_alat.split("|", 1)
                    else:
                        ime_alata = deo_za_alat
                        argument = ""
                    
                    print(f"[SISTEM] {self.ime} pokreće: {ime_alata}...") 
                    rezultat_alata = self.izvrsi_alat(ime_alata.strip(), argument.strip()) 
                    
                    if "pozovi_" in ime_alata: 
                        return rezultat_alata

                    poruka_sa_rezultatom = f"Tool '{ime_alata}' output:\n{rezultat_alata}"
                    self.dodaj_u_memoriju("user", poruka_sa_rezultatom)
                    
                except Exception as e:
                    error_msg = f"System Error executing tool: {str(e)}"
                    self.dodaj_u_memoriju("user", error_msg)