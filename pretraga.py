import arxiv

def alat_pretrazi_arxiv(tema):

    print(f"DEBUG: Pokrećem Arxiv pretragu za: {tema}...")
    try:
        # Pretraga
        search = arxiv.Search(
            query=tema,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )

        rezultati = []
        for result in search.results():
            info = f"NASLOV: {result.title}\nSAŽETAK: {result.summary}\nID: {result.entry_id}\n---"
            rezultati.append(info)
        
        return "\n".join(rezultati)
    except Exception as e:
        return f"Greška prilikom pretrage: {str(e)}"