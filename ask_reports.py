"""
ask_reports.py — Gecmis raporlar arasinda anlamsal (RAG) arama yap.

Kullanim:
    python ask_reports.py "yuksek iade orani"
    python ask_reports.py "model dogrulugu dusuk cikan donemler"
"""
import sys
import rag_utils

DUCKDB_PATH = "data/warehouse.duckdb"

def main():
    if len(sys.argv) < 2:
        print('Kullanim: python ask_reports.py "aranacak konu"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    print(f"\nAraniyor: \"{query}\"\n")

    results = rag_utils.search_similar_reports(DUCKDB_PATH, query, top_k=3)

    if not results:
        print("Henuz aranabilir gecmis rapor yok. Once main_pipeline.py'yi birkac kez calistir.")
        return

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['report_id']}  (benzerlik: {r['similarity']})")
        print(f"   {r['preview']}...\n")

if __name__ == "__main__":
    main()