from src.vanilla_rag import vanilla_csv
from src.agentic_rag import agentic_csv
from src.compute_top import top_by_surgery_90th

CSV_PATH = "data/Surgical_Wait_Times_20260207.csv"


def main():
    print("NS Surgical Wait Times RAG")
    print("Type a question. Type 'exit' to quit.\n")

    while True:
        q = input("Your question> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        q_low = q.lower()

        # If question asks for max/highest/top, show exact computed results first
        if any(w in q_low for w in ["highest", "max", "maximum", "top"]):
            print("\n=== COMPUTED TOP 10 by Surgery_90th (exact) ===\n")
            print(top_by_surgery_90th(CSV_PATH, n=10))
            print("\n(Then RAG explanation below)\n")

        print("\n" + "=" * 60)
        print("VANILLA RAG\n")
        v = vanilla_csv(q, CSV_PATH)
        print("\nVANILLA ANSWER:\n", v)

        print("\n" + "-" * 60)
        print("AGENTIC RAG\n")
        a = agentic_csv(q, CSV_PATH)
        print("\nAGENTIC ANSWER:\n", a)
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
