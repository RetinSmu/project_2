from src.vanilla_rag import vanilla_csv
from src.agentic_rag import agentic_csv

CSV_PATH = "data/Surgical_Wait_Times_20260207.csv"

QUESTIONS = [
    "Which procedures have the longest surgery wait times (Surgery_90th)?",
    "Compare surgery wait times by zone for Gastrointestinal Tract Surgery.",
    "Which facilities show high consult wait times (Consult_90th)?",
    "For Orthopaedic, which procedures look worst for Surgery_Median?",
    "Explain what columns mean in this dataset.",
]

if __name__ == "__main__":
    for q in QUESTIONS:
        print("\n" + "=" * 90)
        print("QUESTION:", q)

        print("\n--- VANILLA ---")
        v = vanilla_csv(q, CSV_PATH)
        print("\nVANILLA ANSWER:\n", v)

        print("\n--- AGENTIC ---")
        a = agentic_csv(q, CSV_PATH)
        print("\nAGENTIC ANSWER:\n", a)
