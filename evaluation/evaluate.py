import argparse
import json
import time
from pathlib import Path
from llm import generate_answer
from rag import retrieve

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["base", "rag"], default="rag")
    parser.add_argument("--questions", default="evaluation/questions.jsonl")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    with open(args.questions, encoding="utf-8") as f:
        questions = [json.loads(line) for line in f if line.strip()]
    results = []
    for item in questions:
        q = item["question"]
        start = time.perf_counter()
        sources = retrieve(q) if args.mode == "rag" else []
        answer = generate_answer(q, retrieved=sources)
        elapsed = time.perf_counter() - start
        results.append({"mode": args.mode, "question": q,
            "expected_topic": item.get("expected_topic", ""), "answer": answer,
            "sources": [s["source"] for s in sources], "latency_seconds": round(elapsed, 3)})
        print(f"\nQ: {q}\nA: {answer}\nLatency: {elapsed:.2f}s")
    out = args.output or f"evaluation/results_{args.mode}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out}")

if __name__ == "__main__":
    main()
