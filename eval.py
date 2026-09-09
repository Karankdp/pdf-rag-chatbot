"""
Manual evaluation harness for the RAG pipeline.
Run this against your own document + a hand-written Q&A set to sanity-check
retrieval and generation quality before/after tuning chunk size, top_k, etc.

Usage: adjust `eval_set` below to match facts you know are in your PDF,
then run this after your pipeline (retrieve/generate_answer) is set up.
"""

def evaluate_manual(eval_set, retrieve_fn, generate_fn, collection, top_k=4):
    results = []
    for item in eval_set:
        retrieved = retrieve_fn(collection, item["question"], top_k=top_k)
        answer = generate_fn(item["question"], retrieved)

        answer_lower = answer.lower()
        matched = [kw for kw in item["expected_keywords"] if kw.lower() in answer_lower]
        hit_rate = len(matched) / len(item["expected_keywords"])

        results.append({
            "question": item["question"],
            "answer": answer,
            "matched_keywords": matched,
            "hit_rate": hit_rate,
        })
    return results


# Example eval set — replace with questions/keywords specific to your own document(s)
eval_set = [
    {
        "question": "What is the main goal of this document?",
        "expected_keywords": ["goal", "purpose"],  # replace with real expected terms
    },
    # Add 5-8 more based on facts you know are in your PDF
]

if __name__ == "__main__":
    print("Import this module's evaluate_manual() into your pipeline script,")
    print("passing your retrieve/generate functions and a live collection.")
