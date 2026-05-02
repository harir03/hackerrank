"""Quick verification of the retriever against the provided corpus."""
import sys
sys.path.insert(0, "code")

from pipeline.retrieve import retrieve_chunks, compute_coverage_score

print("=" * 60)
print("TEST 1: HackerRank - test timer froze")
print("=" * 60)
chunks = retrieve_chunks("test timer froze mid-assessment", "hackerrank", top_k=3)
print(f"HackerRank chunks returned: {len(chunks)}")
for c in chunks:
    print(f"  [{c['domain']}] {c['section']} | score={c['score']:.2f}")
    print(f"  {c['text'][:120]}")
    print()

print("=" * 60)
print("TEST 2: Claude - conversation history disappeared")
print("=" * 60)
chunks2 = retrieve_chunks("conversation history disappeared", "claude", top_k=3)
print(f"Claude chunks returned: {len(chunks2)}")
for c in chunks2:
    print(f"  [{c['domain']}] {c['section']} | score={c['score']:.2f}")
    print(f"  {c['text'][:120]}")
    print()

print("=" * 60)
print("TEST 3: Visa - lost stolen card")
print("=" * 60)
chunks3 = retrieve_chunks("lost stolen visa card", "visa", top_k=3)
print(f"Visa chunks returned: {len(chunks3)}")
for c in chunks3:
    print(f"  [{c['domain']}] {c['section']} | score={c['score']:.2f}")
    print(f"  {c['text'][:120]}")
    print()

print("=" * 60)
print("COVERAGE SCORES")
print("=" * 60)
print(f"  HackerRank: {compute_coverage_score(chunks):.3f}")
print(f"  Claude:     {compute_coverage_score(chunks2):.3f}")
print(f"  Visa:       {compute_coverage_score(chunks3):.3f}")
