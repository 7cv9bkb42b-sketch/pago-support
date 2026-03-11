#!/usr/bin/env python3
import subprocess, sys, json, requests

# Auto-upgrade torch if < 2.1
try:
    import torch
    if int(torch.__version__.split(".")[0]) < 2:
        print("Upgrading PyTorch (~2 min one-time)...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch>=2.1", "--quiet"])
        import os; os.execv(sys.executable, [sys.executable] + sys.argv)
except ImportError:
    pass

from sentence_transformers import SentenceTransformer

QDRANT_URL = "https://aa2561d7-8ff7-4ed5-8e74-43a99dc55cf3.us-east-1-1.aws.cloud.qdrant.io"
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.fVOH150mVWNja10D_Ht2WhjotPd8dtnMaB3F7hWc1hU"
COLLECTION = "pago"

# Load pairs from pairs.json (download from Claude outputs)
with open("pairs.json") as f:
    PAIRS = json.load(f)

print(f"Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print(f"Embedding {len(PAIRS)} pairs...")
vectors = model.encode([p["c"][:512] for p in PAIRS], batch_size=64, show_progress_bar=True).tolist()

print("Uploading to Qdrant...")
points = [{"id": i+1, "vector": v, "payload": {"c": p["c"][:400], "a": p["a"][:350]}}
          for i, (p, v) in enumerate(zip(PAIRS, vectors))]

for i in range(0, len(points), 100):
    batch = points[i:i+100]
    r = requests.put(f"{QDRANT_URL}/collections/{COLLECTION}/points",
        headers={"api-key": API_KEY, "Content-Type": "application/json"},
        json={"points": batch}, timeout=30)
    print(f"  Batch {i//100+1}: {r.status_code} ({min(i+100,len(points))}/{len(points)})")

print(f"Done! {len(points)} points uploaded.")
