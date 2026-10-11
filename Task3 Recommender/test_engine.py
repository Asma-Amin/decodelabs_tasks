"""Quick checks for the recommender. Run: python test_engine.py"""
import numpy as np
from pathlib import Path
from recommender import Recommender, load_catalog

df = load_catalog(Path(__file__).parent / "data" / "catalog.csv")
e = Recommender(df)

# 1. every title vector is unit length -> dot product is a true cosine
assert np.allclose(np.linalg.norm(e.X, axis=1), 1.0)

# 2. scores stay in [0, 1] and results come back sorted
p = e.build_profile(["Romance"], ["Emotional"], {"Your Name": 5})
res = e.recommend(p, n=10)
scores = [s for _, s in res]
assert all(0 <= s <= 1.0001 for s in scores) and scores == sorted(scores, reverse=True)

# 3. excluded titles and filtered types never appear
res = e.recommend(p, n=20, exclude=["Weathering With You"], types=["Anime"])
names = [df.title[i] for i, _ in res]
assert "Weathering With You" not in names and all(df.type[i] == "Anime" for i, _ in res)

# 4. a low rating pushes a title's neighbours down
liked = e.build_profile(ratings={"Death Note": 5})
disliked = e.build_profile(ratings={"Death Note": 1})
i = e.title_idx["monster"]
assert e.score(i, liked) > e.score(i, disliked)

# 5. dislikes alone never create a profile, and their pull is bounded
assert not e.build_profile(ratings={"Death Note": 1, "Monster": 2}).any()
base = e.build_profile(["Mystery", "Thriller"])
many = e.build_profile(["Mystery", "Thriller"], ratings={t: 1 for t in df.title[:30]})
assert many.any() and np.linalg.norm(many - base) <= 0.41

# 6. recommendations never show a negative or near-zero match
for ratings in ({}, {t: 1 for t in df.title[:30]}):
    prof = e.build_profile(["Romance"], ["Emotional"], ratings)
    assert all(s_ > 0.05 for _, s_ in e.recommend(prof, n=50))

# 7. quality check: share of top-5 similar titles that share a genre with the seed
hits = total = 0
for i in range(len(df)):
    seed = {t for t in e.tags[i] if t.startswith("g:")}
    for j, _ in e.similar(df.title[i], n=5):
        total += 1
        hits += bool(seed & {t for t in e.tags[j] if t.startswith("g:")})
print(f"All checks passed. Top-5 similar titles share a genre with the seed: {hits / total:.0%}")