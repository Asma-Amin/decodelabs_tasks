# Watch Next: AI Recommendation Logic (DecodeLabs AI Project 3)

A content-based recommender for movies, anime and dramas, built with Python and Streamlit.
It matches a user's preferences to items using cosine similarity.

## Project requirements
| Requirement | Where it is met |
|---|---|
| Take user input (choices or interests) | Genre and mood pickers, 1-5 title ratings, "Find similar" title search |
| Match preferences using logic or similarity | TF-IDF-weighted tag vectors + cosine similarity (`recommender.py`) |
| Display recommended items | Ranked cards with match %, tags and a "Why" explanation |
| Bonus: similarity score | Match % shown on every card, with a "Why" line listing the matching tags |
| Bonus: let users rate preferences | 1-5 ratings; "Not interested" also teaches the profile |

## Run
```
pip install -r requirements.txt
streamlit run app.py
```
Run the checks with `python test_engine.py`.

## How it works
1. Each title is a vector over tags (genres, moods, type). Rare tags get more weight (IDF).
2. The user's taste (chosen genres/moods + ratings) is a vector too. High ratings pull toward a title's tags; low ratings and "Not interested" give a small, bounded push away. Dislikes alone never create a profile.
3. `cos(A, B) = A . B / (|A| |B|)` ranks every title against the profile.
4. Weak matches (score of 5% or less) are hidden, so no recommendation ever shows a negative or near-zero match.
5. The "Why" line lists the tags that contributed most to each match.

## Features
Genre/mood input, ratings, explain-why, find-similar, cross-media mode, mixed-type results,
type/year/length filters, surprise pick, not-interested feedback, saved list with CSV export,
"What drives your matches" chart, optional Kaggle CSV upload (Netflix titles or anime.csv).

## Files
- `app.py`: Streamlit interface
- `recommender.py`: vectors, cosine similarity, explanations
- `data/catalog.csv`: 126 hand-curated titles (movies, anime, dramas/TV)
- `test_engine.py`: sanity checks

## Limitations
- The catalog is small and hand-written, so results are only as good as its tags.
- Content-based only: it does not learn from other users.
- Uploaded Kaggle files have no mood tags, so matches rely on genres alone.