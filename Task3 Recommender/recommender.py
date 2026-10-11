"""Recommendation engine: content-based filtering with cosine similarity.

How it works
1. Every title becomes a vector over tags (genres, moods, type).
2. Rare tags count more than common ones (IDF weighting).
3. The user's taste (chosen genres/moods + ratings) becomes a vector too.
4. cosine(user, title) = how closely their directions match -> ranked list.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["title", "type", "year", "genres", "moods", "length", "description"]
LENGTH_LABELS = {
    "Film": "Film",
    "Short series": "Short series (under ~30 eps)",
    "Long series": "Long series",
}
# Rating -> how strongly a title pulls the taste profile (low ratings push away)
RATING_WEIGHT = {1: -1.0, 2: -0.5, 3: 0.25, 4: 1.0, 5: 1.5}
GENRE_W, MOOD_W, TYPE_W = 1.0, 0.8, 0.3


def split_tags(value) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [t.strip() for t in str(value).split("|") if t.strip()]


class Recommender:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        n = len(self.df)
        self.tags = [
            [f"g:{g}" for g in split_tags(r.genres)]
            + [f"m:{m}" for m in split_tags(r.moods)]
            + [f"t:{r.type}"]
            for r in self.df.itertuples()
        ]
        self.vocab = sorted({t for ts in self.tags for t in ts})
        self.index = {t: i for i, t in enumerate(self.vocab)}
        v = len(self.vocab)

        doc_freq = np.zeros(v)
        for ts in self.tags:
            for t in set(ts):
                doc_freq[self.index[t]] += 1
        idf = np.log((1 + n) / (1 + doc_freq)) + 1
        group_w = np.array([{"g": GENRE_W, "m": MOOD_W, "t": TYPE_W}[t[0]] for t in self.vocab])
        self.weights = idf * group_w

        X = np.zeros((n, v))
        for i, ts in enumerate(self.tags):
            for t in set(ts):
                X[i, self.index[t]] = self.weights[self.index[t]]
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        self.X = X / np.where(norms == 0, 1, norms)  # unit rows -> dot product = cosine

        self.title_idx = {t.lower(): i for i, t in enumerate(self.df["title"])}
        self.genres = sorted(t[2:] for t in self.vocab if t.startswith("g:"))
        self.moods = sorted(t[2:] for t in self.vocab if t.startswith("m:"))

    # ---- taste profile -------------------------------------------------
    @staticmethod
    def _unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def build_profile(self, genres=(), moods=(), ratings=None, liked_weight=0.5, avoid_weight=0.4):
        """Taste vector = chosen genres/moods + liked titles, minus a bounded push away from disliked ones.

        Dislikes can only nudge the profile (at most `avoid_weight` of a unit vector);
        they never create a profile on their own, so one click cannot flip the results.
        """
        size = len(self.vocab)
        pref, pos, neg, n_neg = np.zeros(size), np.zeros(size), np.zeros(size), 0
        for prefix, items in (("g", genres), ("m", moods)):
            for name in items:
                k = self.index.get(f"{prefix}:{name}")
                if k is not None:
                    pref[k] = self.weights[k]
        for title, r in (ratings or {}).items():
            i = self.title_idx.get(title.lower())
            if i is None:
                continue
            w = RATING_WEIGHT[int(r)]
            if w > 0:
                pos += w * self.X[i]
            else:
                neg += w * self.X[i]
                n_neg += 1
        pref, pos = self._unit(pref), self._unit(pos)
        if pref.any() and pos.any():
            base = (1 - liked_weight) * pref + liked_weight * pos
        else:
            base = pref if pref.any() else pos
        if not base.any():
            return np.zeros(size)  # nothing positive to anchor on
        return base + avoid_weight * neg / max(n_neg, 1)

    def profile_summary(self, profile: np.ndarray, k=10):
        """(DataFrame of strongest liked tags scaled to 0-100, list of tags pushing results away)."""
        rows = [(self.vocab[i][2:], "genre" if self.vocab[i][0] == "g" else "mood", float(profile[i]))
                for i in np.argsort(-profile) if self.vocab[i][0] != "t" and profile[i] > 0][:k]
        top = max((r[2] for r in rows), default=1.0)
        likes = pd.DataFrame({"Tag": [r[0] for r in rows], "Kind": [r[1] for r in rows],
                              "Strength": [round(100 * r[2] / top) for r in rows]})
        avoids = [self.vocab[i][2:] for i in np.argsort(profile)
                  if self.vocab[i][0] != "t" and profile[i] < -0.05][:5]
        return likes, avoids

    # ---- filtering -----------------------------------------------------
    def _mask(self, types=None, year_range=None, lengths=None) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)
        if types is not None:
            mask &= self.df["type"].isin(types).to_numpy()
        if lengths is not None:
            mask &= self.df["length"].isin(lengths).to_numpy()
        if year_range is not None:
            y = self.df["year"]
            mask &= (y.isna() | y.between(*year_range)).to_numpy()
        return mask

    def _round_robin(self, order: list[int]) -> list[int]:
        buckets: dict[str, list[int]] = {}
        for i in order:
            buckets.setdefault(self.df.at[i, "type"], []).append(i)
        out: list[int] = []
        while any(buckets.values()):
            for t in buckets:
                if buckets[t]:
                    out.append(buckets[t].pop(0))
        return out

    def _rank(self, scores, mask, n, mix_types):
        order = [int(i) for i in np.argsort(-scores) if mask[i]]
        if mix_types:
            order = self._round_robin(order)
        return [(i, float(scores[i])) for i in order[:n]]

    # ---- recommend / similar ------------------------------------------
    def recommend(self, profile, n=10, exclude=(), mix_types=False, min_score=0.05, **filters):
        scores = self.X @ self._unit(profile)
        mask = self._mask(**filters) & (scores > min_score)
        for t in exclude:
            i = self.title_idx.get(t.lower())
            if i is not None:
                mask[i] = False
        return self._rank(scores, mask, n, mix_types)

    def similar(self, title, n=10, other_types_only=False, mix_types=False, **filters):
        i = self.title_idx[title.lower()]
        scores = self.X @ self.X[i]
        mask = self._mask(**filters)
        mask[i] = False
        if other_types_only:
            mask &= (self.df["type"] != self.df.at[i, "type"]).to_numpy()
        return self._rank(scores, mask, n, mix_types)

    def score(self, i: int, profile: np.ndarray) -> float:
        return float(self.X[i] @ self._unit(profile))

    # ---- explanations --------------------------------------------------
    def why_profile(self, i, profile, ratings=None, top=4) -> str:
        contrib = [
            (profile[self.index[t]] * self.X[i, self.index[t]], t[2:])
            for t in set(self.tags[i])
            if not t.startswith("t:") and profile[self.index[t]] > 0
        ]
        names = [name for _, name in sorted(contrib, reverse=True)[:top]]
        near, best = None, 0.3
        for title, r in (ratings or {}).items():
            j = self.title_idx.get(title.lower())
            if j is not None and r >= 4 and j != i:
                s = float(self.X[i] @ self.X[j])
                if s > best:
                    near, best = self.df.at[j, "title"], s
        parts = []
        if names:
            parts.append("matches your taste for " + ", ".join(names))
        if near:
            parts.append(f"close to {near}, which you rated highly")
        if not parts:
            return "Closest overall match to your profile"
        text = "; ".join(parts)
        return text[0].upper() + text[1:]

    def why_similar(self, i, j, top=5) -> str:
        shared = [t[2:] for t in self.tags[i] if t in set(self.tags[j]) and not t.startswith("t:")]
        return ("Shares " + ", ".join(shared[:top])) if shared else "Similar overall profile"


# ---- data loading --------------------------------------------------------
_DROP = {"International Movies", "International TV Shows", "Independent Movies", "Classic Movies",
         "Cult Movies", "Movies", "TV Shows", "Anime Features", "Anime Series", "Music & Musicals"}
_MAP = {
    "Dramas": ["Drama"], "TV Dramas": ["Drama"], "Comedies": ["Comedy"], "TV Comedies": ["Comedy"],
    "Thrillers": ["Thriller"], "TV Thrillers": ["Thriller"], "Romantic Movies": ["Romance"],
    "Romantic TV Shows": ["Romance"], "Horror Movies": ["Horror"], "TV Horror": ["Horror"],
    "Action & Adventure": ["Action", "Adventure"], "TV Action & Adventure": ["Action", "Adventure"],
    "Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"], "TV Sci-Fi & Fantasy": ["Sci-Fi", "Fantasy"],
    "Crime TV Shows": ["Crime"], "TV Mysteries": ["Mystery"], "Sports Movies": ["Sports"],
}


def _clean_genres(value) -> str:
    out: list[str] = []
    for g in str(value).split(","):
        g = g.strip()
        if not g or g in _DROP or g.lower() == "nan":
            continue
        for item in _MAP.get(g, [g.removesuffix(" Movies").removesuffix(" TV Shows")]):
            if item not in out:
                out.append(item)
    return "|".join(out)


def load_catalog(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["moods"] = df["moods"].fillna("")
    df["description"] = df["description"].fillna("")
    return df[COLUMNS].drop_duplicates(subset="title").reset_index(drop=True)


def normalize_upload(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a Kaggle-style CSV (Netflix titles or anime.csv) onto the catalog format."""
    cols = {c.lower().strip(): c for c in raw.columns}

    def col(*names):
        for n in names:
            if n in cols:
                return raw[cols[n]]
        return None

    title, genres = col("title", "name"), col("listed_in", "genres", "genre")
    if title is None or genres is None:
        raise ValueError("The CSV needs a title/name column and a genres column (listed_in, genres or genre).")
    out = pd.DataFrame({"title": title.astype(str).str.strip()})
    out["genres"] = genres.map(_clean_genres)
    desc = col("description", "synopsis")
    out["description"] = desc.fillna("").astype(str) if desc is not None else ""
    out["moods"] = ""
    if "listed_in" in cols:  # Netflix-titles style
        typ = col("type").astype(str)
        out["type"] = np.where(typ == "Movie", "Movie", "Drama/TV")
        out["year"] = pd.to_numeric(col("release_year"), errors="coerce") if col("release_year") is not None else np.nan
        dur = col("duration")
        seasons = pd.to_numeric(dur.astype(str).str.extract(r"(\d+)")[0], errors="coerce") if dur is not None else pd.Series(np.nan, index=raw.index)
        out["length"] = np.where(typ == "Movie", "Film", np.where(seasons.fillna(1) <= 2, "Short series", "Long series"))
    else:  # anime.csv style
        out["type"] = "Anime"
        out["year"] = np.nan
        eps = pd.to_numeric(col("episodes"), errors="coerce") if col("episodes") is not None else pd.Series(np.nan, index=raw.index)
        out["length"] = np.where(eps == 1, "Film", np.where(eps.fillna(12) < 30, "Short series", "Long series"))
    return out[COLUMNS]


def merge_catalogs(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([base, extra], ignore_index=True)
    both = both[~both["title"].str.lower().duplicated()]
    return both.reset_index(drop=True)