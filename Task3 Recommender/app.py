"""Watch Next - entertainment recommender (DecodeLabs AI Project 3)."""
import html
from pathlib import Path
import random

import pandas as pd
import streamlit as st

from recommender import (LENGTH_LABELS, Recommender, load_catalog,
                         merge_catalogs, normalize_upload)

st.set_page_config(page_title="Watch Next", page_icon="🎬", layout="centered")
st.markdown(
    """<style>
    .chip{display:inline-block;padding:2px 10px;margin:0 6px 6px 0;border-radius:999px;
          background:#D5E3E5;color:#1F4A4F;font-size:.8rem}
    .chip.type{background:#2F5D62;color:#fff}
    </style>""",
    unsafe_allow_html=True,
)

for key, default in (("ratings", {}), ("skip", set()), ("saved", []), ("surprise", None)):
    st.session_state.setdefault(key, default)


# ---------- data ----------
@st.cache_data
def get_catalog() -> pd.DataFrame:
    return load_catalog(Path(__file__).parent / "data" / "catalog.csv")


@st.cache_resource(show_spinner=False)
def get_engine(_df: pd.DataFrame, key: int) -> Recommender:
    return Recommender(_df)


df = get_catalog()
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Add a Kaggle CSV (optional)", type="csv",
                          help="Netflix titles or anime.csv style files work.")
    if up is not None:
        try:
            df = merge_catalogs(df, normalize_upload(pd.read_csv(up)))
            st.success(f"Catalog now has {len(df):,} titles.")
        except Exception as exc:  # bad file -> keep the built-in catalog
            st.error(f"Could not use that file: {exc}")
engine = get_engine(df, int(pd.util.hash_pandas_object(df["title"]).sum()))
titles = sorted(df["title"])

with st.sidebar:
    st.header("Filters")
    types = st.multiselect("Type", sorted(df["type"].unique()), default=sorted(df["type"].unique()))
    years = df["year"].dropna()
    lo, hi = (int(years.min()), int(years.max())) if len(years) else (1950, 2026)
    year_range = st.slider("Release year", lo, hi, (lo, hi)) if lo < hi else None
    lengths = st.multiselect("Length", list(LENGTH_LABELS), default=list(LENGTH_LABELS),
                             format_func=LENGTH_LABELS.get)
    n_results = st.slider("Number of results", 5, 20, 8)
    mix = st.checkbox("Mix types (movies, anime, dramas)", value=True)
    if st.button("Reset ratings and skipped titles"):
        st.session_state.ratings, st.session_state.skip = {}, set()
        st.session_state.surprise = None

filters = dict(types=types, lengths=lengths, year_range=year_range)


# ---------- callbacks ----------
def add_rating():
    t = st.session_state.get("rate_title")
    if t:
        st.session_state.ratings[t] = int(st.session_state.get("rate_val", 4))


def remove_rating(t):
    st.session_state.ratings.pop(t, None)


def save_item(t):
    if t not in st.session_state.saved:
        st.session_state.saved.append(t)


def unsave_item(t):
    if t in st.session_state.saved:
        st.session_state.saved.remove(t)


def skip_item(t):
    st.session_state.skip.add(t)


# ---------- card ----------
def card(i: int, score: float, why: str, prefix: str):
    row = engine.df.iloc[i]
    t = row["title"]
    with st.container(border=True):
        st.subheader(t)
        chips = [f'<span class="chip type">{html.escape(str(row["type"]))}</span>']
        if pd.notna(row["year"]):
            chips.append(f'<span class="chip">{int(row["year"])}</span>')
        chips.append(f'<span class="chip">{html.escape(LENGTH_LABELS.get(row["length"], str(row["length"])))}</span>')
        chips += [f'<span class="chip">{html.escape(g)}</span>' for g in str(row["genres"]).split("|") if g]
        st.markdown("".join(chips), unsafe_allow_html=True)
        if row["description"]:
            st.write(row["description"])
        st.progress(min(max(score, 0.0), 1.0), text=f"{round(score * 100)}% match")
        st.caption(f"Why: {why}")
        c1, c2 = st.columns(2)
        if t in st.session_state.saved:
            c1.button("Remove from list", key=f"{prefix}_un_{i}", on_click=unsave_item, args=(t,))
        else:
            c1.button("Save to list", key=f"{prefix}_sv_{i}", on_click=save_item, args=(t,))
        c2.button("Not interested", key=f"{prefix}_sk_{i}", on_click=skip_item, args=(t,))


# ---------- page ----------
st.title("Watch Next")
st.caption("Tell it what you like. It matches your taste to movies, anime and dramas using cosine similarity.")
tab_you, tab_sim, tab_list = st.tabs(["For you", "Find similar", "My list"])

with tab_you:
    genres = st.multiselect("Genres you enjoy", engine.genres)
    moods = st.multiselect("Mood you want", engine.moods)
    with st.expander("Rate titles you've watched (makes results sharper)"):
        c1, c2 = st.columns([3, 2])
        c1.selectbox("Title", titles, index=None, placeholder="Search a title", key="rate_title")
        c2.select_slider("Rating", [1, 2, 3, 4, 5], value=4, key="rate_val")
        st.button("Add rating", on_click=add_rating)
        for t, r in list(st.session_state.ratings.items()):
            a, b = st.columns([4, 1])
            a.write(f"{t}: {'★' * r}{'☆' * (5 - r)}")
            b.button("Remove", key=f"rm_{t}", on_click=remove_rating, args=(t,))

    ratings = st.session_state.ratings
    liked_w = 0.5
    if ratings and (genres or moods):
        liked_w = st.slider("Weight on your ratings vs. chosen genres and moods", 0.0, 1.0, 0.5)
    # "Not interested" counts as a 2-star rating, so the profile learns from it
    signals = {**{t: 2 for t in st.session_state.skip}, **ratings}
    profile = engine.build_profile(genres, moods, signals, liked_w)
    exclude = set(ratings) | st.session_state.skip

    if not profile.any():
        st.info("Pick a few genres or moods, or rate a title you've watched, to get recommendations.")
    else:
        if st.button("Surprise me"):
            pool = engine.recommend(profile, n=15, exclude=exclude, mix_types=mix, **filters)
            st.session_state.surprise = engine.df.at[random.choice(pool)[0], "title"] if pool else None
        sp = st.session_state.surprise
        if sp and sp not in exclude:
            i = engine.title_idx[sp.lower()]
            st.success("Surprise pick from your top matches")
            card(i, engine.score(i, profile), engine.why_profile(i, profile, signals), "sup")

        results = engine.recommend(profile, n=n_results, exclude=exclude, mix_types=mix, **filters)
        if not results:
            st.warning("No close matches with these filters. Widen the type, year or length filters, or add more genres.")
        elif len(results) < n_results:
            st.caption(f"Showing {len(results)} close matches. The rest fall below the match threshold.")
        for i, s in results:
            card(i, s, engine.why_profile(i, profile, signals), "rec")

with tab_sim:
    base = st.selectbox("Pick a title you like", titles, index=None, placeholder="Search a title", key="sim_title")
    cross = st.checkbox("Only show other types (e.g. a drama you loved -> anime or movies)")
    if base:
        found = engine.similar(base, n=n_results, other_types_only=cross, mix_types=mix and not cross, **filters)
        if not found:
            st.warning("Nothing matches these filters.")
        bi = engine.title_idx[base.lower()]
        for i, s in found:
            card(i, s, engine.why_similar(i, bi), "sim")
    else:
        st.info("Choose a title to see what is closest to it.")

with tab_list:
    st.subheader("Saved")
    saved = st.session_state.saved
    if saved:
        out = engine.df[engine.df["title"].isin(saved)][["title", "type", "year", "genres"]]
        st.dataframe(out, hide_index=True, width="stretch")
        st.download_button("Download list (CSV)", out.to_csv(index=False), "watchlist.csv", "text/csv")
    else:
        st.info("Nothing saved yet. Use Save to list on any recommendation.")
    st.subheader("What drives your matches")
    likes, avoids = engine.profile_summary(profile)
    if likes.empty:
        st.caption("Pick genres or moods, or rate titles you liked, to see your taste profile.")
    else:
        st.bar_chart(likes, x="Tag", y="Strength", color="Kind", horizontal=True, sort="-Strength")
        st.caption("Longer bar = stronger pull on your recommendations (strongest tag = 100).")
    if avoids:
        st.write("Pushing results away: " + ", ".join(avoids))