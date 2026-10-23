"""Streamlit gallery. Run with: streamlit run src/photo_search/app.py"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from photo_search.cli import DEFAULT_INDEX
from photo_search.encoder import ClipEncoder
from photo_search.index import PhotoIndex

st.set_page_config(page_title="photo-search", page_icon="🔎", layout="wide")


@st.cache_resource
def load_index() -> PhotoIndex:
    return PhotoIndex(DEFAULT_INDEX, ClipEncoder())


index = load_index()

with st.sidebar:
    st.title("photo-search")
    st.caption(f"{len(index):,} photos indexed · runs offline")
    st.caption("Built by [Abbishekk DGS](https://abbishekkdgs.tech)")
    folder = st.text_input("Folder to index", str(Path.home() / "Pictures"))
    if st.button("Index folder", width="stretch"):
        bar = st.progress(0.0)
        stats = index.update(
            Path(folder).expanduser(), progress=lambda d, t: bar.progress(d / max(t, 1))
        )
        st.success(f"+{stats['added']} new, -{stats['removed']} removed")
    k = st.slider("Results", 6, 60, 24, step=6)
    cols_n = st.slider("Columns", 3, 8, 6)

tab_text, tab_image, tab_dupes = st.tabs(["Search by text", "Search by image", "Duplicates"])


def grid(results) -> None:
    cols = st.columns(cols_n)
    for i, r in enumerate(results):
        with cols[i % cols_n]:
            st.image(r.path, width="stretch")
            st.caption(f"{r.score:.2f} · {Path(r.path).name}")


with tab_text:
    query = st.text_input("Describe the photo", placeholder="my dog on the beach at sunset")
    if query:
        grid(index.search_text(query, k))

with tab_image:
    upload = st.file_uploader("Find photos like this one", type=["jpg", "jpeg", "png", "webp"])
    if upload:
        with tempfile.NamedTemporaryFile(suffix=Path(upload.name).suffix) as tmp:
            tmp.write(upload.getvalue())
            tmp.flush()
            grid(index.search_image(Path(tmp.name), k))

with tab_dupes:
    threshold = st.slider("Similarity threshold", 0.85, 0.99, 0.95, 0.01)
    if st.button("Find near-duplicates"):
        for group in index.duplicates(threshold)[:30]:
            st.markdown(f"**{len(group)} similar photos**")
            cols = st.columns(min(len(group), cols_n))
            for i, p in enumerate(group[:cols_n]):
                cols[i].image(p, caption=Path(p).name, width="stretch")
