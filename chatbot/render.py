from __future__ import annotations

import streamlit as st

try:
    # Works when this code is imported as a proper Python package.
    from .formatting import format_duration_ms, format_int_vi
except ImportError:
    # Works on Streamlit Cloud when files are treated as top-level modules.
    from formatting import format_duration_ms, format_int_vi


class Renderer:
    def render_spotify_artist_payload(self, payload: dict) -> None:
        return render_spotify_artist_payload(payload)


def render_spotify_artist_payload(payload: dict) -> None:
    """Render artist + top tracks as image cards."""

    if not isinstance(payload, dict) or not payload:
        return

    artist = payload.get('artist') or {}
    tracks = payload.get('tracks') or []

    if isinstance(artist, dict):
        name = str(artist.get('name') or '')
        url = str(artist.get('external_url') or '')
        img = str(artist.get('image_url') or '')
        followers = artist.get('followers_total')
        popularity = artist.get('popularity')
        genres = artist.get('genres') or []

        cols = st.columns([1, 4], gap='small')
        with cols[0]:
            if img:
                try:
                    st.image(img, width=96)
                except Exception:
                    st.markdown(f"![artist]({img})")
        with cols[1]:
            title = f"**{name}**" if name else '**Nghệ sĩ**'
            if url:
                title = f"{title} — [Spotify]({url})"
            st.markdown(title)

            meta_parts: list[str] = []
            if followers is not None:
                meta_parts.append(f"Followers: {format_int_vi(followers)}")
            if popularity is not None:
                try:
                    meta_parts.append(f"Popularity: {int(popularity)}/100")
                except Exception:
                    pass
            if isinstance(genres, list) and genres:
                meta_parts.append(f"Genres: {', '.join([str(g) for g in genres[:4] if g])}")
            if meta_parts:
                st.caption(' • '.join(meta_parts))

    if not isinstance(tracks, list) or not tracks:
        st.info('Không lấy được danh sách bài nổi bật từ Spotify.')
        return

    st.markdown('**Top bài nổi bật**')
    for idx, t in enumerate(tracks[:5], start=1):
        if not isinstance(t, dict):
            continue

        cover = str(t.get('cover_url') or '')
        title = str(t.get('title') or '')
        artist_text = str(t.get('artist') or '')
        album = str(t.get('album') or '')
        release_date = str(t.get('release_date') or '')
        popularity = t.get('popularity')
        duration_ms = t.get('duration_ms')
        explicit = bool(t.get('explicit'))
        preview_url = str(t.get('preview_url') or '')
        preview_source = str(t.get('preview_source') or '')
        external_url = str(t.get('external_url') or '')

        row = st.columns([1, 4], gap='small')
        with row[0]:
            if cover:
                try:
                    st.image(cover, width=88)
                except Exception:
                    st.markdown(f"![cover]({cover})")
        with row[1]:
            line = f"**{idx}. {title}**" if title else f"**{idx}. (Không rõ tiêu đề)**"
            if explicit:
                line += ' · Explicit'
            st.markdown(line)

            sub_parts: list[str] = []
            if artist_text:
                sub_parts.append(artist_text)
            if album:
                sub_parts.append(album)
            if release_date:
                sub_parts.append(release_date)
            dur = format_duration_ms(duration_ms)
            if dur:
                sub_parts.append(dur)
            if popularity is not None:
                try:
                    sub_parts.append(f"Pop {int(popularity)}/100")
                except Exception:
                    pass
            if sub_parts:
                st.caption(' • '.join(sub_parts))

            if preview_url:
                st.audio(preview_url, format='audio/mpeg')
                if preview_source and preview_source != 'spotify':
                    st.caption(f"Preview source: {preview_source}")
            elif external_url:
                st.markdown(f"Nghe trên Spotify: {external_url}")

        st.divider()
