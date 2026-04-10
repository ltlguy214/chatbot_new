from __future__ import annotations

import base64
import json
import os
import time
import urllib.parse
import urllib.request

from typing import Any

import streamlit as st


class SpotifyClient:
    """Small Spotify Web API client (Client Credentials) with token caching."""

    def __init__(
        self,
        *,
        client_id_env: str = 'SPOTIFY_CLIENT_ID',
        client_secret_env: str = 'SPOTIFY_CLIENT_SECRET',
        token_cache_key: str = 'spotify_token_cache',
    ) -> None:
        self.client_id_env = str(client_id_env)
        self.client_secret_env = str(client_secret_env)
        self.token_cache_key = str(token_cache_key)
        self._local_cache: dict[str, Any] = {}

    def _cache(self) -> Any:
        """Return a mapping-like cache.

        Prefer Streamlit session_state when available; fall back to local dict.
        """

        try:
            # Accessing session_state outside `streamlit run` can warn;
            # still safe and allows reuse when running as an app.
            return st.session_state
        except Exception:
            return self._local_cache

    def access_token(self) -> str | None:
        client_id = os.getenv(self.client_id_env, '')
        client_secret = os.getenv(self.client_secret_env, '')
        if not (client_id and client_secret):
            return None

        cache = self._cache()
        token_cache = None
        try:
            token_cache = cache.get(self.token_cache_key)
        except Exception:
            token_cache = None
        if (
            isinstance(token_cache, dict)
            and token_cache.get('expires_at', 0) > time.time() + 30
            and token_cache.get('token')
        ):
            return str(token_cache.get('token'))

        credentials = f"{client_id}:{client_secret}".encode('utf-8')
        basic = base64.b64encode(credentials).decode('utf-8')
        payload = urllib.parse.urlencode({'grant_type': 'client_credentials'}).encode('utf-8')

        req = urllib.request.Request(
            url='https://accounts.spotify.com/api/token',
            data=payload,
            headers={'Authorization': f'Basic {basic}', 'Content-Type': 'application/x-www-form-urlencoded'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode('utf-8'))

        token = data.get('access_token')
        expires_in = int(data.get('expires_in', 3600))
        if token:
            try:
                cache[self.token_cache_key] = {
                    'token': token,
                    'expires_at': time.time() + expires_in,
                }
            except Exception:
                self._local_cache[self.token_cache_key] = {
                    'token': token,
                    'expires_at': time.time() + expires_in,
                }
        return str(token) if token else None

    def api_get_json(self, url: str, *, timeout: int = 12) -> dict | None:
        try:
            token = self.access_token()
        except Exception:
            token = None
        if not token:
            return None

        try:
            req = urllib.request.Request(
                url=url,
                headers={'Authorization': f'Bearer {token}'},
                method='GET',
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def pick_image_url(self, images: object, *, prefer_width: int = 320) -> str:
        if not isinstance(images, list) or not images:
            return ''
        candidates: list[dict] = [i for i in images if isinstance(i, dict) and i.get('url')]
        if not candidates:
            return ''
        if prefer_width <= 0:
            return str(candidates[0].get('url') or '')

        def _score(img: dict) -> int:
            w = img.get('width')
            try:
                w_int = int(w) if w is not None else prefer_width
            except Exception:
                w_int = prefer_width
            return abs(w_int - prefer_width)

        best = sorted(candidates, key=_score)[0]
        return str(best.get('url') or '')

    def search_artist(self, artist_name: str, *, market: str = 'VN') -> dict | None:
        name = str(artist_name or '').strip()
        if not name:
            return None

        q = urllib.parse.quote(name)
        url = f'https://api.spotify.com/v1/search?q={q}&type=artist&limit=5&market={urllib.parse.quote(market)}'
        data = self.api_get_json(url)
        if not data:
            return None

        items = ((data.get('artists') or {}).get('items') or [])
        if not items:
            return None

        best = items[0] or {}
        if not isinstance(best, dict) or not best.get('id'):
            return None
        return best

    def get_artist_top_tracks(self, artist_id: str, *, market: str = 'VN', limit: int = 5) -> list[dict]:
        artist_id = str(artist_id or '').strip()
        if not artist_id:
            return []

        url = f'https://api.spotify.com/v1/artists/{urllib.parse.quote(artist_id)}/top-tracks?market={urllib.parse.quote(market)}'
        data = self.api_get_json(url)
        tracks = (data or {}).get('tracks') or []
        if not isinstance(tracks, list):
            return []
        return tracks[: max(1, int(limit))]

    def get_track_metadata(self, track_id: str) -> dict | None:
        track_id = str(track_id or '').strip()
        if not track_id:
            return None

        url = f'https://api.spotify.com/v1/tracks/{urllib.parse.quote(track_id)}'
        data = self.api_get_json(url)
        return data if isinstance(data, dict) else None

    def get_tracks_metadata(self, track_ids: list[str], *, batch_size: int = 5) -> dict[str, dict]:
        """Batch fetch track objects by ids.

        Uses Spotify endpoint: GET /v1/tracks?ids=...
        Returns mapping: track_id -> track_payload.
        """

        if not isinstance(track_ids, list) or not track_ids:
            return {}

        try:
            batch_size_int = max(1, int(batch_size))
        except Exception:
            batch_size_int = 5

        # Spotify supports up to 50 ids per request; we intentionally keep it small.
        batch_size_int = min(batch_size_int, 50)

        ids: list[str] = []
        for tid in track_ids:
            tid = str(tid or '').strip()
            if not tid:
                continue
            if tid not in ids:
                ids.append(tid)

        if not ids:
            return {}

        token = self.access_token()
        if not token:
            return {}

        out: dict[str, dict] = {}

        for i in range(0, len(ids), batch_size_int):
            chunk = ids[i : i + batch_size_int]
            q = urllib.parse.quote(','.join(chunk))
            url = f'https://api.spotify.com/v1/tracks?ids={q}'
            try:
                req = urllib.request.Request(
                    url=url,
                    headers={'Authorization': f'Bearer {token}'},
                    method='GET',
                )
                with urllib.request.urlopen(req, timeout=12) as resp:
                    data = json.loads(resp.read().decode('utf-8'))
                tracks = (data or {}).get('tracks') or []
                if not isinstance(tracks, list):
                    continue
                for t in tracks:
                    if isinstance(t, dict) and t.get('id'):
                        out[str(t.get('id'))] = t
            except Exception:
                continue

        return out


_DEFAULT_SPOTIFY_CLIENT = SpotifyClient()


def spotify_access_token() -> str | None:
    """Backward-compatible wrapper."""

    return _DEFAULT_SPOTIFY_CLIENT.access_token()


def spotify_api_get_json(url: str, *, timeout: int = 12) -> dict | None:
    return _DEFAULT_SPOTIFY_CLIENT.api_get_json(url, timeout=timeout)


def spotify_pick_image_url(images: object, *, prefer_width: int = 320) -> str:
    return _DEFAULT_SPOTIFY_CLIENT.pick_image_url(images, prefer_width=prefer_width)


def spotify_search_artist(artist_name: str, *, market: str = 'VN') -> dict | None:
    return _DEFAULT_SPOTIFY_CLIENT.search_artist(artist_name, market=market)


def spotify_get_artist_top_tracks(artist_id: str, *, market: str = 'VN', limit: int = 5) -> list[dict]:
    return _DEFAULT_SPOTIFY_CLIENT.get_artist_top_tracks(artist_id, market=market, limit=limit)


def spotify_get_track_metadata(track_id: str) -> dict | None:
    return _DEFAULT_SPOTIFY_CLIENT.get_track_metadata(track_id)


def spotify_get_tracks_metadata(track_ids: list[str], *, batch_size: int = 5) -> dict[str, dict]:
    return _DEFAULT_SPOTIFY_CLIENT.get_tracks_metadata(track_ids, batch_size=batch_size)
