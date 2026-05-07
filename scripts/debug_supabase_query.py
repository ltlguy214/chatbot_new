from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chatbot.env import load_env
from chatbot.supabase import SupabaseClientFactory
from chatbot.action_handler import get_genre_targets, handle_action


def main() -> None:
    load_env()
    client = SupabaseClientFactory().get_client()
    print('SUPABASE configured:', bool(client))
    if client is None:
        return

    res = (
        client.table('songs')
        .select('spotify_track_id, title, artists, genres')
        .ilike('artists', '%SIVAN%')
        .limit(10)
        .execute()
    )
    data = getattr(res, 'data', None) or []
    print('rows(artist SIVAN):', len(data))
    print(data[:3])

    res2 = (
        client.table('songs')
        .select('spotify_track_id, title, artists, genres')
        .ilike('artists', '%SIVAN%')
        .ilike('genres', '%Indie%')
        .limit(10)
        .execute()
    )
    data2 = getattr(res2, 'data', None) or []
    print('rows(SIVAN+Indie):', len(data2))
    print(data2[:3])

    print('get_genre_targets("indie") ->', get_genre_targets('indie'))
    print('get_genre_targets("indie, pop") ->', get_genre_targets('indie, pop'))

    for label, params in [
        ('TC01-like', {'mood': '', 'genre': 'indie', 'artist': 'SIVAN'}),
        ('TC05-like', {'mood': 'chill', 'genre': 'indie, pop', 'artist': 'SIVAN'}),
    ]:
        out = handle_action('DISCOVER_MUSIC', params, client, embed_fn=None, match_count=5)
        tracks = []
        path = None
        source = None
        error = None
        if isinstance(out, dict):
            tracks = out.get('tracks') or []
            path = out.get('path')
            source = out.get('source')
            error = out.get('error')
        print(f'\n{label}: source={source} error={error} path={path}')
        if isinstance(tracks, list):
            print(f'{label}: tracks={len(tracks)} top_ids={[t.get("spotify_id") for t in tracks[:5]]}')


if __name__ == '__main__':
    main()
