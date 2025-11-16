#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to merge metadata from master_song_list.csv with platform data.
Aggregates statistics from Apple Music, Spotify, NCT, and ZingMP3.
"""

import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
import re

# Set UTF-8 encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# File paths
MASTER_SONG_LIST = 'data/song_list_info.csv'
APPLE_MUSIC_FILE = 'data/apple_music_top100_kworb_vn.csv'
SPOTIFY_FILE = 'data/spotify_top100_kworb_vn.csv'
NCT_FILE = 'data/nct_top50.csv'
ZINGMP3_FILE = 'data/zingmp3_top100.csv'
OUTPUT_FILE = 'data/vpop_master_metadata.csv'

def normalize_text(text):
    """Normalize text for matching: lowercase, remove extra spaces, special chars"""
    if pd.isna(text) or text == '':
        return ''
    text = str(text).lower().strip()
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def create_match_key(title, artist):
    """Create a matching key from title and artist"""
    return f"{normalize_text(title)}|{normalize_text(artist)}"

def get_platform_stats(df_master, platform_df, platform_name):
    """
    Match master song list with platform data and aggregate statistics.
    Returns DataFrame with platform-specific stats.
    """
    print(f"\n📊 Processing {platform_name} data...")
    
    if platform_df is None or platform_df.empty:
        print(f"  ⚠️  No data available for {platform_name}")
        return None
    
    # Create match keys
    df_master['match_key'] = df_master.apply(
        lambda row: create_match_key(row['title'], row['artists']), axis=1
    )
    platform_df['match_key'] = platform_df.apply(
        lambda row: create_match_key(row['Title'], row['Artist']), axis=1
    )
    
    # Group by song and aggregate
    stats = []
    
    for idx, song in df_master.iterrows():
        match_key = song['match_key']
        
        # Find all occurrences of this song in platform data
        matches = platform_df[platform_df['match_key'] == match_key]
        
        if matches.empty:
            continue
        
        song_stats = {
            'match_key': match_key,
            'title': song['title'],
            'artists': song['artists'],
        }
        
        # Platform-specific stats
        if platform_name == 'APPLE_MUSIC':
            song_stats[f'{platform_name}_appearances'] = len(matches)
            song_stats[f'{platform_name}_peak_rank'] = matches['Rank'].min()
            
        elif platform_name == 'SPOTIFY':
            song_stats[f'{platform_name}_appearances'] = len(matches)
            song_stats[f'{platform_name}_peak_rank'] = matches['Rank'].min()
            song_stats[f'{platform_name}_days_on_chart'] = matches['Days_On_Chart'].max()
            song_stats[f'{platform_name}_total_streams'] = matches['Total_Streams'].max()
            song_stats[f'{platform_name}_peak_position'] = matches['Peak_Position'].min()
            
        elif platform_name == 'NCT':
            song_stats[f'{platform_name}_appearances'] = len(matches)
            song_stats[f'{platform_name}_peak_rank'] = matches['Rank'].min()
            song_stats[f'{platform_name}_total_likes'] = matches['Total_Likes'].max()
            
        elif platform_name == 'ZINGMP3':
            song_stats[f'{platform_name}_appearances'] = len(matches)
            song_stats[f'{platform_name}_peak_rank'] = matches['Rank'].min()
            song_stats[f'{platform_name}_total_likes'] = matches['Total_Likes'].max()
            if 'Total_Plays' in matches.columns:
                song_stats[f'{platform_name}_total_plays'] = matches['Total_Plays'].max()
        
        stats.append(song_stats)
    
    if not stats:
        print(f"  ⚠️  No matches found for {platform_name}")
        return None
    
    df_stats = pd.DataFrame(stats)
    print(f"  ✅ Matched {len(df_stats)} songs from {platform_name}")
    
    return df_stats

def main():
    print("=" * 70)
    print("🎵 VPOP METADATA MERGER")
    print("=" * 70)
    
    # Load master song list
    print(f"\n📂 Loading master song list: {MASTER_SONG_LIST}")
    df_master = pd.read_csv(MASTER_SONG_LIST, encoding='utf-8-sig')
    print(f"  ✅ Loaded {len(df_master)} songs")
    
    # Load platform data
    print("\n📂 Loading platform data files...")
    
    try:
        df_apple = pd.read_csv(APPLE_MUSIC_FILE, encoding='utf-8-sig')
        print(f"  ✅ Apple Music: {len(df_apple)} records")
    except Exception as e:
        print(f"  ⚠️  Apple Music: Error loading - {e}")
        df_apple = None
    
    try:
        df_spotify = pd.read_csv(SPOTIFY_FILE, encoding='utf-8-sig')
        print(f"  ✅ Spotify: {len(df_spotify)} records")
    except Exception as e:
        print(f"  ⚠️  Spotify: Error loading - {e}")
        df_spotify = None
    
    try:
        df_nct = pd.read_csv(NCT_FILE, encoding='utf-8-sig')
        print(f"  ✅ NCT: {len(df_nct)} records")
    except Exception as e:
        print(f"  ⚠️  NCT: Error loading - {e}")
        df_nct = None
    
    try:
        df_zingmp3 = pd.read_csv(ZINGMP3_FILE, encoding='utf-8-sig')
        print(f"  ✅ ZingMP3: {len(df_zingmp3)} records")
    except Exception as e:
        print(f"  ⚠️  ZingMP3: Error loading - {e}")
        df_zingmp3 = None
    
    # Get stats from each platform
    stats_dfs = []
    
    if df_apple is not None:
        df_apple_stats = get_platform_stats(df_master.copy(), df_apple, 'APPLE_MUSIC')
        if df_apple_stats is not None:
            stats_dfs.append(df_apple_stats)
    
    if df_spotify is not None:
        df_spotify_stats = get_platform_stats(df_master.copy(), df_spotify, 'SPOTIFY')
        if df_spotify_stats is not None:
            stats_dfs.append(df_spotify_stats)
    
    if df_nct is not None:
        df_nct_stats = get_platform_stats(df_master.copy(), df_nct, 'NCT')
        if df_nct_stats is not None:
            stats_dfs.append(df_nct_stats)
    
    if df_zingmp3 is not None:
        df_zingmp3_stats = get_platform_stats(df_master.copy(), df_zingmp3, 'ZINGMP3')
        if df_zingmp3_stats is not None:
            stats_dfs.append(df_zingmp3_stats)
    
    # Merge all stats
    print("\n🔗 Merging all platform statistics...")
    df_master['match_key'] = df_master.apply(
        lambda row: create_match_key(row['title'], row['artists']), axis=1
    )
    
    df_result = df_master.copy()
    
    for stats_df in stats_dfs:
        df_result = df_result.merge(
            stats_df.drop(columns=['title', 'artists'], errors='ignore'),
            on='match_key',
            how='left'
        )
    
    # Add Spotify metadata (popularity, genres, release date)
    print("\n🎧 Adding Spotify metadata (popularity, genres, release date)...")
    try:
        df_spotify_meta = pd.read_csv(SPOTIFY_DATA_FILE, encoding='utf-8-sig')
        
        # Keep only unique songs with Spotify data
        df_spotify_meta = df_spotify_meta[df_spotify_meta['spotify_track_id'].notna()]
        df_spotify_meta = df_spotify_meta.drop_duplicates(subset=['title', 'artists'])
        
        df_spotify_meta['match_key'] = df_spotify_meta.apply(
            lambda row: create_match_key(row['title'], row['artists']), axis=1
        )
        
        spotify_cols = ['match_key', 'spotify_popularity', 'spotify_release_date', 
                       'spotify_genres', 'spotify_track_id']
        df_spotify_meta_clean = df_spotify_meta[spotify_cols].drop_duplicates(subset=['match_key'])
        
        df_result = df_result.merge(
            df_spotify_meta_clean,
            on='match_key',
            how='left'
        )
        
        print(f"  ✅ Added Spotify metadata for {df_result['spotify_track_id'].notna().sum()} songs")
        
    except Exception as e:
        print(f"  ⚠️  Could not load Spotify metadata: {e}")
    
    # Calculate aggregate statistics
    print("\n📈 Calculating aggregate statistics...")
    
    # Total platforms the song appeared on
    platform_cols = [col for col in df_result.columns if '_appearances' in col]
    df_result['total_platforms'] = df_result[platform_cols].notna().sum(axis=1)
    
    # Best peak rank across all platforms
    peak_cols = [col for col in df_result.columns if '_peak_rank' in col]
    df_result['best_peak_rank'] = df_result[peak_cols].min(axis=1)
    
    # Average rank across all platforms
    avg_rank_cols = [col for col in df_result.columns if '_avg_rank' in col]
    df_result['overall_avg_rank'] = df_result[avg_rank_cols].mean(axis=1).round(2)
    
    # Total appearances across all platforms
    df_result['total_appearances'] = df_result[platform_cols].sum(axis=1)
    
    # Create hit score (lower is better: considers peak rank, total platforms, appearances)
    # Normalize: peak_rank (lower is better), platforms (higher is better), appearances (higher is better)
    df_result['hit_score'] = 0
    
    if df_result['best_peak_rank'].notna().any():
        # Normalize best_peak_rank (1-100) to 0-1 scale, inverted (1 = best)
        max_rank = 100
        df_result['hit_score'] += (max_rank - df_result['best_peak_rank']) / max_rank * 40
    
    if df_result['total_platforms'].max() > 0:
        # Normalize platforms (0-4) to 0-1 scale
        df_result['hit_score'] += (df_result['total_platforms'] / 4) * 30
    
    if df_result['total_appearances'].max() > 0:
        # Normalize appearances to 0-1 scale
        max_appearances = df_result['total_appearances'].max()
        df_result['hit_score'] += (df_result['total_appearances'] / max_appearances) * 30
    
    df_result['hit_score'] = df_result['hit_score'].round(2)
    
    # Clean up and reorder columns
    df_result = df_result.drop(columns=['match_key'])
    
    # Reorder columns: basic info first, then platform stats, then aggregates
    basic_cols = ['title', 'artists', 'featured_artists', 'source']
    
    spotify_meta_cols = [col for col in df_result.columns if col.startswith('spotify_')]
    
    apple_cols = [col for col in df_result.columns if col.startswith('APPLE_MUSIC_')]
    spotify_cols = [col for col in df_result.columns if col.startswith('SPOTIFY_') and col not in spotify_meta_cols]
    nct_cols = [col for col in df_result.columns if col.startswith('NCT_')]
    zingmp3_cols = [col for col in df_result.columns if col.startswith('ZINGMP3_')]
    
    aggregate_cols = ['total_platforms', 'best_peak_rank', 'overall_avg_rank', 
                     'total_appearances', 'hit_score']
    
    column_order = (basic_cols + spotify_meta_cols + aggregate_cols + 
                   apple_cols + spotify_cols + nct_cols + zingmp3_cols)
    
    # Keep only columns that exist
    column_order = [col for col in column_order if col in df_result.columns]
    df_result = df_result[column_order]
    
    # Sort by hit_score descending
    df_result = df_result.sort_values('hit_score', ascending=False).reset_index(drop=True)
    
    # Save result
    print(f"\n💾 Saving merged metadata to: {OUTPUT_FILE}")
    df_result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY STATISTICS")
    print("=" * 70)
    print(f"\nTotal songs in master list: {len(df_result)}")
    print(f"Songs with platform data: {df_result['total_platforms'].notna().sum()}")
    print(f"\nPlatform coverage:")
    print(f"  - Apple Music: {df_result['APPLE_MUSIC_appearances'].notna().sum()} songs")
    print(f"  - Spotify: {df_result['SPOTIFY_appearances'].notna().sum()} songs")
    print(f"  - NCT: {df_result['NCT_appearances'].notna().sum()} songs")
    print(f"  - ZingMP3: {df_result['ZINGMP3_appearances'].notna().sum()} songs")
    
    if 'spotify_track_id' in df_result.columns:
        print(f"\nSpotify metadata available: {df_result['spotify_track_id'].notna().sum()} songs")
    
    print(f"\n📈 Top 10 songs by hit score:")
    top_10 = df_result.head(10)[['title', 'artists', 'hit_score', 'total_platforms', 
                                   'best_peak_rank', 'total_appearances']]
    print(top_10.to_string(index=False))
    
    print("\n" + "=" * 70)
    print(f"✅ DONE! Metadata saved to: {OUTPUT_FILE}")
    print("=" * 70)

if __name__ == '__main__':
    main()
