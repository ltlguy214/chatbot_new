import pandas as pd
from datetime import datetime, timedelta
import sys

# Load NCT data
nct_file = 'data/nct_top50.csv'
try:
    df = pd.read_csv(nct_file, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
except Exception as e:
    print(f"Error loading {nct_file}: {e}")
    sys.exit(1)

# Get unique dates
dates = sorted(df['Date'].unique(), reverse=True)
if len(dates) < 2:
    print("Need at least 2 days of data to compare")
    sys.exit(1)

# Compare latest 2 days
latest = dates[0]
previous = dates[1]

print("=" * 60)
print(f"NCT TOP 50 CHART COMPARISON")
print("=" * 60)
print(f"Latest:   {latest.strftime('%Y-%m-%d')}")
print(f"Previous: {previous.strftime('%Y-%m-%d')}")
print("=" * 60)
print()

# Get data for both dates
df_latest = df[df['Date'] == latest].sort_values('Rank')
df_prev = df[df['Date'] == previous].sort_values('Rank')

# Create comparison
print("📊 TOP 10 MOVEMENTS")
print("-" * 60)
for rank in range(1, 11):
    latest_song = df_latest[df_latest['Rank'] == rank].iloc[0]
    
    # Find same song in previous chart
    prev_match = df_prev[df_prev['Title'] == latest_song['Title']]
    
    if len(prev_match) > 0:
        prev_rank = prev_match.iloc[0]['Rank']
        movement = prev_rank - rank
        
        if movement > 0:
            arrow = f"↑ {movement}"
            color = "🟢"
        elif movement < 0:
            arrow = f"↓ {abs(movement)}"
            color = "🔴"
        else:
            arrow = "="
            color = "⚪"
        
        print(f"{color} #{rank:2d} {arrow:>6s} | {latest_song['Title'][:40]:<40s}")
        print(f"           | {latest_song['Artists'][:40]}")
        
        # Likes change
        latest_likes = int(latest_song['Total_Likes'])
        prev_likes = int(prev_match.iloc[0]['Total_Likes'])
        likes_change = latest_likes - prev_likes
        if likes_change > 0:
            print(f"           | Likes: {latest_likes:,} (+{likes_change:,})")
        else:
            print(f"           | Likes: {latest_likes:,}")
    else:
        print(f"🆕 #{rank:2d}   NEW | {latest_song['Title'][:40]:<40s}")
        print(f"           | {latest_song['Artists'][:40]}")
        print(f"           | Likes: {int(latest_song['Total_Likes']):,}")
    
    print()

# New entries (not in previous chart)
print("=" * 60)
print("🆕 NEW ENTRIES (not in previous Top 50)")
print("-" * 60)
new_songs = []
for _, song in df_latest.iterrows():
    if len(df_prev[df_prev['Title'] == song['Title']]) == 0:
        new_songs.append(song)

if new_songs:
    for song in new_songs[:10]:  # Show max 10
        print(f"#{song['Rank']:2d} {song['Title'][:45]:<45s}")
        print(f"    {song['Artists'][:50]}")
        print()
else:
    print("No new entries")
print()

# Dropped songs
print("=" * 60)
print("📉 DROPPED FROM TOP 50")
print("-" * 60)
dropped = []
for _, song in df_prev.iterrows():
    if len(df_latest[df_latest['Title'] == song['Title']]) == 0:
        dropped.append(song)

if dropped:
    for song in dropped[:10]:  # Show max 10
        print(f"Was #{song['Rank']:2d} {song['Title'][:45]:<45s}")
        print(f"        {song['Artists'][:50]}")
        print()
else:
    print("No songs dropped")
print()

# Biggest movers
print("=" * 60)
print("🚀 BIGGEST GAINERS")
print("-" * 60)
movements = []
for _, latest_song in df_latest.iterrows():
    prev_match = df_prev[df_prev['Title'] == latest_song['Title']]
    if len(prev_match) > 0:
        prev_rank = prev_match.iloc[0]['Rank']
        movement = prev_rank - latest_song['Rank']
        if movement > 0:
            movements.append({
                'Title': latest_song['Title'],
                'Artists': latest_song['Artists'],
                'Current': latest_song['Rank'],
                'Previous': prev_rank,
                'Movement': movement
            })

movements.sort(key=lambda x: x['Movement'], reverse=True)
for m in movements[:5]:
    print(f"#{m['Current']:2d} ↑{m['Movement']:2d} (was #{m['Previous']}) | {m['Title'][:40]}")
    print(f"                      | {m['Artists'][:40]}")
    print()

print("=" * 60)
print(f"📅 Date range: {previous.strftime('%Y-%m-%d')} → {latest.strftime('%Y-%m-%d')}")
print(f"📊 Total songs tracked: {len(df_latest)} (latest), {len(df_prev)} (previous)")
print("=" * 60)
