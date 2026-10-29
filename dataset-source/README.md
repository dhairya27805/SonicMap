# Spotify Audio Features EDA

Exploratory data analysis of ~114,000 tracks from Spotify, examining what audio features (danceability, energy, loudness, acousticness, etc.) reveal about genre, popularity, and each other.

## Dataset
- Source: [Spotify Tracks Dataset (Kaggle)](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset)
- ~114,000 tracks across 114 genres
- 21 columns including popularity, duration, explicit flag, and 9 audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo)

## Questions Explored
- Which genres are most/least popular, danceable, long/short?
- Is popularity predictable from audio features?
- What drives loudness, energy, danceability, and valence up or down?
- Does explicit content relate to popularity or speechiness?
- How do acousticness and instrumentalness relate to each other and to other features?

## Key Findings
- **Popularity is largely unpredictable from audio features** — no feature correlates with popularity beyond ±0.1.
- **Loudness is the central hub** connecting most features — strongly tied to energy (+0.76), acousticness (-0.59), and instrumentalness (-0.43).
- **Explicit content has a weak footprint** — slightly higher average popularity (36.5 vs 32.9) but negligible correlation (0.04); moderately tied to speechiness (0.31).
- **Genre is the strongest differentiator** in the dataset — far more structure emerges from genre grouping than from any single correlation.
- **Acousticness and instrumentalness are conceptually distinct** despite sounding similar (correlation only 0.10) — one captures instrument type, the other vocal presence.

Full analysis, charts, and write-up in [`spotifytrial.ipynb`](./spotifytrial.ipynb).

## Tools
Python, pandas, seaborn, matplotlib

## Author
[Anant Dharia](https://github.com/idoyell)
