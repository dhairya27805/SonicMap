# SonicMap — Global Dataset Information

## Dataset overview
- Dataset name: Original SonicMap music dataset
- Approximate number of records: 20,000
- Purpose: Powers the existing Global Music Map.
- The dataset must remain unchanged.

## Audio features

The dataset uses these nine audio features:

1. danceability
2. energy
3. loudness
4. speechiness
5. acousticness
6. instrumentalness
7. liveness
8. valence
9. tempo

## Other metadata

The dataset also contains track metadata such as:

- track_id
- track_name
- artists
- genre
- popularity

## Existing ML pipeline

The Global Music Map uses:
- Feature preprocessing and scaling.
- KMeans clustering with eight existing clusters.
- PCA for two-dimensional visualization.
- Cosine similarity for finding similar songs.

## Important development instructions

- Do not recreate or replace this dataset.
- Do not generate a synthetic replacement.
- Do not change the existing global clustering model without approval.
- Do not require the CSV to be uploaded to Claude.
- Preserve existing CSV loading paths and processing logic.
- The dataset will remain available locally during development and deployment.
- Build personalized user music processing independently of this dataset.
- Make the dataset path configurable through environment variables or application settings.
- Include clear instructions for placing the original CSV in the correct folder.