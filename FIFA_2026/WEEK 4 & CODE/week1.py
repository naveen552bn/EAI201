# ===========================================================
# FIFA 2026 Finalist Prediction - Week 1 (Data Preparation)
# -----------------------------------------------------------
# Task: Data Collection, Cleaning, and Feature Engineering
# Output: fifa_1930_2022_with_rank.csv
# ===========================================================

from google.colab import files
print("Upload WorldCupMatches (1).csv")
uploaded = files.upload()

print("Upload fifa_ranking-2024-06-20.csv")
uploaded = files.upload()

# STEP 2: Import required libraries
import pandas as pd
import numpy as np

# STEP 3: Load datasets
matches = pd.read_csv("WorldCupMatches (1).csv")
ranking = pd.read_csv("fifa_ranking-2024-06-20.csv")

print("Data loaded successfully.")
print("Matches Shape:", matches.shape)
print("Ranking Shape:", ranking.shape)

# STEP 4: Clean and prepare the matches dataset
matches = matches[['Year', 'Stage', 'Home Team Name', 'Away Team Name',
                   'Home Team Goals', 'Away Team Goals', 'Win conditions']].dropna(
                   subset=['Home Team Name', 'Away Team Name'])
matches.columns = ['Year', 'Stage', 'Home_Team', 'Away_Team',
                   'Home_Goals', 'Away_Goals', 'Win_Conditions']

def match_result(row):
    if row['Home_Goals'] > row['Away_Goals']:
        return row['Home_Team']
    elif row['Home_Goals'] < row['Away_Goals']:
        return row['Away_Team']
    elif 'pen' in str(row['Win_Conditions']).lower():
        if row['Home_Team'] in row['Win_Conditions']:
            return row['Home_Team']
        else:
            return row['Away_Team']
    else:
        return 'Draw'

matches['Winner'] = matches.apply(match_result, axis=1)

# STEP 5: Identify finalist teams
finals = matches[matches['Stage'].str.contains('Final', case=False, na=False)]
finalist_teams = set(finals['Home_Team']).union(set(finals['Away_Team']))
print("Finalist teams identified:", len(finalist_teams))

# STEP 6: Compute yearly team statistics
home_stats = matches.groupby(['Year', 'Home_Team']).agg({
    'Home_Goals': 'sum', 'Away_Goals': 'sum'}).reset_index()
home_stats['Matches_Played'] = matches.groupby(['Year', 'Home_Team']).size().values

away_stats = matches.groupby(['Year', 'Away_Team']).agg({
    'Away_Goals': 'sum', 'Home_Goals': 'sum'}).reset_index()
away_stats['Matches_Played'] = matches.groupby(['Year', 'Away_Team']).size().values

home_stats.columns = ['Year', 'Team', 'Goals_For', 'Goals_Against', 'Matches_Played']
away_stats.columns = ['Year', 'Team', 'Goals_For', 'Goals_Against', 'Matches_Played']

team_stats = pd.concat([home_stats, away_stats]).groupby(['Year', 'Team']).sum().reset_index()

# STEP 7: Feature engineering
team_stats['Goal_Difference'] = team_stats['Goals_For'] - team_stats['Goals_Against']
team_stats['Win_Rate'] = np.round(team_stats['Goals_For'] / (team_stats['Goals_Against'] + 1), 2)

# STEP 8: Prepare and clean the FIFA ranking dataset
ranking.columns = [c.strip().lower() for c in ranking.columns]
if 'rank_date' in ranking.columns:
    ranking['year'] = pd.to_datetime(ranking['rank_date']).dt.year

ranking = ranking[['rank', 'country_full', 'total_points', 'confederation', 'year']]
ranking.columns = ['FIFA_Rank', 'Team', 'FIFA_Points', 'Confederation', 'Year']

ranking_yearly = ranking.groupby(['Year', 'Team']).agg({
    'FIFA_Rank': 'mean',
    'FIFA_Points': 'mean',
    'Confederation': 'first'
}).reset_index()

# STEP 9: Merge team stats with FIFA ranking
merged = pd.merge(team_stats, ranking_yearly, how='left', on=['Year', 'Team'])

# STEP 10: Label finalist teams (target variable)
merged['Is_Finalist'] = merged.apply(
    lambda x: 1 if x['Team'] in finalist_teams and x['Year'] in finals['Year'].values else 0,
    axis=1
)

# STEP 11: Final cleaning
merged = merged.dropna(subset=['FIFA_Rank'])
merged = merged.sort_values(['Year', 'FIFA_Rank']).reset_index(drop=True)

# STEP 12: Save and download
merged.to_csv("fifa_1930_2022_with_rank.csv", index=False)

print("Final dataset saved as fifa_1930_2022_with_rank.csv")
print("Shape:", merged.shape)
print("\nColumns:", merged.columns.tolist())

from google.colab import files
files.download("fifa_1930_2022_with_rank.csv")
