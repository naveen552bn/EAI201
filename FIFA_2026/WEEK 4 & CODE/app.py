from flask import Flask, render_template, request
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split

# -------------------------
# Helper: find uploaded CSV
# -------------------------
POSSIBLE_DIRS = [Path("."), Path("/content"), Path("/mnt/data"), Path.home()]

def find_file(fname):
    for d in POSSIBLE_DIRS:
        p = d / fname
        if p.exists():
            return str(p)
    return None

# -------------------------
# Config / Files
# -------------------------
PRE_MERGED = find_file("fifa_1930_2022_with_rank.csv")
MATCHES = find_file("WorldCupMatches (1).csv")
RANKING = find_file("fifa_ranking-2024-06-20.csv")

FEATURES = ['Goals_For', 'Goals_Against', 'Goal_Difference', 'Win_Rate', 'FIFA_Rank', 'FIFA_Points']
TARGET = 'Is_Finalist'

app = Flask(__name__)

# -------------------------
# Load / build dataframe
# -------------------------
def build_or_load_df():
    if PRE_MERGED:
        df = pd.read_csv(PRE_MERGED)
        return df
    # try to build from raw files
    if not (MATCHES and RANKING):
        raise FileNotFoundError("Required CSVs not found. Place your CSVs in project folder or /content /mnt/data.")
    matches = pd.read_csv(MATCHES)
    ranking = pd.read_csv(RANKING)
    # --- prepare matches
    matches = matches[['Year', 'Stage', 'Home Team Name', 'Away Team Name',
                       'Home Team Goals', 'Away Team Goals', 'Win conditions']].dropna(
                       subset=['Home Team Name', 'Away Team Name'])
    matches.columns = ['Year', 'Stage', 'Home_Team', 'Away_Team',
                       'Home_Goals', 'Away_Goals', 'Win_Conditions']

    def match_winner(r):
        if r['Home_Goals'] > r['Away_Goals']: return r['Home_Team']
        if r['Home_Goals'] < r['Away_Goals']: return r['Away_Team']
        if 'pen' in str(r['Win_Conditions']).lower():
            # best-effort: check substring
            wc = str(r['Win_Conditions'])
            if r['Home_Team'] in wc: return r['Home_Team']
            if r['Away_Team'] in wc: return r['Away_Team']
        return 'Draw'

    matches['Winner'] = matches.apply(match_winner, axis=1)
    finals = matches[matches['Stage'].str.contains('Final', case=False, na=False)]
    finalist_teams = set(finals['Home_Team']).union(set(finals['Away_Team']))

    home_stats = matches.groupby(['Year', 'Home_Team']).agg({'Home_Goals': 'sum', 'Away_Goals': 'sum'}).reset_index()
    home_stats['Matches_Played'] = matches.groupby(['Year', 'Home_Team']).size().values
    away_stats = matches.groupby(['Year', 'Away_Team']).agg({'Away_Goals': 'sum', 'Home_Goals': 'sum'}).reset_index()
    away_stats['Matches_Played'] = matches.groupby(['Year', 'Away_Team']).size().values

    home_stats.columns = ['Year', 'Team', 'Goals_For', 'Goals_Against', 'Matches_Played']
    away_stats.columns = ['Year', 'Team', 'Goals_For', 'Goals_Against', 'Matches_Played']
    team_stats = pd.concat([home_stats, away_stats]).groupby(['Year', 'Team']).sum().reset_index()

    team_stats['Goal_Difference'] = team_stats['Goals_For'] - team_stats['Goals_Against']
    team_stats['Win_Rate'] = (team_stats['Goals_For'] / (team_stats['Goals_Against'] + 1)).round(3)

    # --- prepare ranking (best-effort mapping)
    ranking.columns = [c.strip().lower() for c in ranking.columns]
    if 'rank_date' in ranking.columns:
        ranking['year'] = pd.to_datetime(ranking['rank_date']).dt.year
    # try common names
    if {'rank', 'country_full', 'total_points', 'confederation', 'year'}.issubset(set(ranking.columns)):
        rsmall = ranking[['rank', 'country_full', 'total_points', 'confederation', 'year']].copy()
        rsmall.columns = ['FIFA_Rank', 'Team', 'FIFA_Points', 'Confederation', 'Year']
    else:
        # fallback: try first 5 cols as rank, country, points
        cols = list(ranking.columns)
        rsmall = ranking[[cols[0], cols[1], cols[2]]].copy()
        rsmall.columns = ['FIFA_Rank', 'Team', 'FIFA_Points']
        if 'year' in ranking.columns:
            rsmall['Year'] = ranking['year']
        else:
            rsmall['Year'] = pd.to_datetime(ranking.iloc[:,0], errors='coerce').dt.year.fillna(method='ffill').astype(int)

    ranking_yearly = rsmall.groupby(['Year', 'Team']).agg({
        'FIFA_Rank': 'mean',
        'FIFA_Points': 'mean',
        **({'Confederation':'first'} if 'Confederation' in rsmall.columns else {})
    }).reset_index()

    merged = pd.merge(team_stats, ranking_yearly, on=['Year', 'Team'], how='left')
    merged['Is_Finalist'] = merged.apply(lambda x: 1 if x['Team'] in finalist_teams and x['Year'] in finals['Year'].values else 0, axis=1)
    merged = merged.dropna(subset=['FIFA_Rank']).reset_index(drop=True)
    return merged

# load dataframe
df = build_or_load_df()

# ensure confederation column exists
if 'Confederation' not in df.columns:
    df['Confederation'] = df.get('confederation', pd.NA)

# drop rows missing required features
df = df.dropna(subset=FEATURES).reset_index(drop=True)

# -------------------------
# Train models
# -------------------------
X = df[FEATURES]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)

rf.fit(X_train, y_train)
gb.fit(X_train_scaled, y_train)

# -------------------------
# Utility to get most recent team row
# -------------------------
def latest_team_row(team_name):
    rows = df[df['Team'] == team_name]
    if rows.empty:
        return None
    return rows.sort_values('Year', ascending=False).iloc[0]

# -------------------------
# Location adjustment logic
# -------------------------
HOST_CONFED = "CONCACAF"
HOST_COUNTRIES = {
    "United States 🇺🇸": "United States",
    "Mexico 🇲🇽": "Mexico",
    "Canada 🇨🇦": "Canada"
}

def apply_location_boost(a_prob, b_prob, team_a_row, team_b_row, location):
    # multiplicative boosts
    host_country = HOST_COUNTRIES.get(location, "")
    # boost if team country matches host country (exact or substring)
    def is_host(team_row):
        # check Team name contains host substring OR Team equals host
        if team_row is None: return False
        team_name = str(team_row['Team']).lower()
        return host_country.lower() in team_name or team_name in host_country.lower()

    if is_host(team_a_row) and not is_host(team_b_row):
        a_prob *= 1.10
    if is_host(team_b_row) and not is_host(team_a_row):
        b_prob *= 1.10

    # confederation boost for CONCACAF teams when location is in host set
    if location in HOST_COUNTRIES:
        try:
            conf_a = str(team_a_row.get('Confederation', "")).upper()
            conf_b = str(team_b_row.get('Confederation', "")).upper()
        except Exception:
            conf_a = conf_b = ""
        if conf_a == HOST_CONFED and conf_b != HOST_CONFED:
            a_prob *= 1.05
        if conf_b == HOST_CONFED and conf_a != HOST_CONFED:
            b_prob *= 1.05

    return a_prob, b_prob

# -------------------------
# Flask routes
# -------------------------
@app.route("/")
def index():
    teams = sorted(df['Team'].unique())
    locations = ["United States 🇺🇸", "Mexico 🇲🇽", "Canada 🇨🇦"]
    return render_template("index.html", teams=teams, locations=locations, result=None)

@app.route("/predict", methods=["POST"])
def predict():
    team_a = request.form.get("team_a")
    team_b = request.form.get("team_b")
    location = request.form.get("location")

    teams = sorted(df['Team'].unique())
    locations = ["United States 🇺🇸", "Mexico 🇲🇽", "Canada 🇨🇦"]

    if not team_a or not team_b or not location:
        return render_template("index.html", teams=teams, locations=locations, result="Please select teams and location.")

    if team_a == team_b:
        return render_template("index.html", teams=teams, locations=locations, result="Please choose two different teams.")

    row_a = latest_team_row(team_a)
    row_b = latest_team_row(team_b)

    if row_a is None or row_b is None:
        return render_template("index.html", teams=teams, locations=locations, result="Team data not found for one or both teams.")

    # prepare features
    feat_a = row_a[FEATURES].to_frame().T
    feat_b = row_b[FEATURES].to_frame().T

    # RF expects unscaled, GB expects scaled
    a_prob_rf = rf.predict_proba(feat_a)[0,1]
    b_prob_rf = rf.predict_proba(feat_b)[0,1]

    a_prob_gb = gb.predict_proba(scaler.transform(feat_a))[0,1]
    b_prob_gb = gb.predict_proba(scaler.transform(feat_b))[0,1]

    # ensemble average
    a_prob = (a_prob_rf + a_prob_gb) / 2.0
    b_prob = (b_prob_rf + b_prob_gb) / 2.0

    # apply location boosts
    a_prob, b_prob = apply_location_boost(a_prob, b_prob, row_a, row_b, location)

    # normalize and convert to percentages
    if (a_prob + b_prob) <= 0:
        return render_template("index.html", teams=teams, locations=locations, result="Numerical error in probabilities.")
    a_pct = round((a_prob / (a_prob + b_prob)) * 100, 1)
    b_pct = round((b_prob / (a_prob + b_prob)) * 100, 1)
    draw_pct = round(max(0.0, 100.0 - (a_pct + b_pct)), 1)

    winner = team_a if a_pct > b_pct else team_b

    result = {
        "winner": winner,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win": a_pct,
        "draw": draw_pct,
        "team_b_win": b_pct,
        "location": location
    }
    return render_template("index.html", teams=teams, locations=locations, result=result)

if __name__ == "__main__":
    app.run(debug=True)
