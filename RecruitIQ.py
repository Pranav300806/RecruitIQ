# ============================================================
# RecruitIQ v2 — Intelligent Candidate Discovery
# WITH EMBEDDINGS — Truly Keyword-Free Matching
# India Runs Hackathon — Full Solution
# ============================================================
# Install: pip install sentence-transformers scikit-learn pandas
# ============================================================

import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util

# ============================================================
# STEP 1: Load embedding model
# This understands MEANING — not just keywords
# ============================================================

print("⏳ Loading embedding model (first time takes ~30 seconds)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!\n")

# ============================================================
# STEP 2: Define the Job Description
# The model will understand what kind of role we need
# ============================================================

JOB_DESCRIPTION = """
We are looking for a Machine Learning Engineer or AI Developer
with strong experience in building intelligent systems,
Python programming, deep learning, neural networks,
and deploying AI models to production environments.
Experience with NLP, data science, and predictive modeling is a plus.
"""

# ============================================================
# STEP 3: Load dataset
# ============================================================

df = pd.read_csv(r'C:\Users\Sri Pranav\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\sample_submission.csv')
print(f"✅ Loaded {len(df)} candidates\n")

# ============================================================
# STEP 4: Parse the reasoning column into features
# ============================================================

def parse_reasoning(text):
    role   = re.match(r'^(.+?) with', text)
    years  = re.search(r'(\d+\.?\d*) yrs', text)
    skills = re.search(r'(\d+) AI core skills', text)
    rate   = re.search(r'response rate (\d+\.?\d*)', text)
    return {
        'role':          role.group(1).strip()  if role    else 'Unknown',
        'years_exp':     float(years.group(1))  if years   else 0.0,
        'ai_skills':     int(skills.group(1))   if skills  else 0,
        'response_rate': float(rate.group(1))   if rate    else 0.0,
    }

parsed = df['reasoning'].apply(parse_reasoning).apply(pd.Series)
df = pd.concat([df[['candidate_id']], parsed], axis=1)

print("📋 Parsed Features Sample:")
print(df.head(5).to_string())
print()

# ============================================================
# STEP 5: EMBEDDING-BASED Role Scoring (The Smart Layer)
# Instead of a dictionary — we measure MEANING similarity
# "Deep Learning Specialist" will score high even if not in dict!
# ============================================================

print("🧠 Computing semantic role similarity using embeddings...")

# Encode the job description once
jd_embedding = model.encode(JOB_DESCRIPTION)

# Encode all candidate roles at once (fast batch processing)
unique_roles = df['role'].unique().tolist()
role_embeddings = model.encode(unique_roles)

# Calculate cosine similarity between each role and the JD
role_similarity = {}
for i, role in enumerate(unique_roles):
    similarity = float(util.cos_sim(role_embeddings[i], jd_embedding))
    # Normalize to 0-1 range (cosine similarity can be -1 to 1)
    normalized = (similarity + 1) / 2
    role_similarity[role] = round(normalized, 4)

# Show what the model learned
print("\n📊 Semantic Role Scores (computed by AI, not hardcoded):")
for role, score in sorted(role_similarity.items(), key=lambda x: -x[1]):
    bar = "█" * int(score * 20)
    print(f"  {role:40} {score:.3f}  {bar}")

df['role_score'] = df['role'].map(role_similarity)
print()

# ============================================================
# STEP 6: Other signal scores
# ============================================================

# Experience — sweet spot 3-10 years
def exp_score(years):
    if years < 1:     return 0.3
    elif years < 3:   return 0.6
    elif years <= 10: return 1.0
    else:             return 0.85

df['exp_score']      = df['years_exp'].apply(exp_score)
df['skills_score']   = df['ai_skills'] / 9.0
df['response_score'] = df['response_rate']

# ============================================================
# STEP 7: Weighted Final Score
# ============================================================

W_ROLE     = 0.40   # Semantic role match — most important
W_SKILLS   = 0.30   # AI skills count
W_EXP      = 0.20   # Experience
W_RESPONSE = 0.10   # Engagement

df['final_score'] = (
    W_ROLE     * df['role_score']     +
    W_SKILLS   * df['skills_score']   +
    W_EXP      * df['exp_score']      +
    W_RESPONSE * df['response_score']
)

# ============================================================
# STEP 8: Rank candidates
# ============================================================

df = df.sort_values('final_score', ascending=False).reset_index(drop=True)
df['rank'] = df.index + 1

# ============================================================
# STEP 9: Explainable Reasoning — WHY they ranked here
# ============================================================

def generate_reasoning(row):
    parts = []

    if row['role_score'] >= 0.70:
        parts.append(f"Strong semantic match for AI/ML role ({row['role']})")
    elif row['role_score'] >= 0.58:
        parts.append(f"Moderate role relevance ({row['role']})")
    else:
        parts.append(f"Low role relevance ({row['role']})")

    if 3 <= row['years_exp'] <= 10:
        parts.append(f"ideal experience ({row['years_exp']} yrs)")
    elif row['years_exp'] > 10:
        parts.append(f"highly experienced ({row['years_exp']} yrs)")
    else:
        parts.append(f"early career ({row['years_exp']} yrs)")

    if row['ai_skills'] >= 8:
        parts.append(f"strong AI skill set ({row['ai_skills']}/9)")
    elif row['ai_skills'] >= 5:
        parts.append(f"moderate AI skills ({row['ai_skills']}/9)")
    else:
        parts.append(f"limited AI skills ({row['ai_skills']}/9)")

    if row['response_rate'] >= 0.70:
        parts.append("highly engaged")
    elif row['response_rate'] >= 0.40:
        parts.append("moderately engaged")
    else:
        parts.append("low engagement")

    return "; ".join(parts) + "."

df['reasoning'] = df.apply(generate_reasoning, axis=1)

# ============================================================
# STEP 10: Save output CSV
# ============================================================

output = df[['candidate_id', 'rank', 'final_score', 'reasoning']].copy()
output.columns = ['candidate_id', 'rank', 'score', 'reasoning']
output['score'] = output['score'].round(4)

output.to_csv(r'C:\Users\Sri Pranav\Downloads\[PUB] India_runs_data_and_ai_challenge\[PUB] India_runs_data_and_ai_challenge\India_runs_data_and_ai_challenge\recruitiq_output_v2.csv', index=False)

# ============================================================
# STEP 11: Print Top 10
# ============================================================

print("=" * 65)
print("      🏆 RECRUITIQ v2 — TOP 10 RANKED CANDIDATES")
print("=" * 65)

for _, row in output.head(10).iterrows():
    print(f"\n#{int(row['rank'])} | {row['candidate_id']} | Score: {row['score']}")
    print(f"   {row['reasoning']}")

print("\n" + "=" * 65)
print(f"\n✅ Full ranked output saved to: recruitiq_output_v2.csv")
print(f"   Total candidates ranked: {len(output)}")

# ============================================================
# STEP 12: Bias Check
# ============================================================

top10_roles = df.head(10)['role'].value_counts()
print(f"\n🔍 Bias Check — Top 10 Role Distribution:")
print(top10_roles.to_string())

if top10_roles.max() > 5:
    print("⚠️  Warning: One role dominates — consider diversity weighting")
else:
    print("✅ Good diversity in top 10!")

# ============================================================
# WHAT MAKES v2 BETTER THAN v1:
# v1 used a hardcoded dictionary — unknown roles got 0.30 default
# v2 uses embeddings — ANY role gets a meaningful similarity score
# "Deep Learning Specialist" → high score even though not in dict
# "NLP Researcher" → high score automatically
# "Chef" → low score automatically
# No manual updates needed when new roles appear in dataset!
# ============================================================