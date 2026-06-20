# 🧠 RecruitIQ

RecruitIQ is an AI-powered candidate discovery system designed to make hiring smarter and more meaningful. Instead of relying on exact keyword matches, it understands the context behind a candidate's role and skills using semantic embeddings. This helps recruiters discover talented candidates who might otherwise be missed by traditional ATS systems.

For example, a candidate with the title **"AI Developer"** may be a great fit for a **"Machine Learning Engineer"** role, even though the job titles are different. RecruitIQ identifies these meaningful connections by understanding the semantics of the text rather than just matching keywords.

The system ranks candidates using four important factors:

* **Semantic Role Match (40%)** – How closely the candidate's role aligns with the job description.
* **AI Skills Score (30%)** – Relevant AI and Machine Learning skills possessed by the candidate.
* **Experience Score (20%)** – Preference for candidates with balanced industry experience.
* **Engagement Score (10%)** – Measures candidate activity and responsiveness.

In addition to ranking candidates, RecruitIQ also provides short explanations for every recommendation, making the results easy to understand and transparent for recruiters. It also includes a simple bias check to identify whether a single role category dominates the shortlisted candidates.

## Tech Stack

* Python
* Sentence Transformers (`all-MiniLM-L6-v2`)
* scikit-learn
* pandas

## Running the Project

```bash
pip install sentence-transformers scikit-learn pandas
python RecruitIQ.py
```

The program generates a CSV file containing candidate rankings, scores, and explanations.

## Future Improvements

* Increase the importance of skills over job titles.
* Distinguish between must-have and optional job requirements.
* Add fairness-aware ranking techniques.
* Build an interactive dashboard for recruiters.

---

Built with passion by a 4-member team for the **India Runs – Data & AI Challenge 2026 (Hack2skill × Redrob AI)**.

