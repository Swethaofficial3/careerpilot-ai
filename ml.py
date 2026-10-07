import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from questions import SKILLS

# Ideal skill level (0-10) for each career, order = SKILLS
PROFILES = {
 "Data Scientist":[8,9,3,7,5], "ML Engineer":[9,8,5,5,7],
 "Full-Stack Developer":[6,2,9,6,7], "Data Analyst":[6,4,3,9,4],
 "Software Engineer":[7,2,6,6,9]}

ROADMAPS = {
 "Data Scientist":[("Python + Statistics","NumPy, Pandas, probability, hypothesis testing"),("Data Analysis & SQL","EDA, Matplotlib, joins, window functions"),("Machine Learning","Regression, Random Forest, KNN, Naive Bayes, K-Means"),("Projects","Kaggle project, end-to-end prediction app"),("Deployment & Portfolio","Flask/Streamlit app, GitHub, resume, apply")],
 "ML Engineer":[("Strong Python + DSA","OOP, algorithms, clean code"),("ML Foundations","scikit-learn, model evaluation, feature engineering"),("Deep Learning","PyTorch/TensorFlow, CNN, NLP basics"),("MLOps","Docker, REST APIs, model deployment"),("Portfolio","2 deployed ML products, internships")],
 "Full-Stack Developer":[("Web Basics","HTML, CSS, JavaScript"),("Frontend Framework","React, responsive design"),("Backend + Database","Flask/Node, REST APIs, SQL"),("Full Projects","Auth, CRUD app, deployment"),("Portfolio","GitHub, live demo links, apply")],
 "Data Analyst":[("Excel + SQL","Queries, joins, aggregations"),("Python for Data","Pandas, cleaning, visualisation"),("BI Tools","Power BI / Tableau dashboards"),("Case Studies","3 business-problem analyses"),("Portfolio","Dashboard portfolio, resume, apply")],
 "Software Engineer":[("DSA Core","Arrays, strings, recursion, trees, graphs"),("Problem Solving","150 LeetCode problems, patterns"),("CS Fundamentals","OS, DBMS, networks, OOP"),("Projects","One backend + one full-stack project"),("Interview Prep","Mock interviews, resume, apply")]}

rng = np.random.default_rng(42)
X, y = [], []
for career, p in PROFILES.items():
    for _ in range(200):
        X.append(np.clip(np.array(p) + rng.normal(0, 1.6, len(p)), 0, 10)); y.append(career)
X = np.array(X)
MODELS = {"Random Forest": RandomForestClassifier(100, random_state=1),
          "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=1),
          "KNN": KNeighborsClassifier(7), "Naive Bayes": GaussianNB()}
for m in MODELS.values(): m.fit(X, y)

# K-Means groups learners by (average score, attempts) into 3 levels
att = rng.integers(1, 30, 300)
pct = np.clip(25 + att * 2 + rng.normal(0, 12, 300), 0, 100)
Z = np.c_[pct / 100, att / 30]

def kmeans(Z, k=3, iters=50):
    r = np.random.default_rng(1)
    C = Z[r.choice(len(Z), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((Z[:, None] - C[None]) ** 2).sum(2), axis=1)
        C = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return C

CENTERS = kmeans(Z)
ORDER = np.argsort(CENTERS.sum(axis=1))
LEVELS = {int(c): n for c, n in zip(ORDER, ["Beginner", "Intermediate", "Advanced"])}

def level(avg, attempts):
    p = np.array([avg / 100, min(attempts, 30) / 30])
    return LEVELS[int(np.argmin(((CENTERS - p) ** 2).sum(1)))]
def analyse(vec):
    v = [vec]
    rf = MODELS["Random Forest"]
    probs = sorted(zip(rf.classes_, rf.predict_proba(v)[0]), key=lambda t: -t[1])[:3]
    top = probs[0][0]
    gap = [{"skill": s, "you": vec[i], "need": PROFILES[top][i]} for i, s in enumerate(SKILLS)]
    return {"top": top,
            "careers": [{"name": n, "prob": round(float(p) * 100)} for n, p in probs],
            "votes": {k: m.predict(v)[0] for k, m in MODELS.items()},
            "gap": gap, "roadmap": ROADMAPS[top]}

TIPS = {
 "random forest": "**Random Forest** = many Decision Trees voting together.\n- Each tree sees random data\n- Final answer = majority vote\n- Less overfitting than one tree",
 "decision tree": "**Decision Tree** asks yes/no questions step by step until it reaches an answer, like a flowchart.",
 "knn": "**KNN** looks at the K closest data points and copies the most common label among them.",
 "naive bayes": "**Naive Bayes** uses probability (Bayes' theorem) and assumes features are independent. Fast and great for text.",
 "k-means": "**K-Means** groups similar data into K clusters by moving centers until groups are stable.",
 "roadmap": "Open your dashboard: your **career roadmap** is built from your latest test results."}

def offline_answer(msg):
    m = msg.lower()
    for k, v in TIPS.items():
        if k in m or k.replace("-", "") in m: return v
    return ("I can explain Random Forest, Decision Tree, KNN, Naive Bayes, K-Means and your roadmap offline.\n"
            "For answers to **any** doubt, set the ANTHROPIC_API_KEY environment variable (see README steps).")
