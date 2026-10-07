import os, re, json, random, sqlite3, datetime as dt
from flask import Flask, request
from questions import QUESTIONS, SKILLS
import ml

app = Flask(__name__)
DB = "careerpilot.db"
KEYWORDS = {"Python": ["python"], "Machine Learning": ["machine learning", "scikit-learn", "tensorflow", "ml", "deep learning"],
            "Web Development": ["html", "css", "javascript", "react", "flask", "web"], "SQL": ["sql", "mysql", "database"],
            "DSA": ["dsa", "data structures", "algorithms", "leetcode"]}
SYSTEM = ("You are CareerPilot AI, a friendly mentor for students. Answer any doubt clearly and simply. "
          "Use short sentences, bullet points, a tiny example when useful, and keep answers under 150 words.")

def db():
    c = sqlite3.connect(DB); c.row_factory = sqlite3.Row; return c

with db() as c:
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, college TEXT, goal TEXT, skills TEXT, resume TEXT);
    CREATE TABLE IF NOT EXISTS attempts(id INTEGER PRIMARY KEY, uid INT, day TEXT, score INT, total INT, detail TEXT, skill_scores TEXT);
    CREATE TABLE IF NOT EXISTS asked(uid INT, qid INT);""")

@app.get("/")
def home(): return app.send_static_file("index.html")

@app.post("/api/login")
def login():
    f, text = request.form, ""
    file = request.files.get("resume")
    if file and file.filename:
        if file.filename.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader
                text = " ".join(p.extract_text() or "" for p in PdfReader(file).pages)
            except Exception: text = ""
        else: text = file.read().decode("utf-8", "ignore")
    skills = set(json.loads(f.get("skills", "[]")))
    for s, keys in KEYWORDS.items():
        if any(re.search(r"\b" + re.escape(k) + r"\b", text.lower()) for k in keys): skills.add(s)
    if not skills: skills = set(SKILLS)
    with db() as c:
        c.execute("""INSERT INTO users(name,email,college,goal,skills,resume) VALUES(?,?,?,?,?,?)
            ON CONFLICT(email) DO UPDATE SET name=excluded.name,college=excluded.college,goal=excluded.goal,skills=excluded.skills,resume=excluded.resume""",
            (f["name"], f["email"].lower(), f.get("college", ""), f.get("goal", ""), json.dumps(sorted(skills)), text[:5000]))
        uid = c.execute("SELECT id FROM users WHERE email=?", (f["email"].lower(),)).fetchone()["id"]
    return {"uid": uid, "skills": sorted(skills)}

@app.get("/api/test/<int:uid>")
def test(uid):
    with db() as c:
        skills = json.loads(c.execute("SELECT skills FROM users WHERE id=?", (uid,)).fetchone()["skills"])
        seen = {r["qid"] for r in c.execute("SELECT qid FROM asked WHERE uid=?", (uid,))}
        pool = [i for i, q in enumerate(QUESTIONS) if q[0] in skills]
        if len(pool) < 10: pool = list(range(len(QUESTIONS)))
        fresh = [i for i in pool if i not in seen]
        ids = random.sample(fresh, min(10, len(fresh)))
        if len(ids) < 10:  # bank exhausted: top up with older questions
            rest = [i for i in pool if i not in ids]
            ids += random.sample(rest, min(10 - len(ids), len(rest)))
        c.executemany("INSERT INTO asked VALUES(?,?)", [(uid, i) for i in ids])
    return {"questions": [{"id": i, "skill": QUESTIONS[i][0], "q": QUESTIONS[i][1], "options": QUESTIONS[i][2]} for i in ids]}

@app.post("/api/submit")
def submit():
    d = request.json; detail, per = [], {}
    for qid, idx in d["answers"].items():
        s, q, o, a = QUESTIONS[int(qid)]
        per.setdefault(s, [0, 0]); per[s][1] += 1; per[s][0] += idx == a
        detail.append({"skill": s, "q": q, "options": o, "chosen": idx, "answer": a, "correct": idx == a})
    score = sum(x["correct"] for x in detail)
    sk = {s: round(10 * c / t, 1) for s, (c, t) in per.items()}
    with db() as c:
        c.execute("INSERT INTO attempts(uid,day,score,total,detail,skill_scores) VALUES(?,?,?,?,?,?)",
                  (d["uid"], dt.date.today().isoformat(), score, len(detail), json.dumps(detail), json.dumps(sk)))
    return {"score": score, "total": len(detail)}

@app.get("/api/dashboard/<int:uid>")
def dashboard(uid):
    with db() as c:
        u = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        rows = c.execute("SELECT * FROM attempts WHERE uid=? ORDER BY id DESC", (uid,)).fetchall()
    days, t = {r["day"] for r in rows}, dt.date.today()
    d = t if t.isoformat() in days else t - dt.timedelta(days=1)
    streak = 0
    while d.isoformat() in days: streak += 1; d -= dt.timedelta(days=1)
    week = []
    for i in range(6, -1, -1):
        day = (t - dt.timedelta(days=i)).isoformat()
        rs = [r for r in rows if r["day"] == day]
        week.append({"day": day[5:], "tests": len(rs), "pct": max([round(100 * r["score"] / r["total"]) for r in rs], default=0)})
    sk = {}
    for r in reversed(rows): sk.update(json.loads(r["skill_scores"]))
    avg = round(sum(100 * r["score"] / r["total"] for r in rows) / len(rows)) if rows else 0
    out = {"name": u["name"], "goal": u["goal"], "streak": streak, "attempts": len(rows), "avg": avg,
           "today_done": t.isoformat() in days, "week": week,
           "last": {"score": rows[0]["score"], "total": rows[0]["total"]} if rows else None,
           "history": [{"day": r["day"], "score": r["score"], "total": r["total"],
                        "wrong": [x for x in json.loads(r["detail"]) if not x["correct"]]} for r in rows[:10]]}
    if rows:
        out.update(ml.analyse([sk.get(s, 0) for s in SKILLS])); out["level"] = ml.level(avg, len(rows))
    return out

@app.post("/api/chat")
def chat():
    msg, key = request.json["message"], os.getenv("ANTHROPIC_API_KEY")
    if not key: return {"reply": ml.offline_answer(msg)}
    try:
        import anthropic
        r = anthropic.Anthropic(api_key=key).messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            max_tokens=700, system=SYSTEM, messages=[{"role": "user", "content": msg}])
        return {"reply": r.content[0].text}
    except Exception as e: return {"reply": f"AI service error: {e}"}

if __name__ == "__main__": app.run(debug=True)
