from __future__ import annotations

import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Union

from flask import Flask, flash, g, redirect, render_template, request, session, url_for
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "pdas.db"

app = Flask(__name__)
app.config["SECRET_KEY"] = "pdas-demo-secret-key"

CATEGORIES = ["Academic", "Financial", "Personal", "Work"]
EMOTION_WORDS = {
    "angry",
    "anxious",
    "afraid",
    "upset",
    "frustrated",
    "overwhelmed",
    "panic",
    "stressed",
    "desperate",
    "worried",
    "emotional",
    "fear",
}
DEFAULT_RULES = [
    {
        "name": "High stress (>4)",
        "weight": 2,
        "description": "Add risk when the stress level is greater than 4.",
    },
    {
        "name": "Low confidence (≤2)",
        "weight": 2,
        "description": "Add risk when the confidence level is 2 or below.",
    },
    {
        "name": "Repeated similar decisions",
        "weight": 2,
        "description": "Add risk when the current decision is very similar to a previous decision.",
    },
    {
        "name": "Negative past outcome",
        "weight": 3,
        "description": "Add risk when a similar previous decision recorded a negative outcome.",
    },
]


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        conn = sqlite3.connect(DATABASE)
        conn.row_factory = sqlite3.Row
        g.db = conn
    return g.db


@app.teardown_appcontext
def close_db(_: Any) -> None:
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    db = get_db()
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            confidence_level INTEGER NOT NULL,
            stress_level INTEGER NOT NULL,
            outcome TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS decision_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id INTEGER UNIQUE NOT NULL,
            similarity_score REAL NOT NULL,
            repeated_pattern INTEGER NOT NULL DEFAULT 0,
            negative_past_outcome INTEGER NOT NULL DEFAULT 0,
            emotional_flag INTEGER NOT NULL DEFAULT 0,
            bias_flag INTEGER NOT NULL DEFAULT 0,
            repetition_flag INTEGER NOT NULL DEFAULT 0,
            risk_points INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            explanation TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (decision_id) REFERENCES decisions(id)
        );

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            weight INTEGER NOT NULL,
            description TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            summary TEXT NOT NULL,
            detected_patterns TEXT NOT NULL,
            improvement_suggestions TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
    )
    db.commit()
    seed_defaults(db)


def seed_defaults(db: sqlite3.Connection) -> None:
    now = timestamp()
    admin = db.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone()
    if admin is None:
        db.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            ("admin", generate_password_hash("admin123"), "admin", now),
        )
    for rule in DEFAULT_RULES:
        existing = db.execute("SELECT id FROM rules WHERE name = ?", (rule["name"],)).fetchone()
        if existing is None:
            db.execute(
                "INSERT INTO rules (name, weight, description, updated_at) VALUES (?, ?, ?, ?)",
                (rule["name"], rule["weight"], rule["description"], now),
            )
    db.commit()


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def current_user() -> sqlite3.Row | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    return get_db().execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def login_required() -> sqlite3.Row | None:
    user = current_user()
    if user is None:
        flash("Please log in to continue.", "warning")
        return None
    return user


def analyze_decision(decision: sqlite3.Row, history: list[sqlite3.Row]) -> dict[str, Any]:
    risk_points = 0
    flags: list[str] = []
    explanation_parts: list[str] = []

    similarity_score = 0.0
    repeated_pattern = False
    negative_past_outcome = False

    corpus = [decision["description"]] + [row["description"] for row in history]
    if len(corpus) > 1:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus)
        # Convert sparse matrix to dense for similarity computation
        matrix_dense = matrix.toarray()  # type: ignore
        similarities = cosine_similarity(matrix_dense[0:1], matrix_dense[1:]).flatten()
        if similarities.size:
            similarity_score = float(similarities.max())
            repeated_pattern = similarity_score >= 0.45
            if repeated_pattern:
                risk_points += 2
                flags.append("repetition")
                explanation_parts.append(
                    f"This decision is highly similar to a previous entry (similarity {similarity_score:.2f}), so repetition risk increased by 2."
                )
                similar_index = int(similarities.argmax())
                matched = history[similar_index]
                outcome_text = (matched["outcome"] or "").lower()
                if any(word in outcome_text for word in ["negative", "bad", "failed", "loss", "regret", "poor"]):
                    negative_past_outcome = True
                    risk_points += 3
                    explanation_parts.append(
                        "A similar previous decision had a negative outcome, so risk increased by 3."
                    )
    if decision["stress_level"] > 4:
        risk_points += 2
        flags.append("emotional decision")
        explanation_parts.append("Stress level is above 4, so risk increased by 2.")

    if decision["confidence_level"] <= 2:
        risk_points += 2
        flags.append("bias")
        explanation_parts.append("Confidence level is 2 or below, so risk increased by 2.")

    description_words = {word.strip(".,!?;:").lower() for word in decision["description"].split()}
    emotional_flag = bool(description_words & EMOTION_WORDS)
    if emotional_flag and "emotional decision" not in flags:
        flags.append("emotional decision")
        explanation_parts.append("Emotion-heavy language was detected in the description, so the decision was flagged for emotional influence.")

    risk_level = "Low"
    if 4 <= risk_points <= 6:
        risk_level = "Medium"
    elif risk_points >= 7:
        risk_level = "High"

    if not explanation_parts:
        explanation_parts.append("No exact risk rule was triggered, so the decision remained in the low-risk range.")

    explanation = " ".join(explanation_parts)

    return {
        "similarity_score": similarity_score,
        "repeated_pattern": int(repeated_pattern),
        "negative_past_outcome": int(negative_past_outcome),
        "emotional_flag": int(emotional_flag),
        "bias_flag": int(decision["confidence_level"] <= 2),
        "repetition_flag": int(repeated_pattern),
        "risk_points": risk_points,
        "risk_level": risk_level,
        "flags": sorted(set(flags)),
        "explanation": explanation,
    }


def build_audit_report(user_id: int) -> None:
    db = get_db()
    decisions = db.execute(
        """
        SELECT d.*, da.risk_level, da.risk_points, da.explanation, da.repeated_pattern, da.emotional_flag
        FROM decisions d
        JOIN decision_analysis da ON da.decision_id = d.id
        WHERE d.user_id = ?
        ORDER BY d.created_at DESC
        """,
        (user_id,),
    ).fetchall()

    if not decisions:
        return

    risk_counter = Counter(row["risk_level"] for row in decisions)
    repeated_count = sum(row["repeated_pattern"] for row in decisions)
    emotional_count = sum(row["emotional_flag"] for row in decisions)
    common_category = Counter(row["category"] for row in decisions).most_common(1)[0][0]
    avg_risk = sum(row["risk_points"] for row in decisions) / len(decisions)

    summary = (
        f"{len(decisions)} decisions analyzed. Average risk score is {avg_risk:.1f}. "
        f"Low: {risk_counter.get('Low', 0)}, Medium: {risk_counter.get('Medium', 0)}, High: {risk_counter.get('High', 0)}."
    )
    patterns = (
        f"Most common category: {common_category}. Repeated-pattern flags: {repeated_count}. "
        f"Emotional-language flags: {emotional_count}."
    )
    suggestions = (
        "Review high-stress decisions before acting, compare against earlier similar decisions, "
        "and record outcomes to improve future audit quality."
    )

    db.execute("DELETE FROM audit_reports WHERE user_id = ?", (user_id,))
    db.execute(
        "INSERT INTO audit_reports (user_id, summary, detected_patterns, improvement_suggestions, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, summary, patterns, suggestions, timestamp()),
    )
    db.commit()


@app.before_request
def bootstrap() -> None:
    init_db()


@app.context_processor
def inject_context() -> dict[str, Any]:
    return {"current_user": current_user(), "categories": CATEGORIES}


@app.route("/")
def index() -> Any:
    user = current_user()
    if user:
        return redirect(url_for("dashboard"))
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register() -> Any:
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        db = get_db()
        try:
            db.execute(
                "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'user', ?)",
                (username, generate_password_hash(password), timestamp()),
            )
            db.commit()
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
            return render_template("register.html")
        flash("Account created. Please log in.", "success")
        return redirect(url_for("login"))
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login() -> Any:
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        user = get_db().execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user_id"] = user["id"]
            flash("Logged in successfully.", "success")
            return redirect(url_for("dashboard"))
        flash("Invalid credentials.", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout() -> Any:
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard() -> Any:
    user = login_required()
    if user is None:
        return redirect(url_for("login"))
    db = get_db()
    decisions = db.execute(
        """
        SELECT d.*, da.risk_level, da.risk_points, da.explanation
        FROM decisions d
        LEFT JOIN decision_analysis da ON da.decision_id = d.id
        WHERE d.user_id = ?
        ORDER BY d.created_at DESC
        """,
        (user["id"],),
    ).fetchall()
    report = db.execute(
        "SELECT * FROM audit_reports WHERE user_id = ? ORDER BY created_at DESC LIMIT 1",
        (user["id"],),
    ).fetchone()
    return render_template("dashboard.html", decisions=decisions, report=report)


@app.route("/decisions/new", methods=["GET", "POST"])
def add_decision() -> Any:
    user = login_required()
    if user is None:
        return redirect(url_for("login"))

    if request.method == "POST":
        db = get_db()
        payload = {
            "title": request.form["title"].strip(),
            "category": request.form["category"],
            "description": request.form["description"].strip(),
            "confidence_level": int(request.form["confidence_level"]),
            "stress_level": int(request.form["stress_level"]),
            "outcome": request.form.get("outcome", "").strip() or None,
        }
        cursor = db.execute(
            """
            INSERT INTO decisions (user_id, title, category, description, confidence_level, stress_level, outcome, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                payload["title"],
                payload["category"],
                payload["description"],
                payload["confidence_level"],
                payload["stress_level"],
                payload["outcome"],
                timestamp(),
            ),
        )
        decision_id = cursor.lastrowid
        decision = db.execute("SELECT * FROM decisions WHERE id = ?", (decision_id,)).fetchone()
        history = db.execute(
            "SELECT * FROM decisions WHERE user_id = ? AND id != ? ORDER BY created_at DESC",
            (user["id"], decision_id),
        ).fetchall()
        analysis = analyze_decision(decision, history)
        db.execute(
            """
            INSERT INTO decision_analysis (
                decision_id, similarity_score, repeated_pattern, negative_past_outcome, emotional_flag,
                bias_flag, repetition_flag, risk_points, risk_level, explanation, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                analysis["similarity_score"],
                analysis["repeated_pattern"],
                analysis["negative_past_outcome"],
                analysis["emotional_flag"],
                analysis["bias_flag"],
                analysis["repetition_flag"],
                analysis["risk_points"],
                analysis["risk_level"],
                analysis["explanation"],
                timestamp(),
            ),
        )
        db.commit()
        build_audit_report(user["id"])
        flash(f"Decision analyzed successfully. Risk: {analysis['risk_level']}.", "success")
        return redirect(url_for("dashboard"))

    return render_template("add_decision.html")


@app.route("/admin/rules", methods=["GET", "POST"])
def admin_rules() -> Any:
    user = login_required()
    if user is None:
        return redirect(url_for("login"))
    if user["role"] != "admin":
        flash("Admin access required.", "danger")
        return redirect(url_for("dashboard"))

    db = get_db()
    if request.method == "POST":
        for rule in db.execute("SELECT * FROM rules ORDER BY id").fetchall():
            description = request.form.get(f"description_{rule['id']}", rule["description"]).strip()
            is_active = 1 if request.form.get(f"active_{rule['id']}") == "on" else 0
            db.execute(
                "UPDATE rules SET description = ?, is_active = ?, updated_at = ? WHERE id = ?",
                (description, is_active, timestamp(), rule["id"]),
            )
        db.commit()
        flash("Rule notes updated.", "success")
        return redirect(url_for("admin_rules"))

    rules = db.execute("SELECT * FROM rules ORDER BY id").fetchall()
    analytics = db.execute(
        """
        SELECT COUNT(*) AS decision_count,
               SUM(CASE WHEN risk_level = 'High' THEN 1 ELSE 0 END) AS high_risk_count,
               AVG(risk_points) AS average_risk
        FROM decision_analysis
        """
    ).fetchone()
    return render_template("admin_rules.html", rules=rules, analytics=analytics)


if __name__ == "__main__":
    app.run(debug=True)
