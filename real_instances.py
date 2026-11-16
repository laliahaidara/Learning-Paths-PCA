"""Real, anonymized instances built from the adaptive-learning platform data.

The CSV files in real_data/ hold the modules (12) and topics (78) of the
courses studied in the paper, with the median real study time in minutes.
Here they become knapsack instances: time = minutes, score = one of several
definitions of "value" (the report compares them).
"""

import csv
import os

from algorithms import LearningModule, validate_instance

DATA = os.path.join(os.path.dirname(__file__), "real_data", "guia_modules_anon.csv")
DATA_TOPICS = os.path.join(os.path.dirname(__file__), "real_data", "guia_topics_anon.csv")
# Score definitions: module level = completion rate / number of topics / mean NPS;
# topic level = mean mastery score / number of attempts (engagement).
SCORE_DEFINITIONS = ("conclusao", "conteudo", "nps")
TOPIC_SCORE_DEFINITIONS = ("dominio", "engajamento")


def load_rows():
    """Module rows of the CSV that have a measured median time."""
    with open(DATA, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["mediana_min_real"]]
    return rows


def _score(row, definition):
    """Integer score of a module row under one of the SCORE_DEFINITIONS."""
    if definition == "conclusao":  # completion rate, in percent
        return max(0, round(100 * int(row["n_completed"]) / int(row["n_started"])))
    if definition == "conteudo":  # amount of content: number of topics
        return int(row["n_topics"])
    if definition == "nps":  # mean NPS, scaled by 10 to keep integers
        return round(10 * float(row["nps_medio"]))
    raise ValueError(f"unknown score definition: {definition}")


def real_instance(score="conclusao", unit_minutes=1, course=None):
    """Module-level instance (12 modules): time = median minutes / ``unit_minutes``.

    ``unit_minutes=60`` gives times in hours; ``course`` keeps a single course.
    """
    modules = []
    for row in load_rows():
        if course is not None and int(row["curso_idx"]) != course:
            continue
        minutes = float(row["mediana_min_real"])
        time = max(1, round(minutes / unit_minutes))  # a module always costs at least 1 unit
        name = f"C{row['curso_idx']}M{row['modulo_idx']}"
        modules.append(LearningModule(name, time, _score(row, score)))
    validate_instance(modules, 0)
    return modules


def weekly_budgets_minutes():
    """Weekly study budgets (in minutes) used in the report's scenarios."""
    return {"2h": 120, "5h": 300, "10h": 600, "20h": 1200}


def _topic_score(row, definition):
    """Integer score of a topic row under one of the TOPIC_SCORE_DEFINITIONS."""
    if definition == "dominio":  # mean mastery score
        return round(float(row["score_medio"]))
    if definition == "engajamento":  # number of attempts
        return int(row["n_attempts"])
    raise ValueError(f"unknown topic score definition: {definition}")


def real_topic_instance(score="dominio", unit_minutes=1):
    """Topic-level instance (78 topics).

    The data only has the median time of a whole module, so each topic gets
    the share of it proportional to its number of activities (equal shares if
    the module has no recorded activity).
    """
    with open(DATA_TOPICS, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["mediana_min_modulo"] and r["score_medio"]]
    by_module = {}
    for row in rows:
        by_module.setdefault((row["curso_idx"], row["modulo_idx"]), []).append(row)
    modules = []
    for (course, module), topics in by_module.items():
        total_minutes = float(topics[0]["mediana_min_modulo"])
        total_activities = sum(int(t["n_activities"] or 0) for t in topics)
        for row in topics:
            if total_activities > 0:
                share = int(row["n_activities"] or 0) / total_activities
            else:
                share = 1 / len(topics)
            time = max(1, round(total_minutes * share / unit_minutes))
            name = f"C{course}M{module}T{row['topico_idx']}"
            modules.append(LearningModule(name, time, _topic_score(row, score)))
    validate_instance(modules, 0)
    return modules
