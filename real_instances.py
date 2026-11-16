import csv
import os

from algorithms import LearningModule

DATA = os.path.join(os.path.dirname(__file__), "real_data", "guia_modules_anon.csv")
DATA_TOPICS = os.path.join(os.path.dirname(__file__), "real_data", "guia_topics_anon.csv")
SCORE_DEFINITIONS = ("conclusao", "conteudo", "nps")
TOPIC_SCORE_DEFINITIONS = ("dominio", "engajamento")


def load_rows():
    with open(DATA, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file) if r["mediana_min_real"]]
    return rows


def _score(row, definition):
    if definition == "conclusao":
        return max(0, round(100 * int(row["n_completed"]) / int(row["n_started"])))
    if definition == "conteudo":
        return int(row["n_topics"])
    if definition == "nps":
        return round(10 * float(row["nps_medio"]))
    raise ValueError(f"unknown score definition: {definition}")


def real_instance(score="conclusao", unit_minutes=1, course=None):
    modules = []
    for row in load_rows():
        if course is not None and int(row["curso_idx"]) != course:
            continue
        minutes = float(row["mediana_min_real"])
        time = max(1, round(minutes / unit_minutes))
        name = f"C{row['curso_idx']}M{row['modulo_idx']}"
        modules.append(LearningModule(name, time, _score(row, score)))
    return modules


def weekly_budgets_minutes():
    return {"2h": 120, "5h": 300, "10h": 600, "20h": 1200}


def _topic_score(row, definition):
    if definition == "dominio":
        return round(float(row["score_medio"]))
    if definition == "engajamento":
        return int(row["n_attempts"])
    raise ValueError(f"unknown topic score definition: {definition}")


def real_topic_instance(score="dominio", unit_minutes=1):
    with open(DATA_TOPICS, newline="", encoding="utf-8") as file:
        rows = [r for r in csv.DictReader(file)
                if r["mediana_min_modulo"] and r["score_medio"]]
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
    return modules
