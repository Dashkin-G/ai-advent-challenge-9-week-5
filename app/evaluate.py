"""Метрики: как нарезан текст, насколько хорошо по нарезке находится ответ
и насколько верен ответ модели.

Эталон — вопросы экзаменационных билетов, в подсказке к которым указан пункт
правил, поэтому место ответа в тексте известно заранее. Чанк считается нужным,
если он и эталонный фрагмент совпадают хотя бы на половину меньшего из двух:
короткий пункт должен лежать в чанке хотя бы наполовину, а кусок длинного
пункта — хотя бы наполовину состоять из него.

Ответ модели сверяется с ожиданием контрольного вопроса: есть ли в нём
ключевые факты и назван ли пункт-источник.
"""
import re
import statistics
from math import comb

from .corpus import Unit, variants

TOP = 5         # столько чанков на следующих днях уйдёт модели в контекст
DEPTH = 10      # глубина выдачи для MRR


def _overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def relevant(chunk: tuple[int, int], golden: list[tuple[int, int]]) -> bool:
    return any(_overlap(chunk, g) * 2 >= min(chunk[1] - chunk[0], g[1] - g[0]) for g in golden)


def marks(chunk: tuple[int, int], golden: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Где внутри чанка лежит эталонный фрагмент — позиции относительно начала чанка."""
    out = []
    for g in golden:
        a, b = max(chunk[0], g[0]), min(chunk[1], g[1])
        if a < b:
            out.append((a - chunk[0], b - chunk[0]))
    return out


# --- форма нарезки -------------------------------------------------------------

def shape(rows: list[dict], units: list[Unit]) -> dict:
    """Размеры чанков, разрезанные пункты, чанки на стыке разделов."""
    sizes = [r["chars"] for r in rows]
    tokens = [r["tokens"] for r in rows]
    numbered = [u for u in units if u.number]
    whole = sum(1 for u in numbered if any(r["start"] <= u.start and u.end <= r["end"] for r in rows))
    cross = sum(1 for r in rows
                if len({(u.part, u.section, u.subsection) for u in units
                        if u.start < r["end"] and u.end > r["start"]}) > 1)
    return {
        "chunks": len(rows),
        "chars_avg": round(statistics.mean(sizes)),
        "chars_min": min(sizes),
        "chars_max": max(sizes),
        "tokens_avg": round(statistics.mean(tokens)),
        "cut_units": len(numbered) - whole,
        "cut_units_pct": round(100 * (len(numbered) - whole) / len(numbered), 1),
        "cross_sections": cross,
        "cross_sections_pct": round(100 * cross / len(rows), 1),
        "sizes": sizes,
    }


# --- качество поиска -----------------------------------------------------------

def judge(hits: list[dict], golden: list[tuple[int, int]]) -> dict:
    """Оценка одной выдачи: на каком месте первый нужный чанк, какая доля эталона
    попала в первые TOP и сколько текста они занимают."""
    rank = next((i + 1 for i, h in enumerate(hits) if relevant((h["start"], h["end"]), golden)), None)
    top = hits[:TOP]
    covered = 0
    for g in golden:
        # Перекрытия чанков не считаем дважды: отмечаем покрытые позиции.
        mask = bytearray(g[1] - g[0])
        for h in top:
            a, b = max(g[0], h["start"]), min(g[1], h["end"])
            if a < b:
                mask[a - g[0]:b - g[0]] = b"\x01" * (b - a)
        covered += sum(mask)
    total = sum(g[1] - g[0] for g in golden)
    return {"rank": rank, "coverage": covered / total, "context": sum(h["chars"] for h in top)}


def paired(a: list[bool], b: list[bool]) -> dict:
    """Две стратегии на одних и тех же вопросах. Важны только вопросы, где они
    разошлись: если перевес — случайность, он делился бы пополам, как монетка.
    p — вероятность получить перевес не меньше этого случайно (точный тест Макнемара)."""
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n, k = only_a + only_b, min(only_a, only_b)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"only": [only_a, only_b], "p": round(p, 4)}


def summary(judged: list[dict]) -> dict:
    n = len(judged)
    ranks = [j["rank"] for j in judged]
    return {
        "questions": n,
        "hit1": round(100 * sum(r == 1 for r in ranks) / n, 1),
        f"hit{TOP}": round(100 * sum(r is not None and r <= TOP for r in ranks) / n, 1),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "coverage": round(100 * statistics.mean(j["coverage"] for j in judged), 1),
        "context": round(statistics.mean(j["context"] for j in judged)),
    }


# --- ответ модели ----------------------------------------------------------------

SUPERSCRIPT = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def facts(answer: str, expected: list[dict]) -> list[dict]:
    """Какие ключевые факты из ожидания есть в ответе и где — для подсветки.
    Факт — регулярное выражение по тексту ответа. У факта с absent наоборот:
    совпадение — ошибка (например, сумма штрафа, которой в Правилах нет)."""
    out = []
    for fact in expected:
        spans = [m.span() for m in re.finditer(fact["re"], answer, re.IGNORECASE)]
        out.append({"ok": not spans if fact.get("absent") else bool(spans), "spans": spans})
    return out


def cited(answer: str, number: str) -> bool:
    """Назван ли в ответе пункт: «10.2» — да, «110.2», «10.20» и «10.2.1» — нет.
    «24.2(1)» модель может написать и как «24.2.1», и как «24.2¹». Номер без точки
    («п. 8 Основных положений») засчитывается только после «п.» или «пункт»."""
    forms = variants(number)
    forms += [re.sub(r"\((\d+)\)$", lambda m: m.group(1).translate(SUPERSCRIPT), f) for f in forms if f.endswith(")")]
    for form in forms:
        pattern = re.escape(form) if "." in form else r"(?:п\.|пункт\w*)\s*" + re.escape(form)
        if re.search(rf"(?<![\d.]){pattern}(?![\d(⁰¹²³⁴⁵⁶⁷⁸⁹]|\.\d)", answer, re.IGNORECASE):
            return True
    return False
