"""Документы: разовая выгрузка ПДД и билетов, чистка, структура текста.

Правила скачиваются одной страницей с garant.ru и превращаются в Markdown:
`#` — часть (правила, приложения), `##` — раздел, `###` — подраздел, абзац
пункта начинается с его номера («11.4. Обгон запрещен:»). Примечания Гаранта,
история изменений и сноски выбрасываются — в тексте остаются только правила.

Из экзаменационных билетов берутся вопросы без картинок, в подсказке к которым
есть ссылка на пункт. Это эталон: для каждого вопроса известно, где в тексте
лежит ответ, — по нему сравниваются стратегии нарезки.
"""
import json
import re
from dataclasses import dataclass
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from . import config

# Части документа: блок на странице Гаранта → ключ и заголовок в Markdown.
PARTS = {
    "block_1000": ("rules", "Правила дорожного движения Российской Федерации"),
    "block_1100": ("signs", "Приложение 1. Дорожные знаки"),
    "block_1200": ("marking", "Приложение 2. Дорожная разметка"),
    "block_2000": ("admission", "Основные положения по допуску транспортных средств к эксплуатации"),
    "block_2100": ("faults", "Перечень неисправностей, при которых запрещается эксплуатация"),
}
PART_BY_TITLE = {title: key for key, title in PARTS.values()}
# Сноски («В дальнейшем — Правила») и утратившее силу приложение 3 — не правила.
SKIP = {"block_143", "block_1300"}

RAW_HTML = config.RAW_DIR / "pdd.html"
RAW_TICKETS = config.RAW_DIR / "tickets.json"

BROWSER = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}


class CorpusError(RuntimeError):
    """Ошибка выгрузки — уже человеческими словами."""


# --- выгрузка -----------------------------------------------------------------

def _get(client: httpx.Client, url: str) -> httpx.Response:
    """GET с двумя повторами: через прокси соединение изредка рвётся."""
    for _ in range(3):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response
        except httpx.HTTPError as e:
            error = e
    raise CorpusError(f"{url} не отвечает: {error}") from error


def download() -> None:
    """Скачать страницу ПДД и все билеты A/B как есть — в data/raw."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=60, follow_redirects=True, headers=BROWSER) as client:
        RAW_HTML.write_bytes(_get(client, config.PDD_URL).content)
        listing = _get(client, f"https://api.github.com/repos/{config.TICKETS_REPO}"
                               f"/contents/{config.TICKETS_DIR}").json()
        tickets = []
        for item in sorted(listing, key=lambda f: int(re.sub(r"\D", "", f["name"]) or 0)):
            tickets.extend(_get(client, item["download_url"]).json())
    RAW_TICKETS.write_text(json.dumps(tickets, ensure_ascii=False, indent=1), encoding="utf-8")


def downloaded() -> bool:
    return RAW_HTML.exists() and RAW_TICKETS.exists()


# --- HTML → Markdown ----------------------------------------------------------

def _clean(line: str) -> str:
    line = re.sub(r"\*\(\d+(?:\.\d+)?\)", "", line)       # ссылки на сноски: *(1)
    line = re.sub(r"\s+", " ", line.replace("\xa0", " ")).strip()
    return re.sub(r"\s+([.,;:])", r"\1", line)


def _edition(soup: BeautifulSoup) -> str:
    """Дата последнего изменения правил — из блока «С изменениями и дополнениями от»."""
    block = soup.select_one("#block_1000 div.s_52")
    dates = re.findall(r"\d{1,2} \w+ \d{4} г\.", block.get_text(" ") if block else "")
    return dates[-1] if dates else ""


def to_markdown(html: str) -> tuple[str, str]:
    """Текст правил в Markdown и дата редакции."""
    soup = BeautifulSoup(html, "html.parser")
    edition = _edition(soup)
    # История изменений, комментарии Гаранта и список редакций — не текст правил.
    for note in soup.select("div.s_22, div.s_52, div.s_9"):
        note.decompose()
    for sup in soup.find_all("sup"):                       # 2.1.1<sup>1</sup> → 2.1.1(1)
        sup.replace_with(f"({sup.get_text(strip=True)})")
    for br in soup.find_all("br"):
        br.replace_with(" ")

    lines: list[str] = []
    part, title_seen = None, False
    for p in soup.find_all("p"):
        owner = next((a.get("id") for a in p.parents if a.get("id") in PARTS or a.get("id") in SKIP), None)
        if owner is None or owner in SKIP:
            continue
        classes = p.get("class") or []
        right = "text-align:right" in (p.get("style") or "")   # «Приложение 1 к Правилам…»
        if right or not ({"s_1", "s_3", "s_91"} & set(classes)):
            continue
        text = _clean(p.get_text())
        if not text:
            continue
        if owner != part:
            part, title_seen = owner, False
            lines.append(f"# {PARTS[owner][1]}")
        if "s_3" in classes:
            if not title_seen:          # первый заголовок части — её название, он уже есть
                title_seen = True
                continue
            level = "##" if re.match(r"\d+\.\s", text) else "###"
            lines.append(f"{level} {text}")
        # «Утратил силу с 1 января 2020 г. - Постановление…» — след правки, а не правило.
        elif not re.match(r"[\d.()]*\s*(утратил\w* силу|исключен[аоы]?\b)", text, re.IGNORECASE):
            lines.append(text)
    return "\n\n".join(lines) + "\n", edition


# --- структура: части, разделы, пункты с позициями в тексте -------------------

# Номер пункта в начале абзаца: «11.4.», «2.1.1(1).», знаки «1.1», «1.4.1-1.4.6», «5.19.1,»,
# разметка «1.1*»; одноуровневый («7.») — только с точкой, иначе за номер сошло бы «5 лет».
NUMBER = re.compile(r"(\d+(?:\.\d+)+(?:\(\d+\))?(?:-\d+(?:\.\d+)+)?|\d+(?:\(\d+\))?(?=\.\s))\*?[.,]?\s")


@dataclass
class Unit:
    """Пункт правил (или вводный абзац раздела) и его место в тексте."""
    part: str           # ключ части: rules, signs, marking, admission, faults
    section: str        # «11. Обгон, опережение, встречный разъезд»
    subsection: str     # «Регулируемые перекрестки» или ""
    number: str         # «11.4»; "" — вводный текст раздела без номера
    start: int
    end: int
    head: int           # откуда начинаются заголовки перед пунктом (или start)


def structure(text: str) -> list[Unit]:
    """Разобрать Markdown на пункты. Заголовки — границы, в пункты они не входят."""
    units: list[Unit] = []
    part = section = subsection = ""
    head = None                         # начало заголовков, ещё не отданных пункту
    for m in re.finditer(r"[^\n]+", text):
        line = m.group()
        if line.startswith("#"):
            level, title = line.split(" ", 1)
            if level == "#":
                part, section, subsection = PART_BY_TITLE.get(title, title), "", ""
            elif level == "##":
                section, subsection = title, ""
            else:
                subsection = title
            head = m.start() if head is None else head
            continue
        number = NUMBER.match(line)
        continues = units and head is None and units[-1].part == part and units[-1].section == section
        if number or not continues:
            units.append(Unit(part, section, subsection, number.group(1) if number else "",
                              m.start(), m.end(), m.start() if head is None else head))
            head = None
        else:
            units[-1].end = m.end()
    return units


def variants(number: str) -> list[str]:
    """«22.2(1)» и «22.2.1» — один пункт: в билетах так, у Гаранта бывает и так."""
    m = re.fullmatch(r"(.+)\((\d+)\)", number) or re.fullmatch(r"(.+)\.(\d+)", number)
    if not m:
        return [number]
    base, n = m.groups()
    return [number, f"{base}.{n}" if number.endswith(")") else f"{base}({n})"]


def span(units: list[Unit], text: str, ref: dict) -> tuple[int, int] | None:
    """Где в тексте ответ на ссылку {part, number, term}: пункт вместе с подпунктами
    (2.3 → 2.3.1…2.3.4) или, для термина из 1.2, абзац с его определением."""
    for number in variants(ref["number"]):
        for i, u in enumerate(units):
            if u.part != ref["part"] or u.number != number:
                continue
            end = u.end
            for child in units[i + 1:]:
                if child.part != u.part or not child.number.startswith(u.number + "."):
                    break
                end = child.end
            if "term" not in ref:
                return u.start, end
            for m in re.finditer(r"[^\n]+", text[u.start:end]):
                if _term(m.group()) == _plain(ref["term"]):
                    return u.start + m.start(), u.start + m.end()
            return None
    return None


def _plain(s: str) -> str:
    return re.sub(r"[\"«»“”„]", "", s).strip().lower().replace("ё", "е")


def _term(line: str) -> str:
    """Термин, который определяет абзац пункта 1.2: «"Обгон" - опережение…» → «обгон»."""
    m = re.match(r"[\"«“]([^\"»”]+)[\"»”]", line)
    return _plain(m.group(1)) if m else ""


# --- эталон из билетов --------------------------------------------------------

REFERENCE = re.compile(r"ПДД|Правил|Переч|Основн|[Пп]ункт|знак|разметк")


def _refs(tip: str, terms: set[str]) -> list[dict]:
    """Ссылки из подсказки: «(Пункты 8.1, 8.2 ПДД)», «(«Перечень неисправностей» п. 8.3, 9.2)»,
    «(Пункт 1.2 ПДД, термин «Обгон»)», «(… знак 3.20 …)». Источник по традиции стоит
    в последних скобках; номер без пояснения относится к тому же, что и предыдущий."""
    groups = [g for g in re.findall(r"\(((?:[^()]|\(\d+\))*)\)", tip) if REFERENCE.search(g)]
    if not groups:
        return []
    group = groups[-1]
    part = "faults" if "Переч" in group else "admission" if "Основн" in group else "rules"
    refs = []
    for segment in re.split(r"[;,]|\sи\s", group):
        for word, key in (("знак", "signs"), ("разметк", "marking"), ("Переч", "faults"),
                          ("Основн", "admission"), ("ПДД", "rules"), ("Правил", "rules")):
            if word in segment:
                part = key
                break
        numbers = re.findall(r"\d+(?:\.\d+)+(?:\(\d+\))?", segment)
        if part == "admission" and not numbers:
            numbers = re.findall(r"(?:п\.|пункт\w*)\s*(\d+)", segment)
        refs += [{"part": part, "number": n} for n in numbers]
    # Пункт 1.2 — словарь из сотни терминов; ответ — определение того, что названо в скобках.
    if {"part": "rules", "number": "1.2"} in refs:
        named = [t for t in re.findall(r"[«\"]([^»\"]+)[»\"]", group) if _plain(t) in terms]
        if not named:
            return []                   # весь словарь целиком — по нему не проверить
        refs = [r for r in refs if r != {"part": "rules", "number": "1.2"}]
        refs += [{"part": "rules", "number": "1.2", "term": t} for t in named]
    return refs


def questions(tickets: list[dict], text: str) -> list[dict]:
    """Вопросы без картинок, у которых все ссылки нашлись в тексте правил."""
    units = structure(text)
    glossary = next(u for u in units if u.part == "rules" and u.number == "1.2")
    terms = {_term(line) for line in text[glossary.start:glossary.end].splitlines()} - {""}
    kept = []
    for q in tickets:
        if "no_image" not in q.get("image", ""):
            continue
        refs = _refs(q.get("answer_tip", ""), terms)
        if not refs or any(span(units, text, r) is None for r in refs):
            continue
        answers = [a["answer_text"] for a in q["answers"]]
        kept.append({
            "id": q["id"],
            "ticket": q["ticket_number"],
            "n": int(re.sub(r"\D", "", q["title"]) or 0),
            "question": q["question"].strip(),
            "answers": answers,
            "correct": next(i for i, a in enumerate(q["answers"]) if a["is_correct"]),
            "tip": q["answer_tip"].strip(),
            "refs": refs,
        })
    return kept


# --- сборка -------------------------------------------------------------------

def prepare(force: bool = False) -> dict:
    """Скачать (если ещё нет или просят заново) и собрать текст правил и эталон."""
    if force or not downloaded():
        download()
    text, edition = to_markdown(RAW_HTML.read_bytes().decode("cp1251", errors="replace"))
    tickets = json.loads(RAW_TICKETS.read_text(encoding="utf-8"))
    qs = questions(tickets, text)
    for path, content in ((config.DOC_PATH, text),
                          (config.QUESTIONS_PATH, json.dumps(qs, ensure_ascii=False, indent=1))):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
    fetched = datetime.fromtimestamp(RAW_HTML.stat().st_mtime).isoformat(timespec="seconds")
    meta = {"edition": edition, "fetched": fetched, "tickets": len(tickets)}
    (config.DOC_PATH.parent / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return summary()


def summary() -> dict | None:
    """Что лежит на диске — для панели «Документы»."""
    if not config.DOC_PATH.exists():
        return None
    text = config.DOC_PATH.read_text(encoding="utf-8")
    meta_path = config.DOC_PATH.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    units = structure(text)
    parts = []
    for key, title in PARTS.values():
        own = [u for u in units if u.part == key]
        if own:
            parts.append({"key": key, "title": title, "units": sum(1 for u in own if u.number),
                          "chars": sum(u.end - u.start for u in own)})
    qs = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8")) if config.QUESTIONS_PATH.exists() else []
    return {
        "source": config.PDD_URL,
        "edition": meta.get("edition", ""),
        "fetched": meta.get("fetched", ""),
        "chars": len(text),
        "pages": round(len(text) / 1800),          # учётная страница — 1 800 знаков
        "sections": sum(1 for line in text.splitlines() if line.startswith("## ")),
        "units": sum(1 for u in units if u.number),
        "parts": parts,
        "tickets": meta.get("tickets", 0),
        "questions": len(qs),
    }


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=False, indent=1))
