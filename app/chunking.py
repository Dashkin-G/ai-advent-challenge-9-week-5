"""Две стратегии нарезки одного и того же текста.

fixed — окно фиксированного размера с перекрытием. Режет где придётся: посреди
пункта, на стыке разделов. Слова не рвёт — граница сдвигается к пробелу.

structure — режет только по границам пунктов и никогда не пересекает заголовок.
Мелкие пункты одного раздела собираются в чанк того же размера; пункт длиннее
полутора размеров делится по абзацам, абзац-великан — по предложениям.

Размер у обеих стратегий общий: иначе сравнивали бы размер, а не способ нарезки.
Чанк — непрерывный кусок текста [start, end), поэтому метаданные (часть,
раздел, пункты) считаются по позициям — одинаково для обеих стратегий.
"""
import re
from dataclasses import dataclass

from . import config
from .corpus import PARTS, Unit

STRATEGIES = {"fixed": "Фиксированный размер", "structure": "По структуре"}
TITLES = {key: title for key, title in PARTS.values()}


@dataclass
class Chunk:
    chunk_id: str
    strategy: str
    source: str
    title: str          # часть документа: «Правила дорожного движения…», «Приложение 1…»
    section: str        # «13. Проезд перекрестков › Регулируемые перекрестки»
    points: list[str]   # пункты, которые попали в чанк целиком или частью
    start: int
    end: int
    text: str


def _chunks(strategy: str, spans: list[tuple[int, int]], text: str, units: list[Unit]) -> list[Chunk]:
    """Метаданные по позициям: раздел — у первого пункта в чанке, пункты — все задетые."""
    chunks = []
    for i, (start, end) in enumerate(spans):
        touched = [u for u in units if u.start < end and u.end > start]
        first = touched[0] if touched else next(u for u in units if u.end > start)
        section = first.section + (f" › {first.subsection}" if first.subsection else "")
        chunks.append(Chunk(
            chunk_id=f"{strategy}-{i:04d}", strategy=strategy, source=config.PDD_URL,
            title=TITLES.get(first.part, first.part), section=section,
            points=[u.number for u in touched if u.number],
            start=start, end=end, text=text[start:end],
        ))
    return chunks


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


# --- фиксированный размер ------------------------------------------------------

def fixed(text: str, units: list[Unit], size: int, overlap: int) -> list[Chunk]:
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):                         # конец окна — на последний пробел в окне
            space = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            end = space if space > start + size // 2 else end
        spans.append(_trim(text, start, end))
        if end >= len(text):
            break
        nxt = max(end - overlap, start + 1)         # следующее окно — с перекрытием
        space = text.find(" ", nxt, end)            # и тоже с начала слова
        start = space + 1 if space != -1 else nxt
    return _chunks("fixed", [s for s in spans if s[1] > s[0]], text, units)


# --- по структуре --------------------------------------------------------------

def _pieces(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Длинный пункт → абзацы; абзац длиннее предела → предложения."""
    out = []
    for m in re.finditer(r"[^\n]+", text[start:end]):
        a, b = start + m.start(), start + m.end()
        if b - a <= limit:
            out.append((a, b))
            continue
        for s in re.finditer(r".+?(?:[.;:](?=\s)|$)", text[a:b]):
            out.append(_trim(text, a + s.start(), a + s.end()))
    return out


def structure(text: str, units: list[Unit], size: int) -> list[Chunk]:
    limit = size * 3 // 2
    # Атом — кусок, внутри которого резать нельзя: пункт целиком или абзац длинного пункта.
    # new — перед атомом стоит заголовок: чанк обязан начаться заново.
    atoms: list[tuple[int, int, bool]] = []
    for u in units:
        if u.end - u.head <= limit:
            atoms.append((u.head, u.end, u.head < u.start))
            continue
        pieces = _pieces(text, u.start, u.end, limit)
        atoms.append((u.head, pieces[0][1], u.head < u.start))
        atoms += [(a, b, False) for a, b in pieces[1:]]

    spans: list[tuple[int, int]] = []
    for start, end, new in atoms:
        if spans and not new and end - spans[-1][0] <= size:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return _chunks("structure", spans, text, units)


def split(strategy: str, text: str, units: list[Unit]) -> list[Chunk]:
    if strategy == "fixed":
        return fixed(text, units, config.CHUNK_SIZE, config.FIXED_OVERLAP)
    return structure(text, units, config.CHUNK_SIZE)
