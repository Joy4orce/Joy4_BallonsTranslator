"""Local LLM translator (OpenAI-compatible /v1/chat/completions endpoint).

Works with any local server that speaks OpenAI Chat Completions:
koboldcpp, LM Studio, Ollama (OpenAI-compat mode at /v1), llama.cpp server,
vLLM, text-generation-webui, etc. Configured by base_url + model + prompt;
no key required by most local backends (a benign 'sk-local' is sent so the
OpenAI-side wire format stays valid).

Sends the page's text blocks as a JSON array and expects a JSON array of the
same length back. concate_text = False so BaseTranslator hands us the raw list.
"""

import json
import re
import time
from typing import Dict, List

import requests

from .base import BaseTranslator, register_translator


_DEFAULT_SYSTEM_PROMPT = (
    "You are a professional Japanese-to-Korean translator for manga, webtoons, and "
    "Japanese web images. Produce natural, idiomatic Korean that reads fluently to a "
    "native speaker, never stiff or word-for-word. Match each speaker's tone and register "
    "(casual, rough, cute, polite, archaic) and keep lines short and punchy like real "
    "speech-bubble text. Render onomatopoeia and sound effects as natural Korean "
    "equivalents instead of transliterating them. Adapt Japanese honorifics and name "
    "suffixes (-san, -chan, -kun, -senpai) to natural Korean usage that fits the "
    "relationship. Preserve emotional nuance, emphasis, and interjections. Do not censor, "
    "tone down, or add explanations; translate crude or adult language faithfully. "
    "Translate every word into Korean, including shouts, interjections, and swearing "
    "(for example おい to 야 or 어이, くそ to 젠장 or 제기랄); never leave any Japanese text "
    "untranslated and never write it in romaji. "
    "For personal names, transliterate by their Japanese reading (e.g. 木島 to 키지마, "
    "智美 to 사토미), never by the literal meaning of the characters. For Japanese "
    "abbreviations and net-slang, output the real Korean meaning, not a phonetic spelling "
    "(e.g. パワハラ to 갑질, セクハラ to 성희롱, 彼氏 to 남친, キツい to 까칠하다 or 드세다). "
    "Keep each line and each bullet marker on its own line exactly as in the source; "
    "never merge or reflow lines."
)


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*```(?:json|JSON)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned)
    return cleaned.strip()


_THINK_TAGS = r"(?:thinking|think|thought|reasoning)"
_RE_THINK_BLOCK = re.compile(
    rf"<\s*{_THINK_TAGS}\s*>.*?<\s*/\s*{_THINK_TAGS}\s*>\s*",
    re.IGNORECASE | re.DOTALL,
)

# A double-quoted chunk, honouring backslash escapes.
_RE_QUOTED_CHUNK = re.compile(r'"((?:[^"\\]|\\.)*)"', re.DOTALL)


def _parse_loose_quoted_list(text: str) -> List[str]:
    """Salvage quoted chunks that aren't valid JSON at all.

    Models often put the separating comma INSIDE the quote and then drop it
    between entries — `"응," "야!", "진짜!"` — which no amount of bracket
    wrapping makes parseable. Recover the entries only when the WHOLE
    response is quoted chunks joined by nothing but whitespace and commas,
    so a prose reply still fails loudly instead of being mined for quotes.

    The chunk contents are kept verbatim: a comma inside the quotes may be
    real punctuation, and guessing which ones were meant as separators would
    silently corrupt the translation.
    """
    chunks = list(_RE_QUOTED_CHUNK.finditer(text))
    if not chunks:
        return None
    pos = 0
    for m in chunks:
        if text[pos:m.start()].strip(' \t\r\n,'):
            return None
        pos = m.end()
    if text[pos:].strip(' \t\r\n,'):
        return None

    out = []
    for m in chunks:
        body = m.group(1)
        try:
            out.append(str(json.loads('"' + body + '"')))
        except json.JSONDecodeError:
            out.append(body)
    return out


# Typographic double quotes a model may use in place of ASCII ones.
_CURLY_QUOTES = '“”„‟＂'
_ANY_QUOTE = '"' + _CURLY_QUOTES
# closing quote, comma, opening quote — the boundary between two entries
_RE_ENTRY_BOUNDARY = re.compile(rf'[{_ANY_QUOTE}]\s*,\s*[{_ANY_QUOTE}]')


def _parse_curly_quoted_list(text: str) -> List[str]:
    """Salvage an array whose strings are delimited with typographic quotes.

    Models writing Korean or Japanese reach for `“…”` out of habit, often mixed
    with ASCII on the closing side: `[“a", “b”, “c"]`. The structure is right,
    the delimiters are not JSON. Normalising every curly quote to `"` would
    break as soon as one appears INSIDE an entry as real punctuation, so the
    entries are cut at the boundaries instead — a quote, a comma, a quote — and
    whatever sits between two boundaries is kept verbatim.

    Only attempted when the text is quote-to-quote once the brackets are off
    and actually contains a curly quote; prose with a quotation in it does not
    qualify. An entry that itself contains `", "` over-splits, and the caller's
    length check then rejects the result rather than accepting a wrong one.
    """
    s = text.strip()
    if s.startswith('['):
        s = s[1:]
    if s.endswith(']'):
        s = s[:-1]
    s = s.strip().rstrip(',').strip()
    if len(s) < 2 or s[0] not in _ANY_QUOTE or s[-1] not in _ANY_QUOTE:
        return None
    if not any(q in s for q in _CURLY_QUOTES):
        return None
    return [part.replace('\\"', '"').replace('\\n', '\n')
            for part in _RE_ENTRY_BOUNDARY.split(s[1:-1])]


def _parse_json_array(text: str) -> List[str]:
    cleaned = _RE_THINK_BLOCK.sub("", text)
    cleaned = _strip_markdown_fences(cleaned)
    # 1) Straight parse — the well-behaved case.
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(x) for x in data]
        if isinstance(data, dict):
            for key in ("translations", "lines", "result", "items"):
                if key in data and isinstance(data[key], list):
                    return [str(x) for x in data[key]]
        if isinstance(data, str):
            # Single-entry batch: the model answered `"translated"` with no
            # array around it. The caller's length check validates this.
            return [data]
    except json.JSONDecodeError:
        pass
    # 2) Embedded array — the model wrapped the array in prose.
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    # 3) Missing outer brackets — a very common model mistake: emits
    #    `"a", "b", "c"` instead of `["a", "b", "c"]`. Detect a leading
    #    quote followed by a quoted-comma-quoted pattern and wrap.
    stripped = cleaned.strip().rstrip(',').strip()
    if stripped.startswith('"') and stripped.endswith('"') \
            and re.search(r'"\s*,\s*"', stripped):
        try:
            data = json.loads('[' + stripped + ']')
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    # 4) Not JSON at all — bare quoted chunks, separators misplaced or absent.
    loose = _parse_loose_quoted_list(cleaned)
    if loose is not None:
        return loose
    # 5) Right structure, typographic quotes as the string delimiters.
    return _parse_curly_quoted_list(cleaned)


@register_translator('Local LLM')
class LocalLLMTranslator(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'base url': 'http://localhost:5001/v1',
        'model': 'local',
        'api key': '',
        'system prompt': {
            'type': 'editor',
            'content': _DEFAULT_SYSTEM_PROMPT,
        },
        'temperature': 0.3,
        'max tokens': 4096,
        'repeat penalty': 1.05,
        'frequency penalty': 0.2,
        'timeout': 300,
        'retry attempts': 2,
        'merge system into user': True,
        'delay': 0.0,
    }

    def _setup_translator(self):
        self.lang_map['简体中文'] = 'Simplified Chinese'
        self.lang_map['繁體中文'] = 'Traditional Chinese'
        self.lang_map['日本語'] = 'Japanese'
        self.lang_map['English'] = 'English'
        self.lang_map['한국어'] = 'Korean'
        self.lang_map['Tiếng Việt'] = 'Vietnamese'
        self.lang_map['čeština'] = 'Czech'
        self.lang_map['Nederlands'] = 'Dutch'
        self.lang_map['Français'] = 'French'
        self.lang_map['Deutsch'] = 'German'
        self.lang_map['magyar nyelv'] = 'Hungarian'
        self.lang_map['Italiano'] = 'Italian'
        self.lang_map['Polski'] = 'Polish'
        self.lang_map['Português'] = 'Portuguese'
        self.lang_map['Brazilian Portuguese'] = 'Brazilian Portuguese'
        self.lang_map['limba română'] = 'Romanian'
        self.lang_map['русский язык'] = 'Russian'
        self.lang_map['Español'] = 'Spanish'
        self.lang_map['Türk dili'] = 'Turkish'
        self.lang_map['украї́нська мо́ва'] = 'Ukrainian'
        self.lang_map['Thai'] = 'Thai'

    def _build_system_prompt(self, n: int) -> str:
        src = self.lang_map.get(self.lang_source) or self.lang_source
        tgt = self.lang_map.get(self.lang_target) or self.lang_target
        user_prompt = (self.params.get('system prompt') or {}).get(
            'content', '') or _DEFAULT_SYSTEM_PROMPT
        user_prompt = user_prompt.rstrip()

        suffix = (
            f"\n\nSource language: {src}. Target language: {tgt}. "
            f"Input is a JSON array of {n} strings — text from the SAME manga "
            f"page/scene, in reading order. They may share characters, places, "
            f"and terms, so translate them as one coherent set: keep character "
            f"names, place names, special terms, honorifics, and tone CONSISTENT "
            f"across every entry, and use earlier entries as context for later ones. "
            f"Output MUST be a JSON array of exactly {n} translated strings "
            "in the same order. "
            "Preserve empty strings as empty strings. "
            "Do not merge, split, add, or remove entries. "
            "Output ONLY the raw JSON array. "
            "The output MUST start with `[` and end with `]`. "
            "Do NOT wrap the output in markdown code fences. "
            "Do NOT add any prose, explanation, or commentary before or after the array.\n\n"
            'Example input:  ["Hello", "", "Goodbye"]\n'
            'Example output: ["안녕하세요", "", "안녕히 가세요"]'
        )
        return user_prompt + suffix

    def _api_call(self, system_prompt: str, user_text: str,
                  temperature: float, max_tokens: int, timeout: int) -> str:
        base_url = (self.params.get('base url') or '').strip().rstrip('/')
        if not base_url:
            base_url = 'http://localhost:5001/v1'
        api_key = (self.params.get('api key') or '').strip() or 'sk-local'
        model = (self.params.get('model') or 'local').strip() or 'local'

        try:
            repeat_penalty = float(self.params.get('repeat penalty', 1.05))
        except (TypeError, ValueError):
            repeat_penalty = 1.05
        try:
            frequency_penalty = float(self.params.get('frequency penalty', 0.2))
        except (TypeError, ValueError):
            frequency_penalty = 0.2

        merge_system = bool(self.params.get('merge system into user', True))
        if merge_system:
            # Gemma 2/3 chat templates have no system role; the OpenAI-compat
            # layer in some local servers silently drops the system message
            # when rendering through such a template. Merging into one user
            # turn is the safe cross-model default.
            messages = [{"role": "user", "content": f"{system_prompt}\n\n{user_text}"}]
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        body = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty,
            # Non-OpenAI field; koboldcpp / LM Studio recognize it, others ignore.
            "repeat_penalty": repeat_penalty,
        }

        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=body,
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Local LLM HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        return (data["choices"][0]["message"]["content"] or "").strip()

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        try:
            temperature = float(self.params.get('temperature', 0.3))
        except (TypeError, ValueError):
            temperature = 0.3
        try:
            max_tokens = int(self.params.get('max tokens', 4096))
        except (TypeError, ValueError):
            max_tokens = 4096
        try:
            timeout = int(self.params.get('timeout', 300))
        except (TypeError, ValueError):
            timeout = 300
        try:
            retry_attempts = max(1, int(self.params.get('retry attempts', 2)))
        except (TypeError, ValueError):
            retry_attempts = 2

        system_prompt = self._build_system_prompt(len(src_list))
        user_text = json.dumps(src_list, ensure_ascii=False)

        last_err = None
        for attempt in range(1, retry_attempts + 1):
            # Bump temperature on retry to escape deterministic bad outputs
            # (notably Gemma-class echoing the source at low temps).
            attempt_temp = min(1.0, temperature + 0.2 * (attempt - 1))
            try:
                raw = self._api_call(
                    system_prompt, user_text, attempt_temp, max_tokens, timeout,
                )
            except requests.exceptions.ConnectionError as e:
                self.logger.error(
                    f"Local LLM connection refused (attempt {attempt}/{retry_attempts}): "
                    f"is the server at {self.params.get('base url')!r} running? {e}"
                )
                last_err = repr(e)
                break  # connection refused — retrying won't help
            except requests.exceptions.Timeout:
                last_err = f"timeout after {timeout}s"
                self.logger.warning(
                    f"Local LLM timeout (attempt {attempt}/{retry_attempts})"
                )
                continue
            except Exception as e:
                last_err = repr(e)
                self.logger.warning(
                    f"Local LLM call failed (attempt {attempt}/{retry_attempts}): {e}"
                )
                continue

            parsed = _parse_json_array(raw)
            if parsed is not None and len(parsed) == len(src_list):
                return parsed
            last_err = (
                f"unparseable or length-mismatched response "
                f"(got {len(parsed) if parsed is not None else 'None'} "
                f"for {len(src_list)})"
            )
            self.logger.warning(
                f"Local LLM {last_err}. Raw[:300]={raw[:300]!r}. "
                f"Attempt {attempt}/{retry_attempts}"
            )
            if attempt < retry_attempts:
                time.sleep(0.5)

        self.logger.error(f"Local LLM exhausted retries: {last_err}")
        return ['' for _ in src_list]
