"""Claude (Claude Code CLI) translator.

Calls the local `claude` executable as a subprocess and authenticates via the
user's Claude subscription (OAuth) instead of an API key. Mirrors the pattern
used in the user's other projects (Joy4_Novel / Joy4_sub).

Sends the page's text blocks as a JSON array and expects a JSON array of the
same length back. concate_text = False so BaseTranslator hands us the raw list.
"""

import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List

from .base import BaseTranslator, register_translator
from .exceptions import MissingTranslatorParams


if sys.platform == "win32":
    _NO_WINDOW_KW = {"creationflags": subprocess.CREATE_NO_WINDOW}
else:
    _NO_WINDOW_KW = {}


# ── claude.exe discovery ────────────────────────────────────────────────────

def _can_execute(path: str) -> bool:
    if not path:
        return False
    try:
        r = subprocess.run(
            [path, "--version"],
            capture_output=True,
            timeout=15,
            **_NO_WINDOW_KW,
        )
        return r.returncode == 0
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False


def _find_claude_cli(override: str = "") -> str:
    candidates: List[str] = []
    if override:
        candidates.append(override)

    found = shutil.which("claude")
    if found:
        candidates.append(found)

    appdata = os.environ.get("APPDATA", "")
    if appdata:
        try:
            bundled = sorted(
                glob.glob(os.path.join(appdata, "Claude", "claude-code", "*", "claude.exe")),
                reverse=True,
            )
            candidates.extend(bundled)
        except Exception:
            pass
        base = os.path.join(appdata, "Claude", "claude-code")
        try:
            for sub in sorted(os.listdir(base), reverse=True):
                candidates.append(os.path.join(base, sub, "claude.exe"))
        except Exception:
            pass
        candidates.append(os.path.join(appdata, "npm", "claude.cmd"))

    home = os.path.expanduser("~")
    candidates.extend([
        os.path.join(home, ".local", "bin", "claude.exe"),
        os.path.join(home, ".local", "bin", "claude"),
        "/usr/local/bin/claude",
        os.path.expanduser("~/.claude/local/claude"),
    ])

    seen = set()
    for c in candidates:
        if not c or c in seen:
            continue
        seen.add(c)
        if _can_execute(c):
            return c
    return ""


# ── response cleanup ────────────────────────────────────────────────────────

_THINK_TAGS = r"(?:thinking|think|thought|reasoning)"
_RE_THINK_BLOCK = re.compile(
    rf"<\s*{_THINK_TAGS}\s*>.*?<\s*/\s*{_THINK_TAGS}\s*>\s*",
    re.IGNORECASE | re.DOTALL,
)


def _strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r'^\s*```(?:json|JSON)?\s*\n?', '', cleaned)
    cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned)
    return cleaned.strip()


def _parse_json_array(text: str) -> List[str]:
    cleaned = _RE_THINK_BLOCK.sub("", text)
    cleaned = _strip_markdown_fences(cleaned)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(x) for x in data]
    except json.JSONDecodeError:
        pass
    match = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list):
                return [str(x) for x in data]
        except json.JSONDecodeError:
            pass
    return None


# ── translator ──────────────────────────────────────────────────────────────

@register_translator('Claude')
class ClaudeCLITranslator(BaseTranslator):

    concate_text = False
    cht_require_convert = True
    params: Dict = {
        'claude exe path': '',
        'model': {
            'type': 'selector',
            'options': [
                'haiku',
                'sonnet',
                'opus',
            ],
            'select': 'haiku',
        },
        'oauth token': '',
        'extra system prompt': {
            'type': 'editor',
            'content': '',
        },
        'timeout': 300,
        'retry attempts': 2,
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

        self._path_cache = ("", "")

    def _resolve_path(self) -> str:
        override = (self.params.get('claude exe path') or '').strip()
        cached_override, cached_path = self._path_cache
        if cached_path and cached_override == override and os.path.isfile(cached_path):
            return cached_path
        path = _find_claude_cli(override)
        self._path_cache = (override, path)
        return path

    def _build_system_prompt(self, n: int) -> str:
        src = self.lang_map.get(self.lang_source) or self.lang_source
        tgt = self.lang_map.get(self.lang_target) or self.lang_target
        extra = (self.params.get('extra system prompt') or {}).get('content', '').strip()
        base = (
            f"You are a professional manga / comic translator. "
            f"Translate each entry of the input JSON array from {src} to {tgt}.\n\n"
            f"CONTEXT: The entries are text from the SAME manga page/scene, given "
            f"in reading order. They may refer to the same characters, places, and "
            f"terms. Translate them as ONE coherent set, not in isolation:\n"
            f"- Keep character names, place names, and special terms translated "
            f"IDENTICALLY across every entry.\n"
            f"- Keep honorifics, speech level, and tone consistent for the same "
            f"speaker throughout.\n"
            f"- Use earlier entries as context to disambiguate later ones.\n\n"
            f"OUTPUT RULES (strict):\n"
            f"1. Output ONLY a JSON array of exactly {n} translated strings, "
            f"in the same order as the input. No prose, no markdown fences, no commentary.\n"
            f"2. The very first character of your response MUST be '[' and the last MUST be ']'.\n"
            f"3. Preserve empty strings as empty strings. Do not merge, split, add, or drop entries.\n"
            f"4. Translate the meaning naturally — do not output the source language as-is "
            f"unless the entry is already in {tgt} or is a proper noun with no localization.\n"
            f"5. Do not narrate your reasoning. Do not use <thinking> tags."
        )
        if extra:
            base = base + "\n\n" + extra
        return base

    def _run_claude(self, system_prompt: str, user_input: str, timeout: int) -> str:
        path = self._resolve_path()
        if not path:
            self.logger.error(
                "claude CLI not found. Install Claude Code "
                "(https://claude.ai/install) or set 'claude exe path' in the panel."
            )
            raise MissingTranslatorParams('claude exe path')

        model = (self.params.get('model') or {}).get('select', 'haiku')
        cmd = [
            path,
            "--print",
            "--output-format", "text",
            "--model", model,
            "--tools", "",
            "--system-prompt", system_prompt,
        ]

        env = os.environ.copy()
        # Strip context that breaks claude.exe's own auth when invoked from
        # within a Claude Code session.
        env.pop("CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST", None)
        if not env.get("ANTHROPIC_API_KEY"):
            env.pop("ANTHROPIC_API_KEY", None)

        token = (self.params.get('oauth token') or '').strip()
        if token:
            env["CLAUDE_CODE_OAUTH_TOKEN"] = token

        result = subprocess.run(
            cmd,
            input=user_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            **_NO_WINDOW_KW,
        )
        if result.returncode != 0:
            err = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(f"Claude CLI error (exit {result.returncode}): {err[:400]}")
        return result.stdout or ""

    def _translate(self, src_list: List[str]) -> List[str]:
        if not src_list:
            return []

        try:
            timeout = int(self.params.get('timeout', 300))
        except (TypeError, ValueError):
            timeout = 300
        try:
            retry_attempts = max(1, int(self.params.get('retry attempts', 2)))
        except (TypeError, ValueError):
            retry_attempts = 2

        system_prompt = self._build_system_prompt(len(src_list))
        user_input = json.dumps(src_list, ensure_ascii=False)

        last_err = None
        for attempt in range(1, retry_attempts + 1):
            try:
                raw = self._run_claude(system_prompt, user_input, timeout)
            except MissingTranslatorParams:
                raise
            except subprocess.TimeoutExpired:
                last_err = f"timeout after {timeout}s"
                self.logger.warning(
                    f"Claude CLI timeout (attempt {attempt}/{retry_attempts})"
                )
                continue
            except Exception as e:
                last_err = repr(e)
                self.logger.warning(
                    f"Claude CLI failed (attempt {attempt}/{retry_attempts}): {e}"
                )
                continue

            parsed = _parse_json_array(raw)
            if parsed is not None and len(parsed) == len(src_list):
                return parsed
            last_err = f"unparseable or length-mismatched response (got {len(parsed) if parsed else 'None'} for {len(src_list)})"
            self.logger.warning(
                f"Claude CLI {last_err}. Raw[:300]={raw[:300]!r}. "
                f"Attempt {attempt}/{retry_attempts}"
            )
            if attempt < retry_attempts:
                time.sleep(1.0)

        self.logger.error(f"Claude CLI exhausted retries: {last_err}")
        return ['' for _ in src_list]
