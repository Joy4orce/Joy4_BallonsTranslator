"""Screen translator settings.

All module choices (detector / OCR / translator / langs) default to whatever
BallonsTranslator's main UI has saved in config/config.json. Set any of the
override constants below to a non-None value to force a different choice
just for the screen translator.

Per-module parameter dicts (TRANSLATOR_PARAM_OVERRIDES etc.) are deep-merged
on top of BT's saved params, keyed by module name. Leave them empty {} to
inherit everything from BT exactly.
"""

import json
import os
from types import SimpleNamespace


_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_BT_CONFIG_PATH = os.path.join(_ROOT, 'config', 'config.json')


# ── Module choices — None means "use whatever BT has saved" ─────────────────
TEXT_DETECTOR = None
OCR_MODULE = None
TRANSLATOR_MODULE = None
INPAINTER_MODULE = None
SOURCE_LANG = None
TARGET_LANG = None

# ── Local-only (not in BT config) ───────────────────────────────────────────
DEVICE = 'cuda'   # 'cuda' or 'cpu'

# When True, the captured image is inpainted (original text erased) before
# the translated overlay paints on top — translations look like they replace
# the original text in place. When False, the old behavior is used: white
# translation boxes drawn on top of the unmodified region.
# Disable this if the inpainter is too slow on your machine or fails on
# unusual balloon styles (black background, patterned fills, etc.).
ENABLE_INPAINT = True

# When True, each text block's enclosing speech balloon is detected and the
# translation is laid out to fill (and center within) the whole balloon,
# instead of starting at the original-text bounding box. This fixes the
# "vertical Japanese text was narrow on the right side, so the horizontal
# Korean translation runs out of room" problem. Falls back to the text bbox
# automatically when no clean balloon is found (SFX, open-air text, etc.).
FILL_BALLOON = True

# Param overrides — merged on top of BT's <module>_params. Shape matches
# BallonsTranslator: Dict[module_name, params_dict]. Examples:
#
#   TRANSLATOR_PARAM_OVERRIDES = {
#       'Local LLM': {'temperature': 0.2, 'model': 'gemma-3-12b'},
#       'Claude': {'model': {'select': 'haiku'}},
#   }
#   OCR_PARAM_OVERRIDES = {
#       'manga_ocr_2025': {'device': {'select': 'cuda'}},
#   }
TRANSLATOR_PARAM_OVERRIDES: dict = {}
OCR_PARAM_OVERRIDES: dict = {}
DETECTOR_PARAM_OVERRIDES: dict = {}
INPAINTER_PARAM_OVERRIDES: dict = {}

# ── Trigger / UI ────────────────────────────────────────────────────────────
GLOBAL_HOTKEY = 'ctrl+alt+t'    # None to disable
OVERLAY_AUTO_DISMISS_MS = 0     # 0 = stay until clicked / ESC
OVERLAY_BOX_OPACITY = 0.85
OVERLAY_FONT_SIZE = 14
OVERLAY_FONT_FAMILY = 'Malgun Gothic'


# ── Hard-coded fallbacks (used only if both this file and BT config are silent) ─
_FALLBACK_DETECTOR = 'ctd'
_FALLBACK_OCR = 'manga_ocr_2025'
_FALLBACK_TRANSLATOR = 'google'
_FALLBACK_INPAINTER = 'patchmatch'   # fast classical CV; safe if no DL model is downloaded
_FALLBACK_SOURCE = '日本語'
_FALLBACK_TARGET = '한국어'


def _load_bt_module_config() -> dict:
    """Return the `module` section of BT's config/config.json, or {} on failure.

    Read fresh on every call — that's what makes the screen translator pick
    up settings changed in the main app without a restart.
    """
    try:
        with open(_BT_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f).get('module', {}) or {}
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f'[screen_translator] could not read {_BT_CONFIG_PATH}: {e}')
        return {}


def _merge_per_module_params(bt_params: dict, overrides: dict) -> dict:
    """Deep-merge per-module param dicts.

    Both inputs are Dict[module_name, params_dict]. For each module, we shallow-
    merge the override values onto BT's saved values. This is one level deeper
    than a plain dict.update — without it, setting a single override key for a
    module would wipe out the rest of that module's BT-saved params.
    """
    out = {k: dict(v) if isinstance(v, dict) else v
           for k, v in (bt_params or {}).items()}
    for module_name, override in (overrides or {}).items():
        base = out.get(module_name) or {}
        if isinstance(base, dict) and isinstance(override, dict):
            out[module_name] = {**base, **override}
        else:
            out[module_name] = override
    return out


def resolve() -> SimpleNamespace:
    """Resolve the final, ready-to-use settings.

    Precedence (highest first):
      1. Local override constants in this file (non-None values).
      2. BallonsTranslator's saved settings (config/config.json).
      3. Hard-coded fallbacks.

    Returns a SimpleNamespace with the same shape every caller can rely on.
    """
    bt = _load_bt_module_config()

    return SimpleNamespace(
        text_detector=TEXT_DETECTOR or bt.get('textdetector') or _FALLBACK_DETECTOR,
        ocr_module=OCR_MODULE or bt.get('ocr') or _FALLBACK_OCR,
        translator_module=TRANSLATOR_MODULE or bt.get('translator') or _FALLBACK_TRANSLATOR,
        inpainter_module=INPAINTER_MODULE or bt.get('inpainter') or _FALLBACK_INPAINTER,
        source_lang=SOURCE_LANG or bt.get('translate_source') or _FALLBACK_SOURCE,
        target_lang=TARGET_LANG or bt.get('translate_target') or _FALLBACK_TARGET,

        device=DEVICE,
        enable_inpaint=ENABLE_INPAINT,
        fill_balloon=FILL_BALLOON,

        translator_params=_merge_per_module_params(
            bt.get('translator_params'), TRANSLATOR_PARAM_OVERRIDES),
        ocr_params=_merge_per_module_params(
            bt.get('ocr_params'), OCR_PARAM_OVERRIDES),
        detector_params=_merge_per_module_params(
            bt.get('textdetector_params'), DETECTOR_PARAM_OVERRIDES),
        inpainter_params=_merge_per_module_params(
            bt.get('inpainter_params'), INPAINTER_PARAM_OVERRIDES),

        global_hotkey=GLOBAL_HOTKEY,
        overlay_auto_dismiss_ms=OVERLAY_AUTO_DISMISS_MS,
        overlay_box_opacity=OVERLAY_BOX_OPACITY,
        overlay_font_size=OVERLAY_FONT_SIZE,
        overlay_font_family=OVERLAY_FONT_FAMILY,
    )
