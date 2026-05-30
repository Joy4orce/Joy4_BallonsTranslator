import urllib.request
from ordered_set import OrderedSet
from typing import Dict, List, Union, Set, Callable
import time, requests, re, uuid, base64, hmac, functools, json
from collections import OrderedDict

from .exceptions import InvalidSourceOrTargetLanguage, TranslatorSetupFailure, MissingTranslatorParams, TranslatorNotValid
from utils.textblock import TextBlock
from ..base import BaseModule, DEVICE_SELECTOR
from utils.registry import Registry
from utils.io_utils import text_is_empty
from utils.logger import logger as LOGGER

TRANSLATORS = Registry('translators')
register_translator = TRANSLATORS.register_module

PROXY = urllib.request.getproxies()

LANGMAP_GLOBAL = {
    'Auto': '',
    '简体中文': '',
    '繁體中文': '',
    '日本語': '',
    'English': '',
    '한국어': '',
    'Tiếng Việt': '',
    'čeština': '',
    'Nederlands': '',
    'Français': '',
    'Deutsch': '',
    'magyar nyelv': '',
    'Italiano': '',
    'Polski': '',
    'Português': '',
    'Brazilian Portuguese': '',
    'limba română': '',
    'русский язык': '',
    'Español': '',
    'Türk dili': '',
    'украї́нська мо́ва': '',  
    'Thai': '',
}

SYSTEM_LANG = ''
SYSTEM_LANGMAP = {
    'zh-CN': '简体中文'
}


# Source languages whose comics are read right-to-left (manga / manhua).
# When the source is one of these, a row of balloons is read right→left.
RTL_SOURCE_LANGS = {'日本語', '简体中文', '繁體中文'}


def reading_order_indices(blocks, rtl: bool) -> List[int]:
    """Return indices into `blocks` ordered for human reading.

    Blocks are banded into rows top-to-bottom by their vertical center
    (band height = 0.7 × median block height, so it adapts to the page's
    text size). Within a row, blocks are ordered right-to-left when `rtl`
    (Japanese / Chinese comics) else left-to-right.

    This is used to order the batched translation request so context-aware
    engines (LLMs) read a panel's text in narrative sequence and keep names,
    terms, and honorifics consistent. It does NOT reorder any caller-owned
    list — callers map results back by index.

    Any error (missing/odd geometry) falls back to the original order so a
    layout quirk can never break translation.
    """
    n = len(blocks)
    if n <= 1:
        return list(range(n))
    try:
        def xyxy(b):
            x = b.xyxy
            return float(x[0]), float(x[1]), float(x[2]), float(x[3])

        heights = sorted(max(1.0, xyxy(b)[3] - xyxy(b)[1]) for b in blocks)
        band = max(10.0, heights[n // 2] * 0.7)

        def key(i):
            x1, y1, x2, y2 = xyxy(blocks[i])
            cy = (y1 + y2) / 2
            cx = (x1 + x2) / 2
            return (int(cy // band), -cx if rtl else cx)

        return sorted(range(n), key=key)
    except Exception:
        return list(range(n))


def check_language_support(check_type: str = 'source'):
    
    def decorator(set_lang_method):
        @functools.wraps(set_lang_method)
        def wrapper(self, lang: str = ''):
            if check_type == 'source':
                supported_lang_list = self.supported_src_list
            else:
                supported_lang_list = self.supported_tgt_list
            if not lang in supported_lang_list:
                msg = '\n'.join(supported_lang_list)
                raise InvalidSourceOrTargetLanguage(f'Invalid {check_type}: {lang}\n', message=msg)
            return set_lang_method(self, lang)
        return wrapper

    return decorator


class BaseTranslator(BaseModule):

    concate_text = True
    cht_require_convert = False

    _postprocess_hooks = OrderedDict()
    _preprocess_hooks = OrderedDict()
    
    def __init__(self,
                 lang_source: str, 
                 lang_target: str,
                 raise_unsupported_lang: bool = True,
                 **params) -> None:
        super().__init__(**params)
        self.name = ''
        for key in TRANSLATORS.module_dict:
            if TRANSLATORS.module_dict[key] == self.__class__:
                self.name = key
                break
        self.textblk_break = '\n###\n'
        self.lang_source: str = lang_source
        self.lang_target: str = lang_target
        self.lang_map: Dict = LANGMAP_GLOBAL.copy()
        
        try:
            self.setup_translator()
        except Exception as e:
            if isinstance(e, MissingTranslatorParams):
                raise e
            else:
                raise TranslatorSetupFailure(e)
            
        # enable traditional chinese by converting from simplified chinese
        if self.cht_require_convert and not self.lang_map['繁體中文']:
            self.lang_map['繁體中文'] = self.lang_map['简体中文']

        self.valid_lang_list = [lang for lang in self.lang_map if self.lang_map[lang] != '']

        try:
            self.set_source(lang_source)
            self.set_target(lang_target)
        except InvalidSourceOrTargetLanguage as e:
            if raise_unsupported_lang:
                raise e
            else:
                lang_source = self.supported_src_list[0]
                lang_target = self.supported_tgt_list[0]
                self.set_source(lang_source)
                self.set_target(lang_target)

    def _setup_translator(self):
        raise NotImplementedError

    def setup_translator(self):
        self._setup_translator()

    @check_language_support(check_type='source')
    def set_source(self, lang: str):
        self.lang_source = lang

    @check_language_support(check_type='target')
    def set_target(self, lang: str):
        self.lang_target = lang

    def _translate(self, src_list: List[str]) -> List[str]:
        raise NotImplementedError

    def translate(self, text: Union[str, List]) -> Union[str, List]:
        if text_is_empty(text):
            return text

        is_list = isinstance(text, List)
        concate_text = is_list and self.concate_text
        text_source = self.textlist2text(text) if concate_text else text
        
        src_is_list = isinstance(text_source, List)
        if src_is_list: 
            text_trans = self._translate(text_source)
        else:
            text_trans = self._translate([text_source])[0]
        
        if text_trans is None:
            if is_list:
                text_trans = [''] * len(text)
            else:
                text_trans = ''
        elif concate_text:
            text_trans = self.text2textlist(text_trans)
            
        if is_list:
            try:
                assert len(text_trans) == len(text)
            except:
                LOGGER.error('This translator seems to messed up the translation which resulted in inconsistent translated line count.\n \
                             Set concate_text to False or change textblk_break in the source code may solve the problem.')
                raise
            # for ii, t in enumerate(text_trans):
            #     for callback in self._postprocess_hooks:
            #         text_trans[ii] = callback(t)
        # else:
        #     for callback in self._postprocess_hooks:
        #         text_trans = callback(text_trans)

        return text_trans

    def textlist2text(self, text_list: List[str]) -> str:
        # some translators automatically strip '\n'
        # so we insert '\n###\n' between concated text instead of '\n' to avoid mismatch
        return self.textblk_break.join(text_list)

    def text2textlist(self, text: str) -> List[str]:
        breaker = self.textblk_break.replace('\n', '') or '\n'
        text_list = text.split(breaker)
        return [text.lstrip().rstrip() for text in text_list]

    def translate_textblk_lst(self, textblk_lst: List[TextBlock]):
        '''
        only textblks with non-empty source text would be passed to translator

        Non-empty blocks are batched into a single translate() call, ordered in
        human reading order (top-to-bottom rows; right-to-left within a row for
        Japanese/Chinese sources, left-to-right otherwise). This lets
        context-aware engines (LLMs) see a panel's text in narrative sequence
        and keep character names, terms, and honorifics consistent across the
        batch. Results are mapped back to each block by its original index, so
        the caller's `textblk_lst` order is never changed — only the order
        inside the translation request.
        '''
        translations = [blk.get_text() for blk in textblk_lst]

        # (original_index, source_text) for every non-empty block.
        pending = [(ii, txt) for ii, txt in enumerate(translations)
                   if txt.strip() != '']

        if len(pending) > 0:
            # Order the batch by reading order WITHOUT disturbing textblk_lst.
            order = reading_order_indices(
                [textblk_lst[ii] for ii, _ in pending],
                rtl=self.lang_source in RTL_SOURCE_LANGS,
            )
            ordered = [pending[k] for k in order]
            text_list = [txt for _, txt in ordered]

            _translations = self.translate(text_list)

            # _translations is in the same order as text_list (== ordered),
            # so zip aligns each result with its original block index.
            for (orig_idx, _), tr in zip(ordered, _translations):
                translations[orig_idx] = tr

        for callback_name, callback in self._postprocess_hooks.items():
            callback(translations = translations, textblocks = textblk_lst, translator = self)

        for tr, blk in zip(translations, textblk_lst):
            blk.translation = tr

    def supported_languages(self) -> List[str]:
        return self.valid_lang_list

    @property
    def supported_tgt_list(self) -> List[str]:
        return self.valid_lang_list

    @property
    def supported_src_list(self) -> List[str]:
        return self.valid_lang_list
        
    def delay(self) -> float:
        if 'delay' in self.params:
            delay = self.params['delay']
            if delay:
                try:
                    return float(delay)
                except:
                    pass
        return 0.
