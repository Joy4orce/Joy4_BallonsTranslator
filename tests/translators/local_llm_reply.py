"""What LocalLLMTranslator does with a reply that is not what was asked for:
the wrong language, the wrong number of entries, or nothing that parses.

Plain script, no server needed — the call to it is stubbed out:

    python tests/translators/local_llm_reply.py
"""
import json
import os.path as osp
import sys

APP_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(APP_ROOT)

from modules.translators import trans_local_llm as m

fails = 0


def expect(label, cond, detail=''):
    global fails
    fails += 0 if cond else 1
    print(f'{"PASS" if cond else "FAIL"}  {label:46} {detail}')


W = m._wrong_language_entries
src = ['こんにちは', 'OK', 'LINEで送る', 'GAME OVER', '半径100km', 'ユーチューブ']
out = ['Hello there friend', 'OK', '라인으로 보낼게', 'GAME OVER', '반경 100km', 'YouTube']
expect('english for japanese is caught', W(src, out, 'Korean') == [0, 5], str(W(src, out, 'Korean')))
expect('latin-script targets are not judged', W(['こんにちは'], ['Hello there'], 'English') == []
       and W(['こんにちは'], ['Hola amigo'], 'Spanish') == [])


class Fake(m.LocalLLMTranslator):
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.lang_map = {'한국어': 'Korean', '日本語': 'Japanese'}
        self.lang_source, self.lang_target = '日本語', '한국어'
        self.params = dict(m.LocalLLMTranslator.params)
        self.params['retry attempts'] = 2
        import logging
        self.logger = logging.getLogger('fake')

    def _api_call(self, system_prompt, user_text, *a, **k):
        self.prompts.append((system_prompt, json.loads(user_text)))
        reply = self.replies.pop(0)
        return reply(json.loads(user_text)) if callable(reply) else reply


m.time.sleep = lambda s: None

# English first, Korean when told off.
t = Fake(['["Good morning everyone", "Thank you very much"]', '["좋은 아침", "고마워"]'])
got = t._translate(['おはよう', 'ありがとう'])
expect('wrong language is retried', got == ['좋은 아침', '고마워'], str(got))
expect('the retry says what was wrong', 'not in Korean' in t.prompts[1][0]
       and 'not in Korean' not in t.prompts[0][0])

# English every time: shown anyway rather than nothing.
t = Fake(['["Good morning everyone", "Thank you very much"]'] * 2)
got = t._translate(['おはよう', 'ありがとう'])
expect('persistent english is still returned', got == ['Good morning everyone', 'Thank you very much'], str(got))

# One entry dropped whenever four are sent; halves come back whole.
t = Fake([lambda xs: json.dumps(['가'] * (len(xs) - 1 if len(xs) == 4 else len(xs)),
                                ensure_ascii=False)] * 4)
got = t._translate(['あ', 'い', 'う', 'え'])
expect('miscounted list is halved', got == ['가'] * 4 and [len(p[1]) for p in t.prompts] == [4, 4, 2, 2],
       f'{got} calls {[len(p[1]) for p in t.prompts]}')

# Unparseable prose is not something halving fixes into existence, but a
# single entry cannot be halved: comes back empty, as before.
t = Fake(['I cannot do that.'] * 2)
got = t._translate(['あ'])
expect('single entry failure is empty', got == [''], str(got))

print('all passed' if not fails else f'{fails} failed')
sys.exit(1 if fails else 0)
