"""LongMemEval's own reader and judge prompts, copied verbatim so a score here
is comparable with published LongMemEval numbers.

Source: https://github.com/xiaowu0162/LongMemEval at UPSTREAM_COMMIT,
``src/generation/run_generation.py`` (``prepare_prompt``, natural-language
history, session granularity) and ``src/evaluation/evaluate_qa.py``
(``get_anscheck_prompt``). Only string formatting is reproduced; nothing is
reworded.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

UPSTREAM_COMMIT = '9e0b455f4ef0e2ab8f2e582289761153549043fc'

_READER = ('I will give you several history chats between you and a user. Please answer the question based on the '
           'relevant chat history.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:')
_READER_COT = ('I will give you several history chats between you and a user. Please answer the question based on the '
               'relevant chat history. Answer the question step by step: first extract all the relevant information, '
               'and then reason over the information to get the answer.\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\n'
               'Question: {}\nAnswer (step by step):')

_CONTAINS = ('I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
             'response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct '
             'answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If '
             'the response only contains a subset of the information required by the answer, answer no. ')
_JUDGE_PLAIN = _CONTAINS + ('\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response '
                            'correct? Answer yes or no only.')
_JUDGE_TEMPORAL = _CONTAINS + ('In addition, do not penalize off-by-one errors for the number of days. If the question '
                               'asks for the number of days/weeks/months, etc., and the model makes off-by-one errors '
                               '(e.g., predicting 19 days when the answer is 18), the model\'s response is still correct. '
                               '\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response '
                               'correct? Answer yes or no only.')
_JUDGE_UPDATE = ('I will give you a question, a correct answer, and a response from a model. Please answer yes if the '
                 'response contains the correct answer. Otherwise, answer no. If the response contains some previous '
                 'information along with an updated answer, the response should be considered as correct as long as '
                 'the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: '
                 '{}\n\nIs the model response correct? Answer yes or no only.')
_JUDGE_PREFERENCE = ('I will give you a question, a rubric for desired personalized response, and a response from a '
                     'model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. '
                     'The model does not need to reflect all the points in the rubric. The response is correct as long '
                     'as it recalls and utilizes the user\'s personal information correctly.\n\nQuestion: {}\n\n'
                     'Rubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.')
_JUDGE_ABSTENTION = ('I will give you an unanswerable question, an explanation, and a response from a model. Please '
                     'answer yes if the model correctly identifies the question as unanswerable. The model could say '
                     'that the information is incomplete, or some other information is given but the asked information '
                     'is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly '
                     'identify the question as unanswerable? Answer yes or no only.')

_JUDGE_BY_TYPE = {
    'single-session-user': _JUDGE_PLAIN,
    'single-session-assistant': _JUDGE_PLAIN,
    'multi-session': _JUDGE_PLAIN,
    'temporal-reasoning': _JUDGE_TEMPORAL,
    'knowledge-update': _JUDGE_UPDATE,
    'single-session-preference': _JUDGE_PREFERENCE,
}


@dataclass(frozen=True)
class Turn:
    role: str
    content: str


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str  # the dataset's own string, e.g. '2023/05/20 (Sat) 02:21'
    turns: tuple[Turn, ...]


def history(sessions: Sequence[Session]) -> str:
    """Upstream's session-level natural-language history: sorted by date, numbered from 1."""
    parts = []
    for number, session in enumerate(sorted(sessions, key=lambda s: s.date), 1):
        content = ''.join(f'\n\n{turn.role}: {turn.content.strip()}' for turn in session.turns)
        parts.append(f'\n### Session {number}:\nSession Date: {session.date}\nSession Content:\n{content}\n')
    return ''.join(parts)


def reader_prompt(sessions: Sequence[Session], question_date: str, question: str, *, cot: bool) -> str:
    if not sessions:
        raise ValueError('the reader needs at least one session')
    return (_READER_COT if cot else _READER).format(history(sessions), question_date, question)


def is_abstention(question_id: str) -> bool:
    return '_abs' in question_id


def judge_prompt(question_type: str, question_id: str, question: str, answer: str, response: str) -> str:
    if is_abstention(question_id):
        return _JUDGE_ABSTENTION.format(question, answer, response)
    try:
        template = _JUDGE_BY_TYPE[question_type]
    except KeyError:
        raise ValueError(f'no LongMemEval judge prompt for question type {question_type!r}') from None
    return template.format(question, answer, response)


def judged_correct(verdict: str) -> bool:
    """Upstream's rule, kept as it is: any 'yes' in the lowercased reply."""
    return 'yes' in verdict.lower()
