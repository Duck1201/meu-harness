"""Builds the system prompt from the contracts, instead of hand-written prose.

Two things the model kept getting wrong are things the harness already knows and
never said. Asked to read an image, it spent six calls searching the workspace
and the web for a file that does not exist, because nothing told it the session
has no vision — while ``config/model-profiles.json`` has declared exactly that
all along. And after picking the right tool it called another one to check the
answer, because nothing told it a ResultPayload is the answer.

So the prompt is generated: the capability lines come from the active
RuntimeProfile, and drift between what the contract declares and what the model
is told becomes impossible rather than merely unlikely.
"""

import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from .config import HarnessConfig

# Nas conversas reais do Operator o Gemma respondia "vou atualizar o arquivo" e
# encerrava o turno sem tool call, e o Operator tinha de pedir "chama a tool" de
# novo. O prompt antigo só dizia quando não chamar; esta linha diz quando chamar.
_BASE = (
    "You act only through tools. When the Operator asks you to create, change, fetch, "
    "look up, calculate or read something, call the tool in this step: never reply that "
    "you will do it, and never say a file changed unless a tool changed it in this turn. "
    "Answer without a tool only when the request needs no action and no data you lack. "
    "All workspace paths supplied to tools must be relative to the workspace root."
)

# A tool result is the answer, not a claim to be checked. Re-reading what a
# previous result already reported is the single most common way a Turn spends
# its budget without learning anything. A versão anterior terminava em "não leia
# os arquivos que a busca achou", e o Gemma, pedido para trocar um nome, achou as
# linhas com grep_search e respondeu que o Operator tinha proibido editá-las.
_RESULT_AUTHORITY = (
    "A tool result is authoritative. Do not call another tool to confirm what a "
    "result in this turn already reported: if a search listed the files, that is "
    "the list; if an edit reported success, the file changed. In particular, "
    "after glob or grep_search, do not read the files they named just to confirm "
    "the search; do read or edit them when the request needs their content or a "
    "change to them."
)

# Criar um arquivo é uma chamada só: o modelo lia "precisa do SHA" como "ache o
# arquivo antes" e gastava dez passos procurando um caminho que nunca existiu. O
# SHA saiu do write_file e do edit; a ordem ler-depois-editar é o que sobra dizer.
_MUTATION_DIRECTNESS = (
    "Creating or completely rewriting a file is a single write_file call: nothing has to "
    "be read or located first. To change part of an existing file, read_file it, then "
    "call edit with old_string copied exactly from what read_file returned. "
    "Once a tool reports a path is absent, treat it as absent — do not call the "
    "same tool again with different arguments to look for it."
)


# The one line that is not derived from a contract: the host knows the date and the
# model does not. It is a decision of the Operator, not a measurement — no run has
# been observed spending a step on it — so it is stated in the same shape as the
# lines above, fact plus the waste it forbids, and can be measured later.
def _current_date(today: date) -> str:
    return (
        f"Today's date is {today.isoformat()} (UTC). It comes from the host and is "
        "authoritative: do not derive the date from your training data and do not "
        "call a tool to look it up."
    )


# Only capabilities whose absence changes what the model should do. Streaming and
# parallel tool calls are harness concerns and would be noise in the prompt.
_MODEL_FACING_LIMITS: Mapping[str, str] = {
    "vision": (
        "You cannot see images. If a request depends on looking at one, say so "
        "plainly instead of searching the workspace or the web for it."
    ),
}


# O que substitui a linha de visão quando describe_image está no catálogo: o
# modelo de chat continua cego, mas agora tem a quem perguntar (ADR 0016).
_VISION_TOOL_LINE = (
    "You cannot see images yourself. When a request depends on an image file in the "
    "workspace, ask describe_image a specific question about it, and pass on its caveat "
    "when the answer depends on small text or digits. If there is no such file, say so "
    "plainly instead of searching the web for it."
)


# ponytail: teto em caracteres, não em tokens — o estimator exige o tokenizer
# carregado, que o composition root ainda não tem quando isto roda. São ~1k tokens
# num orçamento de 65536 (config/harness.json#context.initial_budget_tokens).
# Trocar por contagem real de tokens se o teto errar na prática.
_OPERATOR_NOTES_LIMIT = 4000

# Comentário HTML é nota para quem abre o arquivo, não texto para o modelo.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def load_operator_notes(path: Path = Path("SYSTEM-PROMPT.md")) -> str:
    """O texto que o Operator anexou ao prompt, ou ``""`` quando não há arquivo.

    O arquivo inteiro é o texto do Operator, menos os comentários HTML. Ler o
    arquivo é trabalho do composition root, não de ``build_system_prompt``: o
    texto chega lá por argumento pelo mesmo motivo que a data chega, para que
    corpus e produção não passem a montar prompts diferentes em silêncio.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    stripped = _HTML_COMMENT.sub("", text).strip()
    if len(stripped) > _OPERATOR_NOTES_LIMIT:
        raise ValueError(
            f"O texto do Operator em {path} tem {len(stripped)} caracteres e o "
            f"limite é {_OPERATOR_NOTES_LIMIT}: ele entra em todo Turn e comeria "
            "o orçamento de contexto antes do histórico."
        )
    return stripped


def build_system_prompt(
    config: HarnessConfig, *, today: date, operator_notes: str, vision_tool_offered: bool
) -> str:
    """The prompt for the active RuntimeProfile, capability lines included.

    ``today`` has no default on purpose: a prompt that reads the clock by itself
    would make every bench run and every recorded trace differ by the day it ran.
    Production passes the host clock, the corpus passes a fixed date. For the same
    reason ``operator_notes`` is required and never read from disk here: an empty
    string is a stated absence, and a bench run that forgot the Operator's text
    would measure a different system than the one the Operator runs.
    ``vision_tool_offered`` is required for the same reason: whether describe_image
    is in the catalogue depends on the host, and a line telling the model it cannot
    see while the tool that sees is on offer would contradict the catalogue.
    """
    capabilities = config.runtime_profile.capabilities
    limits = [
        _VISION_TOOL_LINE if name == "vision" and vision_tool_offered else sentence
        for name, sentence in _MODEL_FACING_LIMITS.items()
        if (capability := capabilities.get(name)) is None or capability.support != "supported"
    ]
    # Lista vazia e não `operator_notes` direto: uma string vazia no join deixaria
    # um espaço sobrando no fim e o prompt sem bloco deixaria de ser o de sempre.
    extra = [operator_notes] if operator_notes else []
    return " ".join(
        [_BASE, _RESULT_AUTHORITY, _MUTATION_DIRECTNESS, _current_date(today), *limits, *extra]
    )


__all__ = [
    "build_system_prompt",
    "load_operator_notes",
]
