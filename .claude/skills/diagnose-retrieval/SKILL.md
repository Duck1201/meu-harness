---
name: diagnose-retrieval
description: Diagnostica qualidade de RAG olhando o que o modelo recebeu, não a estatística do índice. Use quando a resposta com Corpus vier ruim, inventada, fora do assunto ou magra; quando alguém perguntar se a recuperação "está boa"; quando for calibrar piso de similaridade, tamanho de chunk ou overlap; e antes de mexer em qualquer valor de `config/harness.json#corpus`.
---

# Diagnosticar recuperação

Contagem de chunk não diz se o RAG está bom. Média de token não diz. O que diz é
**a passagem literal que o modelo recebeu naquele turno**, junto da instrução que
veio com ela.

Uma sessão inteira de correções neste repo nasceu de uma frase do Operator —
"não ficou bom" — e de abrir o turno para ver que metade das passagens era
bibliografia. Nenhuma métrica agregada mostrava isso.

## A regra

**Nunca opine sobre qualidade de recuperação sem ler a passagem inteira.**

Ler truncado em 150 caracteres já me fez quase acusar o modelo de citar errado
quando a citação estava correta — o trecho que sustentava a afirmação estava no
caractere 200. Leia inteiro, sempre.

## Onde olhar, nessa ordem

### 1. Telemetria — só para achar o turno
`state_dir/observability.sqlite3`, tabela `observability_events`:

```sql
SELECT occurred_at, payload, turn_id FROM observability_events
WHERE event_type='corpus.retrieval' ORDER BY id DESC LIMIT 20;
```

Devolve `{corpus_id, passages, rewritten, taints}`. Por contrato, telemetria
**nunca** guarda conteúdo — só serve para saber quantas passagens entraram e em
qual turno. `passages: 0` num turno que devia recuperar, ou `passages: 6` num
turno que não devia, já aponta o problema.

### 2. CanonicalHistory — o que o modelo recebeu de fato
`state_dir/conversations.sqlite3`, tabela `canonical_history`, filtrando por
`turn_id`. É a única fonte com a pergunta, a **instrução injetada**, as
**passagens literais**, as chamadas de tool e a resposta.

Duas armadilhas de leitura:

- As `tool_calls` **não têm entrada própria**: vivem dentro do payload de
  `model_attempt`. Procurar por `entry_type='tool_call'` devolve vazio e parece
  que o modelo não chamou nada.
- `internal_automation` carrega o campo `instruction`. **Leia essa instrução.**
  Ela é texto que o harness impõe ao modelo, e um imperativo mal escrito ali
  produz alucinação que parece culpa do modelo.

### 3. Reproduzir a busca fora do turno
`scripts/olhar.py` desta skill roda a mesma busca com os parâmetros do contrato e
imprime similaridade, escore de fusão e endereço de cada passagem:

```bash
uv run python .claude/skills/diagnose-retrieval/scripts/olhar.py "a pergunta" [corpus] [reescrita-em-ingles]
```

É o que separa "a busca errou" de "o modelo ignorou o que recebeu".

## Medir antes de propor

Todo número de contrato que esta sessão mexeu saiu de medição contra material
real, nunca de intuição. O padrão:

1. **Monte um conjunto com os dois lados.** Para calibrar piso: perguntas
   respondíveis, perguntas fora do acervo **e comandos** ("cria um arquivo .md",
   "roda os testes", "Olá"). Foi o terceiro grupo que expôs o furo — a calibragem
   original só tinha perguntas, e comando pontuava exatamente na faixa que passava.
2. **Imprima a faixa de cada grupo** e escolha o corte onde eles separam. Declare
   a margem: se ela for fina, diga que é fina.
3. **Amarre o número ao embedder.** Piso medido no `bge-m3` não transfere.
4. **Commite a medição junto do número**, no corpo da mensagem. O próximo a mexer
   precisa saber contra o quê aquilo foi calibrado.

## Separe o defeito seu do defeito da fonte

Ao olhar texto sujo, classifique antes de consertar:

- **Nosso, corrigível** — ordem de operações na limpeza, extrator mal escolhido,
  fronteira descartada. Exemplo: hífen suave apagado antes da regra que trataria
  a quebra, transformando `pode<AD>\nmos` em `pode mos`.
- **Da fonte, incorrigível** — OCR torto, legenda sem pontuação, página montada
  por template. Diga isso ao Operator e pare.

Limpar com modelo **não é opção**: paráfrase de um modelo pequeno vira fato errado
permanente dentro do índice, invisível na hora de responder.

## Prove que o teste pega o defeito

Duas verificações que salvaram esta sessão:

- **O ramo que você está corrigindo é alcançável?** Escrevi uma correção inteira
  dentro de um `if` morto por construção. O teste passou verde sem exercitar nada.
- **Reverta a correção e rode o teste.** Se ele não falhar, ele não testa nada.
  `cp` do arquivo, desfaz a linha, roda, restaura.

## O ciclo

Correção de ingestão **não vale para o que já está indexado**. Depois de cada
uma: apagar o documento, reingerir, medir de novo. Se o ciclo repetir muito,
levante a falta de uma rota de reindexação como pendência — é sinal de que o
gargalo virou o processo, não o código.

## Armadilhas de operação

- Editou `config/*.json` → `node scripts/validate-contracts.mjs --write` e commite
  o digest. Sem isso a falha aparece longe, como `DatasetDriftError` em testes que
  não têm relação com a mudança.
- `pkill -f "venv/bin/harness"` **mata o próprio shell** — o padrão casa com a
  linha de comando do bash. Use `pgrep -f ... | head -1` e `kill`.
- Suba o servidor com `setsid nohup ... < /dev/null &`, senão ele morre junto com
  a sessão de comando.
- Testar helper privado quebra o pyright strict (`reportPrivateUsage`). Tornar a
  função pública e declará-la no `__all__` é a saída aceita neste repo.
