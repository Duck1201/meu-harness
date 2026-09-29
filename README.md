# Meu Harness

Harness de agente LLM local, web-first, com policy por efeito. O modelo é tratado
como não confiável: seleção de tool, argumentos e resultados passam por validação,
autorização, confirmação e sandbox do harness. Autorização é resolvida pelo efeito
declarado (`workspace_read`, `workspace_write`, `data_egress`, `corpus_read`,
`pure_compute`), nunca pelo nome da tool.

Backend FastAPI + SQLite, frontend React servido pelo próprio backend, modelo local
via Ollama. O contrato normativo está em [`docs/DECISOES-2.0.md`](docs/DECISOES-2.0.md);
o que deliberadamente não entra na primeira release está em
[`docs/RELEASE-PENDING.md`](docs/RELEASE-PENDING.md).

## Requisitos

| Componente | Versão | Observação |
|---|---|---|
| uv | qualquer | Cria o `.venv` e roda todo comando deste README (`uv run …`) |
| Python | 3.13 (`>=3.13,<3.14`) | A plataforma é contrato, não sugestão |
| Sistema | Linux x86_64 | Outras plataformas não são suportadas na v1 |
| Node.js | 22+ | Só para construir o frontend e validar contratos |
| pnpm | 10.13.1 | Via `corepack enable` |
| Ollama | 0.32.5 | Servindo em `http://127.0.0.1:11434` |
| Chromium | qualquer | Opcional, só para a escalação do `web_fetch`. Declare o caminho em Configurações; sem isso o harness procura `brave-browser`, `brave-browser-stable`, `brave`, `chromium`, `chromium-browser` e `google-chrome` no PATH, nessa ordem |
| Docker ou Podman | qualquer | Opcional, só se você quiser SearXNG self-hosted para o `web_search` |

## Instalação

Duas formas, mesmo resultado. Escolha uma.

### Opção A: script

```bash
scripts/bootstrap.sh
```

Idempotente — pode rodar de novo sem duplicar trabalho. Faz, nesta ordem:

1. `uv sync` — instala as dependências Python e cria `.venv`.
2. `corepack enable` + `pnpm install && pnpm build` em `web/` — gera `web/dist`.
3. Lê o perfil ativo (`active_runtime_profile` em `config/model-profiles.json`),
   hoje `gemma4_e4b_qat_ollama`: baixa o modelo base (`gemma4:e4b-it-qat`), cria a
   tag `harness-gemma4-e4b-qat` a partir de
   [`modelfiles/gemma4_e4b_qat_ollama.Modelfile`](modelfiles/gemma4_e4b_qat_ollama.Modelfile)
   (só se ela ainda não existir) e confere o digest instalado contra
   `installation.installed_profile_digest_sha256`, avisando no stderr se divergir.
4. Baixa o `tokenizer.json` do perfil (`installation.tokenizer_url`) para
   `$XDG_STATE_HOME/meu-harness/tokenizer-<perfil>.json` (ou `~/.local/state/meu-harness/`
   sem `XDG_STATE_HOME`), só se o arquivo ainda não existir ali.

Qualquer etapa que precise de uma ferramenta ausente no PATH (`corepack`, `ollama`)
é pulada com aviso — o script não falha por isso.

### Opção B: passo a passo manual

```bash
uv sync
```
Instala as dependências Python e cria `.venv`.

```bash
corepack enable && cd web && pnpm install && pnpm build && cd ..
```
Gera `web/dist`, servido pelo próprio backend.

Para reinstalar de uma vez todos os modelos que o harness usa, rode
`scripts/install-models.sh`. Cada um é uma tag `harness-*` criada de um Modelfile
versionado em [`modelfiles/`](modelfiles/) sobre a base oficial:

| Tag | Modelfile | Base | Papel |
|---|---|---|---|
| `harness-gemma4-e4b-qat` | `gemma4_e4b_qat_ollama.Modelfile` | `gemma4:e4b-it-qat` | perfil core |
| `harness-bge-m3` | `bge_m3_embedding.Modelfile` | `bge-m3:latest` | embedding do Corpus |
| `harness-vision-qwen35-2b` | `qwen35_2b_vision.Modelfile` | `qwen3.5:2b-q4_K_M` | tool de visão |
| `harness-judge-ptbr-v2` | `harness_judge_ptbr_v2.Modelfile` | GGUF do `~/judge-train` | juiz do Corpus |
| `harness-ocr-glm` | `glm_ocr.Modelfile` | `glm-ocr:latest` | OCR de PDF escaneado |

Só o perfil core, à mão:

```bash
ollama pull gemma4:e4b-it-qat
ollama create harness-gemma4-e4b-qat -f modelfiles/gemma4_e4b_qat_ollama.Modelfile
curl -s http://127.0.0.1:11434/api/tags \
  | python3 -c "import json,sys;print(next((m['digest'] for m in json.load(sys.stdin)['models'] if m['name']=='harness-gemma4-e4b-qat:latest'), 'não instalado'))"
```
Cria o perfil de execução a partir do Modelfile versionado do perfil ativo. O Gemma4
E4B QAT é o modelo core desde 23/09/2026: superou o `mitos` (41/50 contra 33/50), que
foi aposentado junto com os demais Challengers (`status: retired` em
`config/model-profiles.json`, modelos e Modelfiles removidos). O segundo
comando imprime o digest do manifesto que o Ollama atribuiu ao modelo instalado,
e ele deve bater com `installation.installed_profile_digest_sha256` em
`config/model-profiles.json` — divergência é erro na inicialização, não aviso. É
esse mesmo campo que `OllamaRuntime.verify_profile` compara com `/api/tags`, por
isso o digest vem de lá e não de um hash do texto do Modelfile: são valores
diferentes, e só um deles identifica os pesos e os parâmetros de fato instalados.

```bash
mkdir -p "${XDG_STATE_HOME:-$HOME/.local/state}/meu-harness"
curl -fsSL \
  https://huggingface.co/google/gemma-4-E4B-it/resolve/main/tokenizer.json \
  -o "${XDG_STATE_HOME:-$HOME/.local/state}/meu-harness/tokenizer-gemma4_e4b_qat_ollama.json"
```
Baixa o `tokenizer.json` para o mesmo caminho que o setup vai sugerir por padrão
(veja abaixo). Pode ir para qualquer outro caminho, desde que informe esse caminho
no setup.

### Tokenizer

O orçamento de contexto (65536 tokens) depende de uma contagem real de tokens, não
de estimativa por caractere. O harness carrega um `tokenizer.json` no formato
HuggingFace: sem o arquivo, a aplicação sobe com
`EngineReadiness(ready=False, reason_code="tokenizer_file_missing")` e recusa
execuções. O arquivo não é versionado neste repositório (é grande, e o conteúdo
correto depende de qual revisão do modelo você instalou). Download direto:
[`tokenizer.json`](https://huggingface.co/google/gemma-4-E4B-it/resolve/main/tokenizer.json)
do repositório do modelo base do perfil ativo (`installation.tokenizer_url`). Trocar
o perfil funcional troca o tokenizer: Acervos já indexados continuam válidos, mas
os novos passam a ser divididos contando tokens do modelo novo.

O SHA-256 do tokenizer **não é digitado no setup**: o harness calcula o digest do
arquivo apontado e grava esse valor em `host.json#tokenizer_digest`, usado depois
em todo boot para detectar o arquivo trocado.

## Primeira execução

```bash
uv run harness                    # ou: uv run harness --port 8765 --host 127.0.0.1
```

**Suba o servidor a partir da raiz do repositório.** `web/dist` e
`SYSTEM-PROMPT.md` são resolvidos relativos ao diretório de trabalho: de outro
lugar o painel responde 404 e o texto do Operator some do prompt em silêncio,
sem erro nenhum.

Na primeira execução o servidor imprime um **token de setup efêmero** no stderr
(TTL de 600 s). Abra `http://127.0.0.1:8765/setup`: o formulário já vem
pré-preenchido com o diretório de estado e o caminho do tokenizer sugeridos (o
mesmo que o `bootstrap.sh` usa) e com a origin atual mais a contraparte de
loopback dela (`127.0.0.1` ↔ `localhost`) — falta só colar o token,
apontar as raízes de workspace e confirmar. Sem interface, o mesmo é feito por
`POST /api/setup` com o header `X-Harness-Setup-Token`, informando
`allowed_workspace_roots`, `tokenizer_path`, `state_dir`, `allowed_origins` e,
opcionalmente, `searxng_url` e `ollama_url`. `GET /api/setup/status` diz se ainda é
necessário e devolve os valores sugeridos em `suggested_state_dir` e
`suggested_tokenizer_path`.

- `allowed_workspace_roots` — diretórios que as tools de arquivo do agente (`workspace_read`/`workspace_write`) podem tocar, um caminho absoluto por linha. É o sandbox: caminho fora dessas raízes é recusado mesmo que o modelo peça. Tipicamente o(s) diretório(s) de projeto que você vai trabalhar com o agente.
- `allowed_origins` — de onde o navegador pode chamar esta API: protocolo, host e porta (ex. `http://127.0.0.1:8765`), um por linha. Requisição HTTP com header `Origin` fora dessa lista é recusada. Normalmente é só a própria origin em que você está acessando o painel — a UI já pré-preenche com ela.

Depois de configurado, o setup só reabre com `harness --setup`. Para mudar qualquer
desses valores sem reabrir o setup, use a aba **Configurações** do painel.

## Configuração

`host.json` é a fonte única: o setup grava, a aba **Configurações** do painel
reescreve, e o servidor lê no boot. Não há variável de ambiente equivalente —
editar no painel sempre tem efeito, e o que está no arquivo é o que vale.

Só o que precisa existir antes do app entra por flag da CLI:

| Flag | Default | Efeito |
|---|---|---|
| `--host` | `127.0.0.1` | Interface do servidor. Ver "Exposição de rede" |
| `--port` | `8765` | Porta do servidor |
| `--host-config` | padrão do XDG | Caminho do `host.json` |
| `--setup` | desligado | Reabre o setup numa instalação já configurada |

O que vive dentro do `host.json` e é editável no painel: raízes de Workspace,
origins autorizadas, caminho do tokenizer (o digest é medido do arquivo),
diretório de estado, URL do Ollama, a instância SearXNG opcional e o executável
do navegador. Salvar grava o
arquivo e devolve `restart_required`: nada é reconstruído a quente, então reinicie
o servidor para aplicar. A senha de Operator continua na rota própria
(`PUT /api/admin/operator-password`) e é o único segredo do host — as três rotas
de administração (`/api/admin/operator-password`, `/api/admin/host-config` e
`/api/admin/yolo`) exigem sessão, ou loopback direto enquanto não houver senha.

`config/harness.json` e os demais contratos não são ajustáveis pelo host: são
selados por digest.

### Busca: SearXNG opcional

`web_search` funciona sem configuração nenhuma — o fallback é o DuckDuckGo, que
não pede chave. Uma instância SearXNG entra no lugar dele quando você declara a
URL em **Configurações**, e a resposta em JSON é melhor: menos scraping, mais
metadados e ranking que você controla.

Self-hosted, com Docker ou Podman:

```bash
scripts/searxng.sh          # sobe em 127.0.0.1:8080 e já aplica a configuração
scripts/searxng.sh --down   # para e remove o container
```

O script é idempotente: rodar de novo reaplica a configuração e religa o
container sem duplicar nada, preservando o `secret_key` do primeiro boot. Ele
termina verificando a instância de verdade e falha se ela não devolver JSON.
`PORT`, `NAME`, `CONFIG_DIR` e `IMAGE` são variáveis de ambiente se você quiser
outra coisa que não `~/searxng` na 8080.

Se preferir na mão, o que o script faz é subir o container e reescrever
`~/searxng/settings.yml` com **duas edições obrigatórias** — `limiter: false` e
`json` nos formatos —, sem as quais toda busca cai no fallback. Um arquivo
reescrito também precisa herdar os padrões, ou a instância nem sobe:

```yaml
use_default_settings: true   # sem esta linha o arquivo substitui os padrões e a instância não sobe

server:
  secret_key: "o que o primeiro boot gerou"
  image_proxy: true
  limiter: false     # o harness fala HTTP direto, não é navegador

search:
  formats:
    - html
    - json           # o padrão do SearXNG é só html
```

O arquivo pertence ao usuário do container, então edite por dentro dele:

```bash
RUNTIME=docker              # ou podman, o mesmo que o script escolhe
$RUNTIME exec -u 0 -it searxng vi /etc/searxng/settings.yml
$RUNTIME restart searxng
curl -s "http://127.0.0.1:8080/search?q=harness&format=json" | head -c 200
```

Saiu JSON com `results`? Cole `http://127.0.0.1:8080/search` em **Configurações →
Host → Instância SearXNG**, salve e reinicie o harness.

Uma instância em loopback é o caso normal. O `EgressGuard` não recusa destinos
locais nem privados: `web_fetch`, a coleta de Corpus e o navegador alcançam a sua
máquina e a sua rede local ([ADR-0017](docs/adr/0017-egress-allows-private-destinations.md)).

Instância pública também serve — `searx.space` lista várias —, mas a maioria
desliga `format=json` justamente contra bots. Teste com o `curl` acima antes de
configurar.

### Navegador: por que ele mora na sua máquina

O `web_fetch` tenta HTTP primeiro e escala para um navegador em dois casos: a
extração legível da página ficou abaixo de 120 caracteres — o que uma página
renderizada por JavaScript parece para um cliente HTTP — ou o servidor respondeu
`401`/`403`. Esse navegador é um binário do host, e isso
é deliberado, não um resto de configuração: cada operação sobe um processo novo
com um `--user-data-dir` descartável e com o destino fixado em
`--host-resolver-rules` a partir do endereço que o `EgressGuard` já validou — o
navegador não resolve DNS por conta própria. Um browser remoto, em container ou
não, quebraria as duas garantias de uma vez ([ADR-0003](docs/adr/0003-guarded-egress-browser-isolation.md)).

O que era palpite e deixou de ser: qual binário. Declare o caminho em
**Configurações → Host → Executável do navegador**; o valor é validado (absoluto,
existente, executável) e gravado no `host.json`. Deixando vazio, o harness varre
o PATH como antes, e aí o comportamento passa a depender da máquina — onde nenhum
Chromium for encontrado, a chamada termina em `failed` com
`browser_escalation_unavailable` e `retryable: false`. Nada do que o HTTP trouxe
volta, e não haveria o que voltar: a escalação só dispara nos dois casos acima,
em que o HTTP não entregou conteúdo utilizável.

```bash
which chromium || which brave-browser || which google-chrome
```

### Exposição de rede

A exposição é derivada da credencial, não de uma flag:

| Estado | Comportamento |
|---|---|
| Sem senha de Operator | **Todo** path exige conexão de loopback direto, a SPA e seus assets estáticos inclusive. Qualquer outra conexão recebe `401 authentication_required`. |
| Com senha de Operator | `/api/` exige o header `X-Harness-Session` obtido em `POST /api/session`. A SPA fica aberta de propósito: é a tela de login, e um cliente remoto precisa carregá-la antes de poder ter sessão. |

Não existe configuração que abra a porta para a rede sem autenticação. Exceções
de rota, e por quê: `GET /api/health` (responde a supervisor antes de haver
login), `/api/setup` e `/api/setup/status` (guardadas pelo token efêmero e
restritas a loopback) e `/api/session` em todos os métodos — `POST` é o login,
`GET` diz se a sessão ainda vale e `DELETE` é o logout.

Definir a primeira senha:

```bash
curl -X PUT http://127.0.0.1:8765/api/admin/operator-password \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:8765' \
  -d '{"password": "pelo-menos-doze-caracteres"}'
```

Rotacionar exige a senha atual em `current_password`, além da sessão:

```bash
curl -X PUT http://127.0.0.1:8765/api/admin/operator-password \
  -H 'Content-Type: application/json' \
  -H 'Origin: http://127.0.0.1:8765' \
  -H "X-Harness-Session: $SESSION" \
  -d '{"current_password": "a-senha-de-agora", "password": "a-nova-senha"}'
```

Trocar a senha invalida todas as sessões abertas. O hash é PBKDF2-HMAC-SHA256
com 600.000 iterações; a senha original nunca é gravada nem devolvida.

O modelo de ameaça está em [`docs/THREAT-MODEL-AUTH.md`](docs/THREAT-MODEL-AUTH.md).

### Reverse proxy

Para publicar além do loopback, configure a senha e coloque um proxy com TLS à
frente. O harness recusa qualquer requisição que traga `Forwarded`, `Via`,
`X-Real-IP` ou `X-Forwarded-*` como se fosse local, então o proxy nunca consegue
se passar por conexão direta — mas você precisa incluir a origem pública nas
origins autorizadas, pela aba **Configurações** do painel:

```nginx
server {
  listen 443 ssl;
  server_name harness.example;

  location / {
    proxy_pass http://127.0.0.1:8765;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $remote_addr;
    proxy_http_version 1.1;
    proxy_buffering off;              # o stream AG-UI é SSE
    proxy_read_timeout 930s;          # acima de loop.max_turn_duration_seconds
  }
}
```

Depois de salvar `https://harness.example` na lista de origins, reinicie:

```bash
uv run harness
```

## System prompt

O prompt é derivado dos contratos (`src/harness/system_prompt.py` e as
capacidades em `config/model-profiles.json`), com a data do host a cada Turn.
Para ver o prompt exato que o modelo recebe hoje:

```bash
uv run python scripts/show-system-prompt.py
```

`SYSTEM-PROMPT.md`, na raiz, é o seu texto: tudo o que estiver ali, fora de
comentários HTML (`<!-- ... -->`), é anexado ao fim do prompt. Arquivo vazio ou
ausente deixa o prompt como o derivado, byte a byte. O limite é 4000 caracteres,
porque ele entra em todo Turn e disputa o orçamento de contexto com o histórico.

Cada execução de eval congela o digest desse texto junto dos digests de
contrato: dois braços com textos de Operator diferentes não são o mesmo sistema
e não devem ser comparados como se fossem.

## Uso

Abra `http://127.0.0.1:8765`. O produto é a superfície web — CLI e desktop estão
fora do roadmap da v1 (`docs/RELEASE-PENDING.md`).

Cada Conversation começa sem autoridade nenhuma. Ler arquivos exige um
`WorkspaceRootGrant`, escrever exige também `WriteGrant`, e qualquer saída de dados
exige `WebAccessGrant`. O modelo não concede nem amplia acesso: quem concede é o
Operator, pela interface.

Grant não é a última palavra: alguns efeitos ainda param para confirmação. Toda
escrita mostra o preview do diff antes de acontecer, e o `data_egress` passa a
pedir decisão quando o Turn está sob taint — o modelo leu uma página, e mandar
esse conteúdo para fora vira uma pergunta. O waiver do Operator vale por
Conversation e cobre a escrita, nunca a chamada sob taint.

**Modo yolo** é a única saída dessa disciplina, e é decisão do Operator: global,
desligada de fábrica, com opt-out por Conversation. Ligada, ela aprova toda
confirmação que o gate saiba fazer — inclusive a escrita sob taint — e registra
cada chamada como `waived`, nunca como `approved`
([ADR-0008](docs/adr/0008-operator-yolo-mode.md)).

As tools expostas: `read_file`, `write_file`, `edit`, `list_directory`, `glob`,
`grep_search`, `web_search`, `web_fetch`, `corpus_search`, `calculate`,
`get_weather` e, com a visão ligada, `describe_image`. O `edit` troca um trecho
exato (`old_string` por `new_string`, com `replace_all` para todas as ocorrências)
e devolve o trecho editado; nem ele nem o `write_file` pedem SHA-256.

### Visão: `describe_image`

O modelo de chat não recebe imagens. `describe_image(file_path, question)` lê uma
imagem PNG, JPEG ou WebP do Workspace e pergunta a um modelo de visão local, o
Qwen3.5-2B, fixado por digest em `config/harness.json#vision`
([ADR-0016](docs/adr/0016-local-vision-tool.md)). A resposta volta com um aviso de
que números e texto miúdo podem estar errados. Vem ligada desde a promoção de
`vision_tool_enablement`; precisa do modelo no Ollama:

```bash
ollama pull qwen3.5:2b-q4_K_M
ollama create harness-vision-qwen35-2b -f modelfiles/qwen35_2b_vision.Modelfile
uv run python scripts/vision-bench.py                  # mede nas imagens de evals/vision
uv run python scripts/vision-bench.py --cases <pasta>  # mede nos seus prints (cases.json + imagens)
```

Para desligar, use `"mode": "disabled"` em `config/harness.json#vision`.

Para testar pelo chat, anexe a imagem no campo de mensagem (botão **imagem**, ou
cole com Ctrl+V). Ela é gravada em `anexos/` no Workspace da Conversation
(`POST /api/conversations/{id}/attachments`, só PNG, JPEG e WebP, conferidos pelos
bytes, até `vision.max_image_bytes`), e a mensagem leva o caminho para o modelo
chamar o `describe_image`.

### RAG: acervos por Corpus

A aba **RAG** monta acervos de documentos. Cada Corpus é um arquivo SQLite em
`$XDG_STATE_HOME/meu-harness/corpora/`, isolado dos outros e dos dois stores do
harness; apagar o Corpus apaga o arquivo, e a retenção de conversas não o alcança.

Alimente por upload (`.txt`, `.md`, `.html`, `.pdf`) ou por coleta web. Uma semente
que responde `/api.php` é coletada pela API do MediaWiki, que devolve a wiki inteira
em texto puro; qualquer outro site cai num crawl com teto que respeita `robots.txt`.
A coleta roda em segundo plano, mostra progresso e pode ser cancelada; disparar de
novo pula o que já entrou.

PDF escaneado, sem camada de texto, passa por OCR
([ADR-0018](docs/adr/0018-ocr-for-scanned-pdfs.md)): cada página é renderizada pelo
`pdftoppm` (pacote `poppler-utils`) e lida pelo GLM-OCR (`harness-ocr-glm`, de
`modelfiles/glm_ocr.Modelfile`), em segundo plano, com progresso por página
(~10 s por página na RX 7600). O texto entra marcado como transcrição automática, e
a passagem avisa o modelo de que pode ter letra ou dígito trocado. PDF que já tem
camada de texto não passa por OCR: medido no Kurose, o OCR trocava os erros da
camada por outros e perdia acentos. Sem o `pdftoppm` ou o modelo, o PDF escaneado
continua recusado.

Cada Conversation escolhe um Corpus ou "Desligado", no painel de contexto do chat.
A escolha é o `CorpusGrant`: enquanto estiver ligada, o harness recupera as passagens antes
do modelo responder, a resposta cita `[1]`, `[2]` e o card do turno mostra o trecho
literal com o documento, a seção e a página. Passagem coletada da web fica marcada
e faz o harness voltar a pedir confirmação para sair de novo à rede.

O modelo de embedding é o `bge-m3`, instalado como `harness-bge-m3` (`ollama pull
bge-m3` e `ollama create harness-bge-m3 -f modelfiles/bge_m3_embedding.Modelfile`),
declarado com digest próprio em `config/model-profiles.json`. Sem ele instalado, a aba diz isso em vez
de fingir um acervo vazio.

Dois interruptores em `config/harness.json#corpus` mudam a coleta
([ADR-0014](docs/adr/0014-corpus-browser-route-scrapling.md)). Medidos em
23/09/2026, a rota de browser está ligada e o extrator Scrapling continua
desligado:

- `ingestion.html_extractor: "scrapling"` passa o HTML pela limpeza do
  [Scrapling](https://scrapling.readthedocs.io/) antes de indexar — some o texto
  escondido por `display:none`, `aria-hidden` ou `<template>`, que é onde uma
  página hostil esconde instrução para o modelo. O download continua no
  transporte guardado do harness.
- `scraper.html_crawl.browser_escalation.mode: "symptom"` renderiza num Chromium
  efêmero a página que o HTTP trouxe sem texto (site montado em JavaScript). Toda
  requisição do browser passa pelo mesmo guard de rede. Precisa do extra:

  ```bash
  uv sync --extra browser
  uv run scrapling install     # baixa o Chromium do Scrapling
  ```
- `answer_judge.mode: "advisory"` (o padrão desde a promoção de 23/09/2026) liga um juiz local no molde do Jev
  ([ADR-0015](docs/adr/0015-corpus-answer-judge.md)): o Qwen3-Reranker-0.6B lê
  cada passagem recuperada e anota a probabilidade de ela trazer o que foi
  perguntado; se nenhuma trouxer, o modelo é avisado de que o acervo
  provavelmente não responde. Ele nunca bloqueia nem reescreve a resposta. Roda
  no mesmo Ollama, conferido por digest, e ocupa ~750 MB de VRAM:

  ```bash
  ollama pull hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0
  ```

  Chat, `bge-m3` e juiz cabem juntos em 8 GB, mas só se o Ollama puder manter
  três modelos carregados — com `OLLAMA_MAX_LOADED_MODELS=2` o juiz desloca o
  modelo de chat a cada Turn. Suba para 3 com `sudo systemctl edit ollama`.

## Desenvolvimento

```bash
node scripts/validate-contracts.mjs        # coerência entre docs e contratos JSON
node scripts/validate-contracts.mjs --write  # resela os digests derivados após editar contratos
uv run ruff check . && uv run ruff format --check .
uv run pyright
uv run pytest
cd web && pnpm test && pnpm exec tsc -b
```

Os JSON de `config/` e `evals/` carregam digests derivados do próprio conteúdo.
Depois de editar qualquer contrato ou fixture, rode o validador com `--write` para
reselar e commite a mudança de digest junto.

## Avaliações

`evals/fixtures/regressions.json` guarda o corpus de regressões determinísticas e
`evals/experiments.json` a matriz de experimentos. O protocolo está em
[`evals/README.md`](evals/README.md): piloto de 15 execuções por braço (direcional),
promoção com 50 execuções por braço e 3 seeds registradas. Violação de segurança
reprova imediatamente, independente do resto.

Nada disso é só arquivo: a aba **Avaliações** do painel lista os experimentos,
dispara execuções e mostra os relatórios, sobre o mesmo runner que responde em
`POST /api/evals/runs`.

Dois experimentos travavam a release e os dois foram promovidos em 15/08/2026:
`guarded_web_brave_escalation` (50 casos por braço, todos passam) e
`corpus_retrieval_vs_baseline` (24/50 com acervo contra 1/50 sem, e 30 alegações
sem lastro contra 55 do baseline). `model_view_serialization` rodou no mesmo dia e
reprovou o candidato, mantendo o JSON como formato do harness — a reversão de
ADR-0010. Os demais travam **mudanças de baseline** (temperatura, thinking, limite
de passos, idioma do prompt, verificação de página) e não são dívida de release:
enquanto não rodarem, o baseline vigente permanece o que está no contrato.

Medição inventada para preencher contrato é proibida (`docs/DECISOES-2.0.md`). Um
experimento sem execução fica com `result: null`.

### Bake-off de modelos e o llama.cpp

Outros modelos entram como **Challengers**: perfis instalados e medidos que
nenhuma rota de produção seleciona, comparados ao perfil funcional pelo mesmo
protocolo. Os que o Ollama não serve direito rodam no `llama-server` do llama.cpp
([ADR-0013](docs/adr/0013-second-local-runtime-llama-cpp.md)). Para a bancada
subir esses perfis, declare no `host.json` o binário e onde está cada GGUF (a aba
Configurações preserva esses campos ao salvar):

```json
{
  "llama_server_executable": "/home/voce/.local/opt/llama.cpp/llama-server",
  "gguf_paths": {"cove_4b_llamacpp": "/home/voce/.local/share/meu-harness/gguf/CoVe-4B.Q4_K_M.gguf"}
}
```

Cada perfil mede o orçamento com o próprio tokenizer, em
`.harness/tokenizers/<runtime_profile_id>.json`. Com isso:

```bash
uv run python scripts/run-experiment.py --tier model_smoke --profile cove_4b_llamacpp
uv run python scripts/run-experiment.py runtime_profile_bakeoff --phase pilot
```

Para servir a rota por um perfil llama.cpp, suba o servidor com
`scripts/llama-server.sh`: ele monta a linha de comando a partir do `host.json`
e dos `server_args` do perfil, e o harness confere o digest do GGUF carregado
antes do primeiro Turn.

## Licença

[Apache-2.0](LICENSE).

## Documentação

Ordem de leitura em [`docs/README.md`](docs/README.md). Em resumo: o glossário
canônico está em [`CONTEXT.md`](CONTEXT.md), o baseline normativo em
`docs/DECISOES-2.0.md`, as exclusões da release em `docs/RELEASE-PENDING.md`, o
modelo de ameaça da autenticação em `docs/THREAT-MODEL-AUTH.md` e as fronteiras
arquiteturais em `docs/adr/`.
