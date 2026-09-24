# Documentação do Harness 2.0

Contrato vigente do Harness 2.0. A primeira release tem um único RuntimeProfile
funcional, local via Ollama; demais runtimes e modelos não pertencem
automaticamente ao roadmap.

## Ordem de leitura

1. [Glossário canônico](../CONTEXT.md) — linguagem de domínio usada pelos
   contratos.
2. [Decisões normativas do 2.0](DECISOES-2.0.md) — baseline vigente,
   invariantes e critérios formais para mudar uma decisão.
3. [Pendências da primeira release](RELEASE-PENDING.md) — exclusões, motivos,
   gates e limitações residuais.
4. [ADRs](adr/) — fronteiras arquiteturais difíceis de reverter.
5. [Modelo de ameaça da autenticação](THREAT-MODEL-AUTH.md) — ativos, ameaças,
   mitigações e o que fica fora do modelo na primeira release.

## Contratos executáveis

- [Modelfile do perfil core (Gemma 4 E4B QAT)](../modelfiles/gemma4_e4b_qat_ollama.Modelfile)
- [RuntimeProfiles e evidência](../config/model-profiles.json)
- [Configuração normativa do harness](../config/harness.json)
- [Registry canônico de tools, automações e proibições](../config/tool-registry.json)
- [Corpus de regressões](../evals/fixtures/regressions.json)
- [Matriz de experimentos](../evals/experiments.json)
- [Como executar as avaliações](../evals/README.md)

Validação local, sem dependências externas:

```bash
node scripts/validate-contracts.mjs
```

O perfil instalado já foi recriado a partir do `Modelfile`; seu digest e a
evidência de coerência estão em `model-profiles.json`. Recriação futura é efeito
operacional deliberado e não faz parte da validação documental.

## Decisões que não devem se perder

- Separar RuntimeProfile de ExecutionRoute; identidade de runtime não é policy
  de execução.
- Fixar pesos, tokenizer/template, backend e parsers como uma única versão do
  sistema.
- Herdar o `TEMPLATE` de chat do artefato base sem sobrescrever, em vez de
  presumir o Hermes do Qwen3 anterior. Isso é o chat template do modelo, não o
  formato do ModelView — esse está congelado por execução em
  `model_view_format`.
- No perfil local, começar com `temperature=0.3`, `presence_penalty=0`, contexto
  65.536 e o thinking já medido; toda mudança é um braço de bancada.
- Tratar seleção, argumentos e resultados do modelo como não confiáveis;
  validação, autorização, confirmação e sandbox pertencem ao harness.
- Tratar ModelView e AG-UI como projeções; CanonicalHistory permanece a fonte
  autoritativa e reasoning transitório não é persistido.

## Estado da evidência

Somente `DECISOES-2.0.md` e os contratos JSON definem a release atual. Os
documentos de pesquisa e a herança empírica do Harness 1.0 foram retirados do
repositório; a proveniência dos fatos que sobreviveram está declarada nos
próprios contratos (`config/model-profiles.json`,
`evals/fixtures/regressions.json`) e o texto integral permanece no git.
