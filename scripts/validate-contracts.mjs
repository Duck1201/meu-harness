#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const failures = [];

function check(condition, message) {
  if (!condition) failures.push(message);
}

function readJson(relativePath) {
  try {
    return JSON.parse(fs.readFileSync(path.join(root, relativePath), "utf8"));
  } catch (error) {
    failures.push(`${relativePath}: JSON inválido ou ilegível: ${error.message}`);
    return null;
  }
}

function unique(values, label) {
  const seen = new Set();
  for (const value of values) {
    check(typeof value === "string" && value.length > 0, `${label}: identificador vazio`);
    check(!seen.has(value), `${label}: valor duplicado: ${value}`);
    seen.add(value);
  }
  return seen;
}

function sameValues(left, right) {
  return JSON.stringify([...left].sort()) === JSON.stringify([...right].sort());
}

function canonicalJson(value) {
  if (value === null || typeof value !== "object") return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
    .join(",")}}`;
}

function sha256(value) {
  return crypto.createHash("sha256").update(value).digest("hex");
}

function documentDigest(document) {
  return sha256(canonicalJson(document));
}

function scopedDigest(document, label) {
  const contract = document.digest_contract;
  check(contract?.algorithm === "sha256", `${label}: digest deve usar sha256`);
  check(
    contract?.canonicalization === "sorted_keys_compact_utf8",
    `${label}: canonicalização de digest inválida`,
  );
  check(Array.isArray(contract?.scope), `${label}: digest_contract.scope ausente`);
  const selected = {};
  for (const field of contract?.scope ?? []) {
    check(Object.hasOwn(document, field), `${label}: campo de digest ausente: ${field}`);
    selected[field] = document[field];
  }
  return sha256(canonicalJson(selected));
}

function replaceDigestField(text, key, value, label) {
  const pattern = new RegExp(`("${key}"\\s*:\\s*")[0-9a-f]{64}(")`);
  if (!pattern.test(text)) {
    failures.push(`${label}: campo de digest não encontrado para reescrita: ${key}`);
    return text;
  }
  return text.replace(pattern, `$1${value}$2`);
}

// Reescreve os digests derivados a partir do conteúdo atual dos contratos. A substituição é
// textual e cirúrgica porque os JSON usam arrays compactos que um JSON.stringify desfaria,
// transformando cada selagem num diff do arquivo inteiro.
function sealDigests() {
  const fixturesPath = "evals/fixtures/regressions.json";
  const experimentsPath = "evals/experiments.json";
  let fixturesText = fs.readFileSync(path.join(root, fixturesPath), "utf8");
  let experimentsText = fs.readFileSync(path.join(root, experimentsPath), "utf8");

  const datasetDigest = scopedDigest(JSON.parse(fixturesText), "fixtures");
  fixturesText = replaceDigestField(
    fixturesText,
    "dataset_digest_sha256",
    datasetDigest,
    fixturesPath,
  );
  experimentsText = replaceDigestField(
    experimentsText,
    "dataset_digest_sha256",
    datasetDigest,
    experimentsPath,
  );
  for (const [field, contractPath] of [
    ["model_profiles_sha256", "config/model-profiles.json"],
    ["harness_sha256", "config/harness.json"],
    ["tool_registry_sha256", "config/tool-registry.json"],
  ]) {
    const digest = documentDigest(JSON.parse(fs.readFileSync(path.join(root, contractPath), "utf8")));
    experimentsText = replaceDigestField(experimentsText, field, digest, experimentsPath);
  }

  // O manifesto cobre dataset e contract_digests, então só fecha depois deles.
  const manifestDigest = scopedDigest(JSON.parse(experimentsText), "experiments");
  experimentsText = replaceDigestField(
    experimentsText,
    "manifest_digest_sha256",
    manifestDigest,
    experimentsPath,
  );

  if (failures.length > 0) return;
  fs.writeFileSync(path.join(root, fixturesPath), fixturesText);
  fs.writeFileSync(path.join(root, experimentsPath), experimentsText);
  console.log(`Digests selados: ${fixturesPath}, ${experimentsPath}`);
}

if (process.argv.includes("--write")) sealDigests();

function collectMarkdownFiles(directory) {
  const result = [];
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) result.push(...collectMarkdownFiles(absolutePath));
    if (entry.isFile() && entry.name.endsWith(".md")) result.push(absolutePath);
  }
  return result;
}

function collectGithubHeadingAnchors(content) {
  const anchors = new Set();
  const occurrences = new Map();
  for (const line of content.split("\n")) {
    const match = /^(#{1,6})\s+(.+?)\s*#*\s*$/.exec(line);
    if (!match) continue;
    const base = match[2]
      .toLocaleLowerCase("pt-BR")
      .replace(/<[^>]+>/g, "")
      .replace(/[`*~]/g, "")
      .replace(/[^\p{L}\p{N}\s_-]/gu, "")
      .trim()
      .replace(/\s+/g, "-");
    const seen = occurrences.get(base) ?? 0;
    occurrences.set(base, seen + 1);
    anchors.add(seen === 0 ? base : `${base}-${seen}`);
  }
  return anchors;
}

// `1.0` é o mesmo número em JSON e dois digests diferentes: o canonicalizador JS
// escreve `1`, o do Python escreve `1.0`, e a divergência só aparece longe daqui,
// como drift de contrato no carregamento dos evals. Recusar a grafia é mais barato
// que ensinar os dois lados a concordar sobre ela.
// A regra vale só para o que o digest cobre: `results` guarda medição registrada,
// fica fora do escopo e não se reescreve para agradar um canonicalizador.
function withoutRecordedResults(text) {
  const start = text.indexOf('"results"');
  if (start < 0) return text;
  const open = text.indexOf("[", start);
  if (open < 0) return text;
  let depth = 0;
  for (let index = open; index < text.length; index += 1) {
    if (text[index] === "[") depth += 1;
    if (text[index] === "]") depth -= 1;
    if (depth === 0) return text.slice(0, open) + text.slice(index + 1);
  }
  return text.slice(0, open);
}

for (const contractPath of [
  "config/model-profiles.json",
  "config/harness.json",
  "config/tool-registry.json",
  "evals/fixtures/regressions.json",
  "evals/experiments.json",
]) {
  const scoped = withoutRecordedResults(
    fs.readFileSync(path.join(root, contractPath), "utf8"),
  );
  for (const match of scoped.matchAll(/:\s*(-?\d+\.0+)\s*[,}\n]/g)) {
    failures.push(
      `${contractPath}: número inteiro escrito como decimal (${match[1]}); use ${Number(match[1])}`,
    );
  }
}

const profilesDocument = readJson("config/model-profiles.json");
const harness = readJson("config/harness.json");
const registry = readJson("config/tool-registry.json");
const fixturesDocument = readJson("evals/fixtures/regressions.json");
const experimentsDocument = readJson("evals/experiments.json");

if (profilesDocument && harness && registry && fixturesDocument && experimentsDocument) {
  check(profilesDocument.schema_version === 2, "model-profiles deve usar schema_version 2");
  check(harness.schema_version === 2, "harness deve usar schema_version 2");
  check(registry.schema_version === 2, "tool-registry deve usar schema_version 2");
  check(fixturesDocument.schema_version === 2, "fixtures devem usar schema_version 2");
  check(experimentsDocument.schema_version === 2, "experiments deve usar schema_version 2");

  const profiles = profilesDocument.runtime_profiles ?? [];
  const profileIds = unique(profiles.map((profile) => profile.id), "RuntimeProfiles");
  const activeProfile = profiles.find(
    (profile) => profile.id === profilesDocument.active_runtime_profile,
  );
  const functionalProfiles = profiles.filter((profile) => profile.status === "functional");

  const runtimeBackends = new Set(["ollama", "llama_cpp"]);
  check(Boolean(activeProfile), "active_runtime_profile não referencia um RuntimeProfile");
  check(functionalProfiles.length === 1, "deve existir exatamente um RuntimeProfile funcional");
  check(
    functionalProfiles[0]?.id === activeProfile?.id,
    "o único RuntimeProfile funcional deve ser o perfil ativo",
  );
  // Challenger é candidato de bancada: tem artefato instalado e digest medido, mas
  // nenhuma rota de produção o seleciona e ele não entra em release sem promoção.
  for (const profile of profiles) {
    check(
      ["functional", "challenger", "retired"].includes(profile.status),
      `${profile.id}: status deve ser functional, challenger ou retired`,
    );
    check(
      profile.status === "functional" || profile.release_eligible === false,
      `${profile.id}: challenger e retired não são elegíveis para release`,
    );
    check(
      runtimeBackends.has(profile.runtime?.backend),
      `${profile.id}: runtime.backend deve ser ollama ou llama_cpp`,
    );
    for (const field of ["tool_markup_leak_markers", "reasoning_leak_markers"]) {
      if (!Object.hasOwn(profile, field)) continue;
      check(
        Array.isArray(profile[field]) &&
          profile[field].every((marker) => typeof marker === "string" && marker.length > 0),
        `${profile.id}.${field}: deve ser lista de marcadores não vazios`,
      );
    }
    check(
      !Object.hasOwn(profile, "internal_automation_role") ||
        ["tool", "user"].includes(profile.internal_automation_role),
      `${profile.id}: internal_automation_role deve ser tool ou user`,
    );
    const numCtx = Number(profile.installation?.parameters?.num_ctx);
    check(
      Number.isInteger(numCtx) && numCtx > 0,
      `${profile.id}: installation.parameters.num_ctx ausente`,
    );
  }
  check(
    harness.default_runtime_profile === profilesDocument.active_runtime_profile,
    "harness.default_runtime_profile diverge do perfil ativo",
  );

  const shaPattern = /^[a-f0-9]{64}$/;
  const evidenceVariants = profilesDocument.component_evidence_union?.variants ?? {};
  const capabilityContract = profilesDocument.capability_contract ?? {};
  for (const profile of profiles) {
    check(shaPattern.test(profile.profile_digest_sha256 ?? ""), `${profile.id}: digest inválido`);
    check(
      profile.installation?.installed_profile_digest_sha256 === profile.profile_digest_sha256,
      `${profile.id}: instalação e RuntimeProfile usam digests diferentes`,
    );
    if (profile.runtime?.backend === "ollama") {
      check(
        profile.installation?.matches_repository_modelfile === true,
        `${profile.id}: instalação Ollama deve estar coerente com o Modelfile`,
      );
    }
    // O llama-server carrega o arquivo, não uma tag: o digest que prova o modelo é
    // o do GGUF em disco, e é ele que o runtime confere antes do primeiro Turn.
    if (profile.runtime?.backend === "llama_cpp") {
      const weights = (profile.components ?? []).find(
        (component) => component.id === "model_weights",
      );
      check(
        shaPattern.test(profile.installation?.gguf_sha256 ?? "") &&
          weights?.evidence?.sha256 === profile.installation.gguf_sha256,
        `${profile.id}: perfil llama_cpp exige installation.gguf_sha256 igual a model_weights`,
      );
      // A janela e o template são do servidor, não do pedido: -c diferente do
      // num_ctx corta o ModelView, e sem --jinja as tool calls voltam como texto.
      const serverArgs = profile.installation?.server_args ?? [];
      const contextFlag = serverArgs.indexOf("-c");
      check(
        Array.isArray(serverArgs) &&
          contextFlag >= 0 &&
          serverArgs[contextFlag + 1] === String(profile.installation?.parameters?.num_ctx) &&
          serverArgs.includes("--jinja"),
        `${profile.id}: server_args deve fixar -c igual a num_ctx e --jinja`,
      );
      check(
        !["-m", "--model", "--host", "--port", "--alias"].some((flag) => serverArgs.includes(flag)),
        `${profile.id}: server_args não fixa arquivo, endereço nem alias; esses vêm do host`,
      );
    }

    unique((profile.components ?? []).map((component) => component.id), `${profile.id}.components`);
    for (const component of profile.components ?? []) {
      const evidence = component.evidence ?? {};
      const variant = evidenceVariants[evidence.kind];
      check(Boolean(variant), `${profile.id}.${component.id}: variante de evidência inválida`);
      for (const field of variant?.required ?? []) {
        check(
          evidence[field] !== null && evidence[field] !== undefined && evidence[field] !== "",
          `${profile.id}.${component.id}: evidência ${evidence.kind} sem ${field}`,
        );
      }
      if (evidence.kind === "digest_sha256") {
        check(shaPattern.test(evidence.sha256 ?? ""), `${profile.id}.${component.id}: SHA-256 inválido`);
      }
      if (evidence.kind === "embedded_in_profile") {
        check(
          evidence.profile_digest_sha256 === profile.profile_digest_sha256,
          `${profile.id}.${component.id}: componente embutido referencia outro perfil`,
        );
      }
    }
    check(
      (profile.components ?? []).some(
        (component) => component.evidence?.kind === "embedded_in_profile",
      ),
      `${profile.id}: faltam evidências discriminadas de componentes embutidos`,
    );

    // O embedding é outro modelo servido pelo mesmo runtime: ou o perfil o declara
    // inteiro — digest, dimensão e componente próprio — ou não declara nada.
    if (profile.embedding) {
      check(
        shaPattern.test(profile.embedding.digest_sha256 ?? ""),
        `${profile.id}.embedding: digest inválido`,
      );
      check(
        Number.isInteger(profile.embedding.dimensions) && profile.embedding.dimensions > 0,
        `${profile.id}.embedding: dimensões ausentes`,
      );
      const embeddingComponent = (profile.components ?? []).find(
        (component) => component.id === "embedding_model",
      );
      check(
        embeddingComponent?.evidence?.sha256 === profile.embedding.digest_sha256,
        `${profile.id}: componente embedding_model diverge do embedding declarado`,
      );
      check(
        profile.capabilities?.embeddings?.support === "supported",
        `${profile.id}: perfil com embedding deve declarar a capacidade embeddings`,
      );
    }

    for (const [name, capability] of Object.entries(profile.capabilities ?? {})) {
      check(
        sameValues(Object.keys(capability), capabilityContract.required_fields ?? []),
        `${profile.id}.capabilities.${name}: deve conter apenas support/evidence/gate_status`,
      );
      check(
        capabilityContract.support_values?.includes(capability.support),
        `${profile.id}.capabilities.${name}: support inválido`,
      );
      check(
        capabilityContract.evidence_kinds?.includes(capability.evidence?.kind),
        `${profile.id}.capabilities.${name}: evidence inválida`,
      );
      check(
        typeof capability.evidence?.source === "string" && capability.evidence.source.length > 0,
        `${profile.id}.capabilities.${name}: evidence.source ausente`,
      );
      check(
        capabilityContract.gate_status_values?.includes(capability.gate_status),
        `${profile.id}.capabilities.${name}: gate_status inválido`,
      );
      check(
        !(capability.support === "unknown" && capability.gate_status === "passed"),
        `${profile.id}.capabilities.${name}: suporte desconhecido não pode ter gate aprovado`,
      );
    }
  }

  // Todo perfil Ollama nasce de um Modelfile versionado: o do funcional e o de cada
  // Challenger, com as mesmas regras — senão a tag sobe com o num_ctx padrão do
  // Ollama e o braço é medido numa janela que o contrato nunca declarou.
  // Perfil `retired` é registro de medição: o modelo saiu do host e o Modelfile do repo.
  for (const ollamaProfile of profiles.filter(
    (profile) => profile.runtime?.backend === "ollama" && profile.status !== "retired",
  )) {
    const activeProfile = ollamaProfile;
    const modelfilePath = path.join(root, activeProfile.installation.repository_modelfile);
    check(fs.existsSync(modelfilePath), `${activeProfile.id}: Modelfile ausente`);
    if (fs.existsSync(modelfilePath)) {
      const modelfile = fs.readFileSync(modelfilePath, "utf8");
      const modelfileHash = sha256(modelfile);
      check(
        activeProfile.installation.repository_modelfile_sha256 === modelfileHash,
        `${activeProfile.id}: SHA-256 do Modelfile diverge da instalação declarada`,
      );
      const modelfileComponent = activeProfile.components?.find(
        (component) => component.id === "modelfile",
      );
      check(
        modelfileComponent?.evidence?.sha256 === modelfileHash,
        `${activeProfile.id}: componente modelfile diverge do arquivo atual`,
      );

      const directives = modelfile
        .split("\n")
        .map((line) => line.trim())
        .filter((line) => line && !line.startsWith("#"));
      const fromDirectives = directives.filter((line) => /^FROM\s+/i.test(line));
      check(fromDirectives.length === 1, `${activeProfile.id}: Modelfile deve ter exatamente um FROM`);
      check(
        fromDirectives[0] === `FROM ${activeProfile.model.base_model}`,
        `${activeProfile.id}: Modelfile FROM diverge do RuntimeProfile`,
      );
      check(
        !directives.some((line) => /^(TEMPLATE|SYSTEM|MESSAGE)\b/i.test(line)),
        `${activeProfile.id}: Modelfile não deve duplicar TEMPLATE, SYSTEM ou MESSAGE`,
      );

      const actualParameters = Object.fromEntries(
        directives
          .filter((line) => /^PARAMETER\s+/i.test(line))
          .map((line) => {
            const match = /^PARAMETER\s+(\S+)\s+(.+)$/i.exec(line);
            check(Boolean(match), `diretiva PARAMETER inválida: ${line}`);
            return match ? [match[1], match[2]] : [line, null];
          }),
      );
      check(
        canonicalJson(actualParameters) === canonicalJson(activeProfile.installation.parameters),
        `${activeProfile.id}: parâmetros do Modelfile divergem da instalação declarada`,
      );
      for (const requestScoped of ["think", "seed", "num_predict", "stop"]) {
        check(
          !Object.hasOwn(actualParameters, requestScoped),
          `Modelfile não deve fixar parâmetro da ExecutionRoute: ${requestScoped}`,
        );
      }
    }
  }

  const routes = harness.execution_routes ?? [];
  const routeIds = unique(routes.map((route) => route.id), "ExecutionRoutes");
  const defaultRoute = routes.find((route) => route.id === harness.default_execution_route);
  check(Boolean(defaultRoute), "default_execution_route não referencia uma ExecutionRoute");
  for (const route of routes) {
    check(profileIds.has(route.runtime_profile), `${route.id}: RuntimeProfile inexistente`);
    check(!Object.hasOwn(route, "model"), `${route.id}: ExecutionRoute não deve definir model`);
    check(!Object.hasOwn(route, "runtime"), `${route.id}: ExecutionRoute não deve definir runtime`);
  }
  check(defaultRoute?.status === "active", "ExecutionRoute padrão deve estar ativa");
  check(defaultRoute?.surface === "web", "ExecutionRoute padrão deve ser web-first");
  check(defaultRoute?.streaming === false, "streaming deve começar desligado");
  check(defaultRoute?.parallel_tool_execution === false, "paralelismo deve começar desligado");
  check(
    defaultRoute?.reasoning_persistence === "transient_only",
    "reasoning da ExecutionRoute deve ser apenas transitório",
  );
  check(
    activeProfile?.capabilities?.structured_tool_calls?.support === "supported" &&
      activeProfile?.capabilities?.structured_tool_calls?.gate_status === "passed",
    "ExecutionRoute com tools exige structured_tool_calls suportado e aprovado",
  );
  for (const route of routes) {
    const routeProfile = profiles.find((profile) => profile.id === route.runtime_profile);
    check(
      harness.context?.initial_budget_tokens <=
        Number(routeProfile?.installation?.parameters?.num_ctx),
      `${route.id}: context.initial_budget_tokens excede o num_ctx do RuntimeProfile`,
    );
  }
  check(
    !defaultRoute?.sampling?.thinking ||
      (activeProfile?.capabilities?.reasoning?.support === "supported" &&
        activeProfile?.capabilities?.reasoning?.gate_status === "passed"),
    "ExecutionRoute com thinking exige reasoning suportado e aprovado",
  );

  // Visão é um segundo modelo local (ADR 0016): o contrato fixa qual, pelo digest,
  // e os limites que o executor aplica antes de ler um byte da imagem.
  const vision = harness.vision ?? {};
  check(["disabled", "enabled"].includes(vision.mode), "vision.mode deve ser disabled ou enabled");
  check(
    typeof vision.ollama_tag === "string" && vision.ollama_tag.length > 0,
    "vision.ollama_tag ausente",
  );
  check(shaPattern.test(vision.ollama_digest ?? ""), "vision.ollama_digest deve ser SHA-256");
  check(
    Array.isArray(vision.accepted_formats) &&
      vision.accepted_formats.length > 0 &&
      vision.accepted_formats.every((format) => ["png", "jpeg", "webp"].includes(format)),
    "vision.accepted_formats só aceita png, jpeg e webp",
  );
  for (const field of ["max_image_bytes", "max_output_tokens", "context_tokens"]) {
    check(
      Number.isInteger(vision[field]) && vision[field] > 0,
      `vision.${field} deve ser inteiro positivo`,
    );
  }
  check(
    harness.policy?.effect_grants?.local_inference?.length === 0,
    "local_inference não exige grant: nada sai da máquina",
  );

  check(
    harness.platform?.python === "3.13" &&
      harness.platform?.operating_system === "linux" &&
      harness.platform?.architecture === "x86_64",
    "plataforma normativa deve ser Python 3.13/Linux x86_64",
  );
  check(
    harness.loop?.max_steps === 15 &&
      harness.loop?.max_tool_calls_per_step === 4 &&
      harness.loop?.max_tool_calls_per_turn === 20 &&
      harness.loop?.max_turn_duration_seconds === 900 &&
      harness.loop?.max_output_tokens === 8192,
    "limites do loop devem ser 15 steps/4 calls por step/20 por turn/15 min/8192 output",
  );
  check(
    harness.loop?.model_generation_timeout_seconds > 0 &&
      harness.loop?.model_generation_timeout_seconds <= harness.loop?.max_turn_duration_seconds &&
      harness.loop?.max_malformed_model_attempts >= 1,
    "timeout de geração deve caber no orçamento do Turn e o limite de malformadas ser positivo",
  );
  check(
    harness.loop?.stream_tool_calls === false &&
      harness.loop?.parallel_tool_execution === false &&
      harness.loop?.execute_calls_in_emission_order === true &&
      harness.loop?.offer_tools_on_final_step === false,
    "tool loop deve ser completo, serial, ordenado e sem tools no passo final",
  );
  check(
    harness.state?.model_view === "derived_disposable" &&
      harness.state?.ag_ui === "derived_projection" &&
      harness.state?.reasoning?.transient === true &&
      harness.state?.reasoning?.persist_to_canonical_history === false &&
      harness.state?.reasoning?.persist_to_telemetry === false,
    "ModelView/AG-UI devem ser projeções e reasoning não pode ser persistido",
  );
  check(
    harness.context?.code_compression?.enabled === false &&
      harness.context?.code_compression?.status === "disabled_experimental",
    "compressão de código deve permanecer desligada e experimental",
  );
  check(
    harness.workspace?.model_visible_paths === "workspace_relative" &&
      harness.workspace?.reject_absolute_paths === true &&
      harness.workspace?.deny_outside_workspace === true,
    "Workspace deve exigir paths relativos e confinados",
  );

  const expectedPageKey = [
    "workspace_id",
    "relative_path",
    "html_sha256",
    "workspace_revision",
    "verifier_digest",
  ];
  check(
    JSON.stringify(harness.page_verification?.page_revision_key_fields) ===
      JSON.stringify(expectedPageKey),
    "PageRevision deve usar workspace/path/html/workspace revision/verifier digest",
  );
  check(
    harness.page_verification?.kind === "InternalAutomation" &&
      harness.page_verification?.model_selectable === false,
    "verificação de página deve ser InternalAutomation não selecionável",
  );

  check(
    harness.policy?.basis === "effects" && harness.policy?.default === "deny",
    "policy deve ser baseada em efeitos e negar por padrão",
  );
  const grantTypes = new Set(harness.policy?.grant_types ?? []);
  check(
    sameValues(grantTypes, ["WorkspaceRootGrant", "WriteGrant", "WebAccessGrant", "CorpusGrant"]),
    "grant_types deve conter WorkspaceRootGrant, WriteGrant, WebAccessGrant e CorpusGrant",
  );
  check(
    harness.network?.effect === "data_egress" &&
      harness.network?.required_grant === "WebAccessGrant" &&
      harness.network?.default === "deny",
    "rede deve ser data_egress guardado por WebAccessGrant",
  );
  check(
    harness.network?.web_search?.provider === "searxng" &&
      harness.network?.web_search?.fallback === "duckduckgo_lite" &&
      harness.network?.browser?.engine === "chromium",
    "busca deve usar SearXNG com fallback DuckDuckGo e navegador Chromium",
  );
  check(
    harness.network?.browser?.separate_context_per_operation === true &&
      harness.network?.browser?.web_and_page_verification_contexts_separate === true &&
      harness.network?.browser?.share_cookies_storage_cache_or_service_workers === false,
    "contextos de browser devem ser isolados e não compartilhar estado",
  );
  check(
    harness.network?.web_results?.taint === "UntrustedWebTaint" &&
      harness.network?.web_results?.propagate_to_derivations === true &&
      harness.network?.web_results?.taint_never_grants_authority === true,
    "dados web devem propagar UntrustedWebTaint sem conceder autoridade",
  );

  const stores = harness.stores ?? [];
  const storeIds = unique(stores.map((store) => store.id), "stores");
  check(stores.length === 3, "harness deve declarar exatamente três stores");
  check(
    storeIds.has("canonical_state") && storeIds.has("telemetry") && storeIds.has("corpus"),
    "stores devem separar canonical_state, telemetry e corpus",
  );
  check(
    stores.find((store) => store.id === "canonical_state")?.stores_reasoning === false &&
      stores.find((store) => store.id === "telemetry")?.stores_content === false,
    "stores não podem persistir reasoning e telemetria não pode guardar conteúdo",
  );
  const corpusStore = stores.find((store) => store.id === "corpus");
  check(
    corpusStore?.one_file_per_corpus === true &&
      corpusStore?.stores_conversation_state === false &&
      corpusStore?.stores_reasoning === false,
    "store de Corpus deve ser um arquivo por acervo, sem estado de conversa nem reasoning",
  );
  check(
    harness.retention?.policy_scope === "global" &&
      harness.retention?.deletion_unit === "conversation" &&
      harness.retention?.partial_history_deletion === false,
    "retenção deve ser global e remover Conversation inteira",
  );
  check(
    harness.retention?.reaches_corpus_stores === false &&
      harness.corpus?.store?.reached_by_retention === false,
    "retenção não pode alcançar Corpus, que é acervo do Operator e não histórico",
  );

  check(
    harness.corpus?.effect === "corpus_read" &&
      harness.corpus?.required_grant === "CorpusGrant" &&
      harness.corpus?.grant_scope === "corpus_id",
    "Corpus deve ser efeito corpus_read guardado por CorpusGrant com escopo do corpus",
  );
  check(
    harness.corpus?.ingestion?.cleaning === "deterministic_only" &&
      harness.corpus?.ingestion?.model_written_text_is_never_indexed_as_fact === true &&
      harness.corpus?.retrieval?.translate_documents === false,
    "ingestão não pode indexar texto escrito nem traduzido pelo modelo",
  );
  // O piso não pode morar no escore de fusão: rank recíproco ordena e não mede,
  // e o primeiro colocado pontua igual respondendo ou não à pergunta.
  check(
    harness.corpus?.retrieval?.empty_when_nothing_clears_floor === true &&
      harness.corpus?.retrieval?.floor_measured_on === "dense_cosine_similarity" &&
      typeof harness.corpus?.retrieval?.dense_similarity_floor === "number",
    "o piso de relevância deve ser lido na similaridade densa e zerar o resultado",
  );
  check(
    harness.corpus?.retrieval?.injected_passages <= harness.corpus?.retrieval?.dense_candidates,
    "não se pode injetar mais passagens do que a busca produz candidatos",
  );
  check(
    harness.corpus?.scraper?.result_taint === "UntrustedWebTaint" &&
      harness.corpus?.scraper?.requires_grant === false &&
      harness.corpus?.scraper?.authorized_by === "Operator",
    "coleta é ato do Operator e produz UntrustedWebTaint",
  );
  check(
    harness.corpus?.scraper?.html_crawl?.respect_robots_txt === true &&
      harness.corpus?.scraper?.html_crawl?.same_registrable_domain_only === true,
    "crawl HTML deve respeitar robots.txt e não sair do domínio da semente",
  );
  check(
    harness.corpus?.jobs?.resume_on_boot === false,
    "job de ingestão não pode retomar egress sozinho no boot",
  );
  check(
    harness.ui?.delivery === "web_first" &&
      harness.ui?.protocol === "AG-UI" &&
      harness.ui?.projection_source === "canonical_state" &&
      harness.evals?.primary_surface === "web",
    "UX e evals devem ser web-first e AG-UI deve ser projeção",
  );
  check(
    harness.roadmap?.excluded_models?.includes("Qwen2.5-Coder-7B-Instruct"),
    "Qwen2.5-Coder-7B-Instruct deve ficar fora do roadmap",
  );
  check(
    !(harness.outcomes?.terminal_outcomes ?? []).some((value) =>
      (harness.outcomes?.task_verdicts ?? []).includes(value),
    ),
    "TerminalOutcome e TaskVerdict devem ter vocabulários disjuntos",
  );

  check(registry.source_of_truth === true, "tool registry deve declarar source_of_truth");
  check(
    harness.tool_registry === "config/tool-registry.json" &&
      harness.tool_exposure?.source === harness.tool_registry &&
      harness.tool_exposure?.collection === "model_tools",
    "exposição do modelo deve vir somente de tool-registry.model_tools",
  );
  for (const legacyCollection of ["tools", "deferred", "forbidden"]) {
    check(!Object.hasOwn(registry, legacyCollection), `registry ainda contém coleção legada: ${legacyCollection}`);
  }
  check(Array.isArray(registry.model_tools), "registry.model_tools deve ser coleção");
  check(Array.isArray(registry.internal_automations), "registry.internal_automations deve ser coleção");
  check(
    Array.isArray(registry.prohibited_capabilities),
    "registry.prohibited_capabilities deve ser coleção",
  );

  const modelTools = registry.model_tools ?? [];
  const modelToolNames = unique(modelTools.map((tool) => tool.name), "model_tools");
  const automationIds = unique(
    (registry.internal_automations ?? []).map((automation) => automation.id),
    "internal_automations",
  );
  const prohibitedNames = unique(
    (registry.prohibited_capabilities ?? []).map((capability) => capability.name),
    "prohibited_capabilities",
  );
  for (const name of modelToolNames) {
    check(!automationIds.has(name), `${name}: não pode ser model tool e automação`);
    check(!prohibitedNames.has(name), `${name}: não pode ser model tool e capacidade proibida`);
  }

  const namePattern = new RegExp(registry.name_pattern);
  for (const tool of modelTools) {
    check(namePattern.test(tool.name), `${tool.name}: nome inválido`);
    check(tool.status === "enabled", `${tool.name}: model tool deve estar enabled`);
    check(
      tool.parameters?.additionalProperties === false,
      `${tool.name}: parameters.additionalProperties deve ser false`,
    );
    check(Array.isArray(tool.effects) && tool.effects.length > 0, `${tool.name}: effects ausentes`);
    const requiredGrants = new Set(tool.required_grants ?? []);
    for (const effect of tool.effects ?? []) {
      const grantsForEffect = harness.policy?.effect_grants?.[effect];
      check(Boolean(grantsForEffect), `${tool.name}: efeito sem policy: ${effect}`);
      for (const grant of grantsForEffect ?? []) {
        check(requiredGrants.has(grant), `${tool.name}: efeito ${effect} exige ${grant}`);
      }
    }
    for (const grant of requiredGrants) {
      check(grantTypes.has(grant), `${tool.name}: grant desconhecido: ${grant}`);
    }
    if (tool.effects?.includes("data_egress")) {
      check(
        tool.result_taints?.includes("UntrustedWebTaint"),
        `${tool.name}: data_egress deve produzir UntrustedWebTaint`,
      );
    }
    for (const [parameterName, schema] of Object.entries(tool.parameters?.properties ?? {})) {
      if (parameterName.endsWith("_path")) {
        check(
          /relative/i.test(schema.description ?? ""),
          `${tool.name}.${parameterName}: descrição deve exigir path relativo`,
        );
      }
    }
  }

  for (const automation of registry.internal_automations ?? []) {
    check(automation.kind === "InternalAutomation", `${automation.id}: kind inválido`);
    check(automation.model_selectable === false, `${automation.id}: automação não pode ser model-selectable`);
    for (const effect of automation.effects ?? []) {
      check(Boolean(harness.policy?.effect_grants?.[effect]), `${automation.id}: efeito sem policy`);
    }
    for (const grant of automation.required_grants ?? []) {
      check(grantTypes.has(grant), `${automation.id}: grant desconhecido: ${grant}`);
    }
  }
  check(
    automationIds.has(harness.page_verification?.automation_id),
    "page_verification referencia InternalAutomation inexistente",
  );
  check(
    automationIds.has(harness.corpus?.automation_id),
    "corpus.automation_id referencia InternalAutomation inexistente",
  );

  check(
    sameValues(registry.result_envelope?.required ?? [], [
      "status",
      "retryable",
      "data",
      "error",
      "meta",
    ]),
    "ResultPayload deve exigir status/retryable/data/error/meta",
  );
  check(
    registry.result_envelope?.status_issuers?.blocked === "harness_only",
    "status blocked deve pertencer somente ao harness",
  );

  const fixtures = fixturesDocument.fixtures ?? [];
  unique(fixtures.map((fixture) => fixture.id), "fixtures");
  const fixtureTags = new Set(fixtures.flatMap((fixture) => fixture.tags ?? []));
  const terminalOutcomes = new Set(harness.outcomes?.terminal_outcomes ?? []);
  const knownNonCallable = new Set([...automationIds, ...prohibitedNames]);
  for (const fixture of fixtures) {
    check(Boolean(fixture.origin), `${fixture.id}: origin ausente`);
    check(Boolean(fixture.oracle), `${fixture.id}: oracle ausente`);
    for (const calledTool of fixture.oracle?.must_call ?? []) {
      check(modelToolNames.has(calledTool), `${fixture.id}: must_call não é model tool: ${calledTool}`);
    }
    for (const forbiddenCall of fixture.oracle?.must_not_call ?? []) {
      check(
        modelToolNames.has(forbiddenCall) || knownNonCallable.has(forbiddenCall),
        `${fixture.id}: must_not_call referencia nome desconhecido: ${forbiddenCall}`,
      );
    }
    if (fixture.oracle?.terminal_outcome) {
      check(
        terminalOutcomes.has(fixture.oracle.terminal_outcome),
        `${fixture.id}: TerminalOutcome inválido: ${fixture.oracle.terminal_outcome}`,
      );
    }
    check(
      (fixture.oracle?.max_tool_calls ?? 0) <= harness.loop.max_tool_calls_per_turn,
      `${fixture.id}: oracle excede max_tool_calls_per_turn`,
    );
    const payload = fixture.oracle?.result_payload;
    if (payload) {
      check(
        sameValues(Object.keys(payload), registry.result_envelope.required),
        `${fixture.id}: ResultPayload do oracle não contém o envelope completo`,
      );
      check(
        registry.result_envelope.status_values.includes(payload.status),
        `${fixture.id}: ResultPayload.status inválido`,
      );
      check(
        !(payload.data !== null && payload.error !== null),
        `${fixture.id}: data e error não podem ser não nulos juntos`,
      );
      if (payload.status === "blocked") {
        check(payload.meta?.producer === "harness", `${fixture.id}: blocked não foi produzido pelo harness`);
      }
      for (const field of registry.result_envelope.meta_required_fields ?? []) {
        check(Object.hasOwn(payload.meta ?? {}, field), `${fixture.id}: ResultPayload.meta sem ${field}`);
      }
    }
  }

  check(fixturesDocument.primary_surface === "web", "fixtures devem ser web-first");
  const calculatedDatasetDigest = scopedDigest(fixturesDocument, "fixtures");
  check(
    fixturesDocument.dataset_digest_sha256 === calculatedDatasetDigest,
    `fixtures: dataset_digest_sha256 divergente; esperado ${calculatedDatasetDigest}`,
  );
  check(
    experimentsDocument.dataset?.path === "evals/fixtures/regressions.json" &&
      experimentsDocument.dataset?.dataset_id === fixturesDocument.dataset_id &&
      experimentsDocument.dataset?.dataset_version === fixturesDocument.dataset_version &&
      experimentsDocument.dataset?.dataset_digest_sha256 === calculatedDatasetDigest,
    "experiments.dataset diverge do dataset versionado",
  );
  check(experimentsDocument.primary_surface === "web", "experiments devem ser web-first");
  check(
    experimentsDocument.promotion_protocol?.security_gate === "zero_violations",
    "gate de segurança de promoção deve ser zero_violations",
  );
  check(Array.isArray(experimentsDocument.results), "experiments.results deve existir, mesmo vazio");

  // `results` está fora do digest_contract.scope: uma medição registrada não pode
  // mudar de valor ao reselar. Sem selo, a única defesa contra um campo digitado
  // errado é conferir o nome contra freeze_per_run. Regra de superconjunto, para
  // que uma run anterior a um campo novo continue válida com o que de fato teve.
  const frozenFields = new Set(experimentsDocument.promotion_protocol?.freeze_per_run ?? []);
  for (const result of experimentsDocument.results ?? []) {
    for (const [field, value] of Object.entries(result.frozen ?? {})) {
      check(
        frozenFields.has(field),
        `${result.experiment_id}: frozen.${field} não está em freeze_per_run`,
      );
      check(
        !field.endsWith("_digest") || /^[0-9a-f]{64}$/.test(String(value)),
        `${result.experiment_id}: frozen.${field} não é um SHA-256`,
      );
    }
  }

  const experiments = experimentsDocument.experiments ?? [];
  unique(experiments.map((experiment) => experiment.id), "experimentos");
  for (const experiment of experiments) {
    check(
      profileIds.has(experiment.runtime_profile),
      `${experiment.id}: RuntimeProfile inexistente: ${experiment.runtime_profile}`,
    );
    check(
      routeIds.has(experiment.execution_route),
      `${experiment.id}: ExecutionRoute inexistente: ${experiment.execution_route}`,
    );
    const route = routes.find((candidate) => candidate.id === experiment.execution_route);
    check(
      route?.runtime_profile === experiment.runtime_profile,
      `${experiment.id}: ExecutionRoute seleciona outro RuntimeProfile`,
    );
    // Um braço de bake-off troca só o modelo; o perfil que ele pede tem de existir
    // e declarar suporte a tool calls, senão o braço mede a recusa do runtime.
    for (const arm of experiment.arms ?? []) {
      if (!Object.hasOwn(arm, "runtime_profile")) continue;
      const armProfile = profiles.find((profile) => profile.id === arm.runtime_profile);
      check(
        Boolean(armProfile),
        `${experiment.id}.${arm.id}: RuntimeProfile inexistente: ${arm.runtime_profile}`,
      );
      check(
        armProfile?.capabilities?.structured_tool_calls?.support !== "unsupported",
        `${experiment.id}.${arm.id}: perfil sem structured_tool_calls`,
      );
      check(
        !(arm.thinking ?? experiment.fixed?.thinking) ||
          armProfile?.capabilities?.reasoning?.support === "supported",
        `${experiment.id}.${arm.id}: thinking pedido para perfil sem reasoning`,
      );
    }
    for (const tag of experiment.fixture_tags ?? []) {
      check(fixtureTags.has(tag), `${experiment.id}: tag de fixture inexistente: ${tag}`);
    }
  }
  const pageExperiment = experiments.find(
    (experiment) => experiment.id === harness.page_verification.promotion_experiment,
  );
  check(Boolean(pageExperiment), "experimento de promoção da PageRevision não existe");
  check(
    JSON.stringify(pageExperiment?.page_revision_key_fields) === JSON.stringify(expectedPageKey),
    "experimento de PageRevision usa chave divergente",
  );

  const expectedContractDigests = {
    model_profiles_sha256: documentDigest(profilesDocument),
    harness_sha256: documentDigest(harness),
    tool_registry_sha256: documentDigest(registry),
  };
  for (const [field, expected] of Object.entries(expectedContractDigests)) {
    check(
      experimentsDocument.contract_digests?.[field] === expected,
      `experiments.contract_digests.${field} divergente; esperado ${expected}`,
    );
  }
  const calculatedManifestDigest = scopedDigest(experimentsDocument, "experiments");
  check(
    experimentsDocument.manifest_digest_sha256 === calculatedManifestDigest,
    `experiments: manifest_digest_sha256 divergente; esperado ${calculatedManifestDigest}`,
  );
}

const contextPath = path.join(root, "CONTEXT.md");
const contextContent = fs.readFileSync(contextPath, "utf8");
const canonicalTerms = [
  "Operator",
  "WorkspaceRootGrant",
  "Workspace",
  "Conversation",
  "PendingRequest",
  "Turn",
  "AgentStep",
  "ToolResult",
  "ResultPayload",
  "CanonicalHistory",
  "ModelView",
  "RuntimeProfile",
  "Challenger",
  "ExecutionRoute",
  "TerminalOutcome",
  "TaskVerdict",
  "WriteGrant",
  "WebAccessGrant",
  "UntrustedWebTaint",
  "InternalAutomation",
  "PageRevision",
  "Corpus",
  "Document",
  "Chunk",
  "CorpusGrant",
];
const glossaryTerms = [...contextContent.matchAll(/^\*\*([^*]+)\*\*:/gm)].map(
  (match) => match[1],
);
check(
  JSON.stringify(glossaryTerms) === JSON.stringify(canonicalTerms),
  "CONTEXT.md deve conter somente os termos canônicos, na ordem acordada",
);
const contextHeadings = contextContent.match(/^#{1,6}\s+.+$/gm) ?? [];
check(
  sameValues(contextHeadings, ["# Harness 2.0", "## Language"]),
  "CONTEXT.md deve ser apenas glossário no formato domain-modeling",
);

const adrDirectory = path.join(root, "docs/adr");
const adrFiles = fs
  .readdirSync(adrDirectory)
  .filter((name) => /^\d{4}-.*\.md$/.test(name))
  .sort();
const foundationalAdrs = adrFiles.filter((name) => /^000[1-4]-/.test(name));
check(foundationalAdrs.length === 4, "docs/adr deve conter os quatro ADRs fundacionais");
for (const adrFile of adrFiles) {
  const content = fs.readFileSync(path.join(adrDirectory, adrFile), "utf8");
  check(/^# .+/m.test(content), `${adrFile}: título ausente`);
  check(content.trim().split("\n").length <= 8, `${adrFile}: ADR deve permanecer curto`);
}

const releasePending = fs.readFileSync(path.join(root, "docs/RELEASE-PENDING.md"), "utf8");
for (const requiredTopic of [
  "vLLM",
  "Backend remoto",
  "Visão",
  "Shell",
  "Streaming",
  "Paralelismo",
  "Compressão de código",
  "Branching",
  "Approve-with-edits",
  "Dark mode",
  "Qwen2.5-Coder-7B",
  "Limitações residuais",
]) {
  check(releasePending.includes(requiredTopic), `RELEASE-PENDING.md não cobre: ${requiredTopic}`);
}

const markdownFiles = [
  contextPath,
  ...collectMarkdownFiles(path.join(root, "docs")),
  ...collectMarkdownFiles(path.join(root, "evals")),
];
const anchorsByFile = new Map(
  markdownFiles.map((markdownFile) => [
    markdownFile,
    collectGithubHeadingAnchors(fs.readFileSync(markdownFile, "utf8")),
  ]),
);

for (const markdownFile of markdownFiles) {
  const content = fs.readFileSync(markdownFile, "utf8");
  const relativeMarkdownPath = path.relative(root, markdownFile);
  const fenceCount = content.match(/^```/gm)?.length ?? 0;
  check(fenceCount % 2 === 0, `${relativeMarkdownPath}: cercas de código desbalanceadas`);

  for (const match of content.matchAll(/\]\(([^)]+)\)/g)) {
    const href = match[1].replace(/^<|>$/g, "");
    if (/^(https?:|mailto:)/.test(href)) continue;
    const [encodedTarget, encodedFragment] = href.split("#", 2);
    const target = decodeURIComponent(encodedTarget);
    const fragment = encodedFragment ? decodeURIComponent(encodedFragment) : null;
    const resolvedTarget = target
      ? path.resolve(path.dirname(markdownFile), target)
      : markdownFile;
    check(fs.existsSync(resolvedTarget), `${relativeMarkdownPath}: link local quebrado: ${href}`);
    if (fragment && fs.existsSync(resolvedTarget) && resolvedTarget.endsWith(".md")) {
      const targetAnchors =
        anchorsByFile.get(resolvedTarget) ??
        collectGithubHeadingAnchors(fs.readFileSync(resolvedTarget, "utf8"));
      check(
        targetAnchors.has(fragment),
        `${relativeMarkdownPath}: âncora local quebrada: ${href}`,
      );
    }
  }
}

const docsIndex = fs.readFileSync(path.join(root, "docs/README.md"), "utf8");
for (const requiredReference of [
  "CONTEXT.md",
  "DECISOES-2.0.md",
  "RELEASE-PENDING.md",
  "model-profiles.json",
  "harness.json",
  "tool-registry.json",
]) {
  check(docsIndex.includes(requiredReference), `docs/README.md deve apontar para ${requiredReference}`);
}

if (failures.length > 0) {
  for (const failure of failures) console.error(`ERRO: ${failure}`);
  console.error(`\n${failures.length} erro(s).`);
  process.exit(1);
}

console.log("OK: contratos documentais e executáveis coerentes.");
