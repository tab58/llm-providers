# AGENTS.md

Root documentation for this repository. Source of truth for layout, behavior, and conventions (see `CLAUDE.md` for the doc-maintenance rules). Update this file in the same change whenever code alters anything described here.

## What this library is

A unified Go interface over multiple LLM providers. Callers program against one interface (`common.LLM`) and one set of request/response types; each provider package adapts those to its native API. Module path: `github.com/tab58/llm-providers`.

## Design goals

1. **One interface, swappable backends.** Provider choice is a construction-time detail. All clients satisfy `common.LLM`; `providers.go` at the root enforces this with compile-time checks (`var _ common.LLM = (*anthropic.Client)(nil)` etc.). Adding a provider means adding a check there.
2. **Provider-agnostic types at the boundary.** `common.CompletionRequest`, `CompletionResponse`, `Message`, `ContentBlock`, `StreamEvent`, and `ToolDefinition` normalize Anthropic's and OpenAI's differing shapes (content blocks vs. tool call objects, `end_turn` vs. `stop`, system-as-parameter vs. system-as-message). SDK types never leak out of a provider package.
3. **Share the OpenAI-compatible implementation.** Cerebras, OpenRouter, Lightning, and OpenAI itself differ only in base URL, defaults, and flags. They all delegate to one embedded `openai_compat.Client` rather than four copies of the same logic.
4. **Client-side rate limiting by default, in one place.** `ratelimit` owns all limiting: `ratelimit.Wrap` decorates any `common.LLM` with a `Limiter` and a cost strategy. Each provider's `NewClient` returns the wrapped client with a provider-appropriate default unless the caller opts out (`WithNoRateLimit()`), so naive callers don't blow through provider quotas. Provider clients themselves contain no limiter code.
5. **Graceful degradation over hard failure.** Operations a provider can't do return `common.ErrNotSupported` (e.g. Ollama `CountTokens`) or an estimate (OpenAI-compat token counting) rather than panicking or silently lying.
6. **No API keys needed for tests.** Tests run against local mock HTTP servers. Endpoint override per provider: `Config.BaseURL` where the value is deployment/host-specific (lightning, ollama), a `WithBaseURL` option for proxy/gateway cases (openai, openrouter), or none at all where the endpoint is fixed (anthropic, cerebras — tests construct the client directly in-package). All default to a pinned `*BaseURL` const. `go test ./...` must pass offline.

## Architecture

```
providers.go                 compile-time common.LLM compliance checks
common/                      the contract: LLM interface + shared types
  llm.go                     LLM interface, ErrNotSupported, defaults
  chat.go                    Message, ContentBlock, CompletionRequest/Response,
                             StreamEvent, ToolDefinition, StopReason, Usage
  convert.go                 small shared helpers (CombinedText)
anthropic/                   native Anthropic SDK adapter
openai_compat/               shared impl for all OpenAI-compatible APIs
  openai_compat.go           Client struct + LLM methods
  openai_compat_convert.go   common <-> OpenAI SDK type conversion
  openai_compat_retry.go     429 detection + exponential backoff w/ jitter
openai/                      thin config wrapper over openai_compat
cerebras/                    thin config wrapper over openai_compat
openrouter/                  thin config wrapper over openai_compat
lightning/                   thin config wrapper over openai_compat
ollama/                      native Ollama API adapter (local or cloud)
ratelimit/                   all client-side rate limiting: Limiter interface,
                             Wrap decorator, cost strategies, TokenBucket, Semaphore
errors/                      errors.Wrap helper
logger/                      minimal Logger interface (nil = silent)
testutils/                   CollectEvents helper for stream tests
```

### The contract (`common`)

`LLM` has seven methods: `SendSyncMessage`, `SendStreamingMessage`, `SendMessageWithTools`, `CountTokens`, `ListModels`, `GetCurrentModel`, `GetContextWindowSize`. `ContentBlock` is a tagged union discriminated by `Type` (`text`, `tool_use`, `tool_result`); constructors (`NewTextContent`, `NewToolUseContent`, `NewToolResultContent`, `NewUserMessage`, …) are the intended way to build them.

### Streaming protocol

`SendStreamingMessage` takes a caller-supplied `chan<- common.StreamEvent` and **always closes it** when the stream ends (success or error). Event order: `start` → `delta`/`thinking`(s) → `stop`, with `error` possible at any point. The `stop` event carries the full accumulated `*CompletionResponse`. Errors are delivered both as an `error` event and as the method's return value.

### openai_compat.Client

Exported-field struct (not functional options) because only the thin provider wrappers construct it. It never rate-limits itself — the thin wrappers apply `ratelimit.Wrap` around it. Behavior flags:

- `RetryRateLimit` — retry HTTP 429 up to 5 attempts, exponential backoff (2s base, 60s cap, ±50% jitter). `BaseBackoff`/`MaxBackoff` are test seams. This stays in the client (transport-level response handling), unlike client-side throttling.
- `UseMaxCompletionTokens` — send `max_completion_tokens` (newer OpenAI models) instead of deprecated `max_tokens`.

`CountTokens` here is an estimate (no OpenAI counting endpoint); Anthropic uses its real counting API.

### Anthropic specifics

Non-streaming requests are capped by the SDK's 10-minute guard (`nonStreamingCap` = 128000/6 tokens) plus per-model limits — `maxNonStreamingTokens` clamps `MaxTokens` accordingly. Default model: `ModelClaudeSonnet4_6`. Default rate limit: 10K input tokens/min, burst 10K, 10 concurrent.

### Rate limiting (`ratelimit`)

The single home for client-side rate limiting. `Wrap(llm, limiter, costFn)` returns a `common.LLM` whose three send methods do cost → `Acquire` → delegate → `Release`; `CountTokens`, `ListModels`, and the getters pass through unlimited. A nil `Limiter` returns the inner client unchanged. If `Acquire` fails before a streaming call, the wrapper closes the events channel itself to honor the streaming contract.

- `Limiter` interface: `Acquire(ctx, cost)` / `Release()`. Implementations: `TokenBucket` (token rate + concurrency semaphore) and `Semaphore` (concurrency only; ignores cost). Callers may supply custom implementations.
- Cost strategies: `CostPerRequest` (1 per call, the default) and `CostByTokenCount` (estimated input tokens via the inner client's `CountTokens`; falls back to 1 on `ErrNotSupported`).
- Defaults per provider: anthropic and openai wrap with a 10K tokens/min bucket + `CostByTokenCount`; cerebras (30 req/min) and lightning (15 req/min) use `CostPerRequest`; ollama uses `Semaphore(3)`; openrouter ships unwrapped.

## Conventions

- **Errors:** wrap with context and the provider name: `fmt.Errorf("ollama count tokens: %w", err)`. Sentinel: `common.ErrNotSupported` (check with `errors.Is`).
- **Construction:** exported `Config` struct + `NewClient(cfg, opts...)`; every provider accepts functional options (`WithNoRateLimit` where a default limiter exists, `WithBaseURL` on openai/openrouter, `WithLocalURL` on ollama). Empty `Config` fields get sensible defaults (model, base URL). `NewClient` returns `common.LLM` — the client wrapped with its default limiter — while `WithNoRateLimit()` returns the raw `*Client` (tests needing the concrete type construct with it and type-assert).
- **Tests:** table-driven, run against `httptest` mock servers — via the `Config.BaseURL` override (lightning, ollama), the `WithBaseURL` option (openai, openrouter), or direct in-package client construction (anthropic, cerebras) — never against live APIs. Use `testutils.CollectEvents` to drain and assert on stream events. Run `go test ./...` and `go test -race ./...` before calling work done.
- **Type conversion:** each provider owns its `common` ↔ SDK conversion code, kept in a separate `*_convert.go` file where sizable.
- **Adding a provider:** implement `common.LLM` (or wrap `openai_compat.Client` if the API is OpenAI-compatible), return it from `NewClient` via `ratelimit.Wrap` with a provider-appropriate default limiter, add the compile-time check in `providers.go`, add mock-server tests, and update the provider table in `README.md` and this file.
