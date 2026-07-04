# llm-providers

Unified Go interface for multiple LLM providers. One request/response type, one `LLM` interface, six backends: Anthropic, OpenAI, Cerebras, OpenRouter, Ollama, and Lightning AI.

## Features

- **Single interface** — every provider implements `common.LLM`: sync completions, streaming, tool use, token counting, and model listing.
- **Provider-agnostic types** — `CompletionRequest`, `CompletionResponse`, `Message`, and `ContentBlock` normalize the differences between Anthropic's and OpenAI's APIs.
- **Built-in rate limiting** — token-bucket limiter on by default (configurable or disabled per client).
- **Streaming** — channel-based event stream with `start`/`delta`/`thinking`/`stop`/`error` events.
- **Tool use** — one `ToolDefinition` type mapped to each provider's native tool/function calling.

## Installation

```bash
go get github.com/tab58/llm-providers
```

Requires Go 1.25+.

## Usage

### Synchronous completion

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/tab58/llm-providers/anthropic"
	"github.com/tab58/llm-providers/common"
)

func main() {
	client := anthropic.NewClient(anthropic.Config{
		APIKey: os.Getenv("ANTHROPIC_API_KEY"),
		Model:  anthropic.ModelClaudeSonnet4_6,
	})

	resp, err := client.SendSyncMessage(context.Background(), common.CompletionRequest{
		System: "You are a concise assistant.",
		Messages: []common.Message{
			common.NewUserMessage("What is a monad in one sentence?"),
		},
		MaxTokens: 1024,
	})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(resp.Text())
	fmt.Printf("tokens: %d in / %d out\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}
```

### Streaming

```go
events := make(chan common.StreamEvent)
go func() {
	if err := client.SendStreamingMessage(ctx, req, events); err != nil {
		log.Fatal(err)
	}
}()

for ev := range events {
	switch ev.Type {
	case common.StreamEventDelta:
		fmt.Print(ev.Text)
	case common.StreamEventStop:
		fmt.Printf("\nstop reason: %s\n", ev.Response.StopReason)
	case common.StreamEventError:
		log.Fatal(ev.Err)
	}
}
```

### Tool use

```go
tools := []common.ToolDefinition{{
	Name:        "get_weather",
	Description: "Get the current weather for a city",
	InputSchema: json.RawMessage(`{
		"type": "object",
		"properties": {"city": {"type": "string"}},
		"required": ["city"]
	}`),
}}

resp, err := client.SendMessageWithTools(ctx, common.CompletionRequest{
	Messages:  []common.Message{common.NewUserMessage("Weather in Tokyo?")},
	MaxTokens: 1024,
}, tools)
if err != nil {
	log.Fatal(err)
}

for _, call := range resp.ToolCalls() {
	fmt.Printf("model wants %s with input %s\n", call.ToolName, call.ToolInput)

	// Run the tool, then send the result back in a follow-up message:
	result := common.NewToolResultContent(call.ToolUseID, call.ToolName, `{"temp_c": 21}`)
	_ = common.Message{Role: common.RoleTool, Content: []common.ContentBlock{result}}
}
```

### Swapping providers

Every client satisfies `common.LLM`, so provider choice is a construction detail:

```go
import (
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/ollama"
	"github.com/tab58/llm-providers/openai"
)

var llm common.LLM

switch os.Getenv("LLM_PROVIDER") {
case "openai":
	llm = openai.NewClient(openai.Config{APIKey: os.Getenv("OPENAI_API_KEY")})
case "ollama":
	llm = ollama.NewClient(ollama.Config{BaseURL: "http://localhost:11434"})
default:
	llm = anthropic.NewClient(anthropic.Config{APIKey: os.Getenv("ANTHROPIC_API_KEY")})
}
```

## Providers

| Provider   | Package                | Notes                                                        |
| ---------- | ---------------------- | ------------------------------------------------------------ |
| Anthropic  | `anthropic`  | Native SDK; token counting via dedicated API endpoint         |
| OpenAI     | `openai`     | Chat completions API; token counts estimated                  |
| Cerebras   | `cerebras`   | OpenAI-compatible endpoint                                    |
| OpenRouter | `openrouter` | OpenAI-compatible endpoint                                    |
| Ollama     | `ollama`     | Local or cloud; configurable `ContextSize` and logger         |
| Lightning  | `lightning`  | OpenAI-compatible endpoint; custom `BaseURL`                  |

The OpenAI-compatible providers share one implementation (`openai_compat`) and differ only in configuration.

## Rate limiting

All client-side rate limiting lives in one place: `ratelimit`. Each provider's `NewClient` returns a `common.LLM` already wrapped with its default limiter (e.g. Anthropic: 10K input tokens/min, max 10 concurrent calls). Disable it with the provider's option:

```go
client := anthropic.NewClient(cfg, anthropic.WithNoRateLimit())
```

To apply a custom limit, disable the default and wrap yourself:

```go
raw := anthropic.NewClient(cfg, anthropic.WithNoRateLimit())
llm := ratelimit.Wrap(raw, ratelimit.NewTokenBucket(ratelimit.TokenBucketConfig{
	Rate:           1000.0 / 60.0,
	BurstSize:      1000,
	MaxConcurrency: 5,
}), ratelimit.CostByTokenCount)
```

`ratelimit.Wrap` guards the three send methods of any `common.LLM`; `CountTokens`, `ListModels`, and the getters pass through unlimited. Limiters: `TokenBucket` (token rate + concurrency) and `Semaphore` (concurrency only), or any custom `ratelimit.Limiter`. Cost strategies: `CostPerRequest` (one unit per call) and `CostByTokenCount` (estimated input tokens via the client's `CountTokens`).

## The `LLM` interface

```go
type LLM interface {
	SendSyncMessage(ctx context.Context, req CompletionRequest) (CompletionResponse, error)
	SendStreamingMessage(ctx context.Context, req CompletionRequest, events chan<- StreamEvent) error
	SendMessageWithTools(ctx context.Context, req CompletionRequest, tools []ToolDefinition) (CompletionResponse, error)
	CountTokens(ctx context.Context, req CompletionRequest) (TokenCount, error)
	ListModels(ctx context.Context) ([]ModelInfo, error)
	GetCurrentModel() string
	GetContextWindowSize() int
}
```

Providers that don't support an operation return `common.ErrNotSupported`.

## Testing

```bash
go test ./...
go test -race ./...
```

Tests run against mock HTTP servers — no API keys required.

## License

No license file yet — all rights reserved by default.
