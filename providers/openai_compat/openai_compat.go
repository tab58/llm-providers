package openai_compat

import (
	"context"
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/tab58/llm-providers/providers/common"
	"github.com/tab58/llm-providers/utils"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
)

const (
	compatMaxRetries    = 5
	compatBaseBackoff   = 2 * time.Second
	compatMaxBackoff    = 60 * time.Second
	compatBackoffJitter = 0.5
)

type Model = common.Model

type Client struct {
	// Name identifies the provider in error messages and logs.
	Name   string
	Client *openai.Client
	// Model is required; it supplies the default max tokens and the
	// context window size.
	Model       Model
	RateLimiter *utils.TokenBucket
	// TokenCostLimit acquires estimated input tokens from the rate limiter
	// instead of one unit per request.
	TokenCostLimit bool
	// RetryRateLimit retries requests that fail with HTTP 429 using
	// exponential backoff.
	RetryRateLimit bool
	// UseMaxCompletionTokens sends max_completion_tokens instead of the
	// deprecated max_tokens, required by newer OpenAI models.
	UseMaxCompletionTokens bool
	// BaseBackoff and maxBackoff override retry backoff timing; zero values
	// fall back to compatBaseBackoff/compatMaxBackoff. Test seam.
	BaseBackoff time.Duration
	MaxBackoff  time.Duration
}

// backoff sleeps before retry attempt+1 using the client's backoff bounds.
func (c *Client) backoff(ctx context.Context, attempt int) error {
	base, maxB := c.BaseBackoff, c.MaxBackoff
	if base == 0 {
		base = compatBaseBackoff
	}
	if maxB == 0 {
		maxB = compatMaxBackoff
	}
	return rateLimitBackoff(ctx, c.Name, attempt, base, maxB)
}

func (c *Client) enforceRateLimit(ctx context.Context, req common.CompletionRequest) error {
	if c.RateLimiter == nil {
		return nil
	}

	cost := int64(1)
	if c.TokenCostLimit {
		tokenCount, err := c.CountTokens(ctx, req)
		if err != nil {
			return fmt.Errorf("%s count tokens: %w", c.Name, err)
		}
		cost = tokenCount.InputTokens
	}
	if err := c.RateLimiter.Acquire(ctx, cost); err != nil {
		return fmt.Errorf("%s rate limiter acquire: %w", c.Name, err)
	}
	return nil
}

func (c *Client) releaseRateLimit() {
	if c.RateLimiter != nil {
		c.RateLimiter.Release()
	}
}

func (c *Client) maxAttempts() int {
	if c.RetryRateLimit {
		return compatMaxRetries
	}
	return 1
}

func (c *Client) SendSyncMessage(ctx context.Context, req common.CompletionRequest) (common.CompletionResponse, error) {
	return c.send(ctx, req)
}

// SendMessageWithTools sends a completion request with the given tools,
// overriding any tools already set on the request.
func (c *Client) SendMessageWithTools(ctx context.Context, req common.CompletionRequest, tools []common.ToolDefinition) (common.CompletionResponse, error) {
	req.Tools = tools
	return c.send(ctx, req)
}

func (c *Client) send(ctx context.Context, req common.CompletionRequest) (common.CompletionResponse, error) {
	params, err := c.buildParams(req)
	if err != nil {
		return common.CompletionResponse{}, err
	}

	if err := c.enforceRateLimit(ctx, req); err != nil {
		return common.CompletionResponse{}, err
	}
	defer c.releaseRateLimit()

	return retryOnRateLimit(ctx, c.Name, c.maxAttempts(), c.backoff, func() (common.CompletionResponse, error) {
		res, err := c.Client.Chat.Completions.New(ctx, params)
		if err != nil {
			return common.CompletionResponse{}, fmt.Errorf("%s send message: %w", c.Name, err)
		}
		return fromOpenAIResponse(res), nil
	})
}

// SendStreamingMessage streams a completion. The events channel is always
// closed before returning, including on error. Rate-limited attempts are
// retried only if no events have been emitted yet, so consumers never see
// duplicated deltas.
func (c *Client) SendStreamingMessage(ctx context.Context, req common.CompletionRequest, events chan<- common.StreamEvent) error {
	defer close(events)

	params, err := c.buildParams(req)
	if err != nil {
		return err
	}
	params.StreamOptions = openai.ChatCompletionStreamOptionsParam{
		IncludeUsage: param.NewOpt(true),
	}

	if err := c.enforceRateLimit(ctx, req); err != nil {
		return err
	}
	defer c.releaseRateLimit()

	attempts := c.maxAttempts()
	for attempt := range attempts {
		emitted, err := c.streamOnce(ctx, params, events)
		if err == nil {
			return nil
		}
		if !emitted && isRateLimitError(err) && attempt < attempts-1 {
			if backoffErr := c.backoff(ctx, attempt); backoffErr != nil {
				return backoffErr
			}
			continue
		}
		events <- common.StreamEvent{Type: common.StreamEventError, Err: err}
		return fmt.Errorf("%s streaming: %w", c.Name, err)
	}
	return nil
}

// streamOnce runs a single streaming attempt. It reports whether any events
// were emitted so the caller knows if a retry is safe.
func (c *Client) streamOnce(ctx context.Context, params openai.ChatCompletionNewParams, events chan<- common.StreamEvent) (bool, error) {
	stream := c.Client.Chat.Completions.NewStreaming(ctx, params)

	type pendingToolCall struct {
		id   string
		name string
		args strings.Builder
	}

	var accumulated common.CompletionResponse
	var text strings.Builder
	toolCalls := map[int64]*pendingToolCall{}
	emitted := false

	for stream.Next() {
		chunk := stream.Current()

		if accumulated.ID == "" && chunk.ID != "" {
			accumulated.ID = chunk.ID
			accumulated.Model = chunk.Model
		}
		if !emitted {
			events <- common.StreamEvent{Type: common.StreamEventStart}
			emitted = true
		}

		if len(chunk.Choices) > 0 {
			choice := chunk.Choices[0]
			if choice.Delta.Content != "" {
				text.WriteString(choice.Delta.Content)
				events <- common.StreamEvent{
					Type: common.StreamEventDelta,
					Text: choice.Delta.Content,
				}
			}
			for _, tc := range choice.Delta.ToolCalls {
				call := toolCalls[tc.Index]
				if call == nil {
					call = &pendingToolCall{}
					toolCalls[tc.Index] = call
				}
				if tc.ID != "" {
					call.id = tc.ID
				}
				if tc.Function.Name != "" {
					call.name = tc.Function.Name
				}
				call.args.WriteString(tc.Function.Arguments)
			}
			if choice.FinishReason != "" {
				accumulated.StopReason = fromOpenAIFinishReason(choice.FinishReason)
			}
		}

		if chunk.Usage.TotalTokens > 0 {
			accumulated.Usage = common.Usage{
				InputTokens:  chunk.Usage.PromptTokens,
				OutputTokens: chunk.Usage.CompletionTokens,
			}
		}
	}

	if err := stream.Err(); err != nil {
		return emitted, err
	}

	if text.Len() > 0 {
		accumulated.Content = append(accumulated.Content, common.NewTextContent(text.String()))
	}

	indexes := make([]int64, 0, len(toolCalls))
	for idx := range toolCalls {
		indexes = append(indexes, idx)
	}
	slices.Sort(indexes)
	for _, idx := range indexes {
		call := toolCalls[idx]
		args := call.args.String()
		if args == "" {
			args = "{}"
		}
		accumulated.Content = append(accumulated.Content, common.NewToolUseContent(call.id, call.name, json.RawMessage(args)))
	}

	events <- common.StreamEvent{
		Type:     common.StreamEventStop,
		Response: &accumulated,
	}
	return emitted, nil
}

func (c *Client) GetCurrentModel() string {
	return c.Model.GetName()
}

func (c *Client) GetContextWindowSize() int {
	return c.Model.GetContextWindowSize()
}

// CountTokens estimates input tokens using the ~4 chars per token rule of
// thumb; OpenAI-compatible APIs have no token counting endpoint.
func (c *Client) CountTokens(_ context.Context, req common.CompletionRequest) (common.TokenCount, error) {
	var totalChars int
	for _, msg := range req.Messages {
		for _, block := range msg.Content {
			totalChars += len(block.Text)
		}
	}
	totalChars += len(req.System)

	return common.TokenCount{InputTokens: int64(totalChars / 4)}, nil
}

func (c *Client) ListModels(ctx context.Context) ([]common.ModelInfo, error) {
	page, err := c.Client.Models.List(ctx)
	if err != nil {
		return nil, fmt.Errorf("%s list models: %w", c.Name, err)
	}

	models := make([]common.ModelInfo, 0, len(page.Data))
	for _, m := range page.Data {
		models = append(models, common.ModelInfo{
			ID:   m.ID,
			Name: m.ID,
		})
	}

	return models, nil
}

func (c *Client) buildParams(req common.CompletionRequest) (openai.ChatCompletionNewParams, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = int64(c.Model.GetMaxTokens())
	}

	msgs := toOpenAIMessages(req.Messages)
	if req.System != "" {
		msgs = append([]openai.ChatCompletionMessageParamUnion{openai.SystemMessage(req.System)}, msgs...)
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(req.Model),
		Messages: msgs,
	}

	if c.UseMaxCompletionTokens {
		params.MaxCompletionTokens = param.NewOpt(maxTokens)
	} else {
		params.MaxTokens = param.NewOpt(maxTokens)
	}

	if req.Temperature != nil {
		params.Temperature = param.NewOpt(*req.Temperature)
	}

	if len(req.Tools) > 0 {
		tools, err := toOpenAITools(req.Tools)
		if err != nil {
			return openai.ChatCompletionNewParams{}, fmt.Errorf("%s: %w", c.Name, err)
		}
		params.Tools = tools
	}

	return params, nil
}
