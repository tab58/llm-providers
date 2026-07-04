package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/tab58/llm-providers/providers/common"
	"github.com/tab58/llm-providers/utils"

	anthropicSDK "github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/shared/constant"
)

// nonStreamingCap is the largest max_tokens the SDK accepts on a
// non-streaming request: its guard rejects requests expected to take longer
// than 10 minutes, scaled at 128000 tokens per hour.
const nonStreamingCap int64 = 128000 / 6

// maxNonStreamingTokens returns the largest max_tokens the SDK permits for a
// non-streaming request to the given model, honoring the SDK's per-model
// limits on top of the 10-minute guard.
func maxNonStreamingTokens(model string) int64 {
	limit := nonStreamingCap
	if modelLimit, ok := constant.ModelNonStreamingTokens[model]; ok {
		limit = min(limit, int64(modelLimit))
	}
	return limit
}

type Model string

const (
	ModelClaudeOpus4_6   = Model(anthropicSDK.ModelClaudeOpus4_6)
	ModelClaudeSonnet4_6 = Model(anthropicSDK.ModelClaudeSonnet4_6)
	ModelClaudeHaiku4_5  = Model(anthropicSDK.ModelClaudeHaiku4_5)

	MaxTokensClaudeOpus4_6   int64 = 128000
	MaxTokensClaudeSonnet4_6 int64 = 64000

	ContextWindowClaudeOpus4_6   = 200_000
	ContextWindowClaudeSonnet4_6 = 200_000
	ContextWindowClaudeHaiku4_5  = 200_000

	ContextWindowDefault = 200_000
)

var anthropicContextWindows = map[Model]int{
	ModelClaudeOpus4_6:   ContextWindowClaudeOpus4_6,
	ModelClaudeSonnet4_6: ContextWindowClaudeSonnet4_6,
	ModelClaudeHaiku4_5:  ContextWindowClaudeHaiku4_5,
}

type Client struct {
	client      *anthropicSDK.Client
	rateLimiter *utils.TokenBucket
	model       Model
}

type Config struct {
	APIKey string
	// BaseURL overrides the API endpoint when set. Used for testing.
	BaseURL string
	Model   Model
}

type configOptions struct {
	rateLimiter *utils.TokenBucket
	haveLimiter bool
}

type Option func(*configOptions)

func WithNoRateLimit() Option {
	return func(o *configOptions) {
		o.rateLimiter = nil
		o.haveLimiter = true
	}
}

func NewClient(cfg Config, opts ...Option) *Client {
	clientOpts := []option.RequestOption{option.WithAPIKey(cfg.APIKey)}
	if cfg.BaseURL != "" {
		clientOpts = append(clientOpts, option.WithBaseURL(cfg.BaseURL))
	}
	client := anthropicSDK.NewClient(clientOpts...)
	model := cfg.Model
	if model == "" {
		model = ModelClaudeSonnet4_6
	}

	var o configOptions
	for _, opt := range opts {
		opt(&o)
	}
	if !o.haveLimiter {
		o.rateLimiter = utils.NewTokenBucket(utils.TokenBucketConfig{
			Rate:           10_000.0 / 60.0, // 10K input tokens per minute
			BurstSize:      10_000,          // 10K possible to pull in one request
			MaxConcurrency: 10,              // max 10 calls concurrent
		})
	}

	return &Client{
		client:      &client,
		rateLimiter: o.rateLimiter,
		model:       model,
	}
}

func (a *Client) enforceRateLimit(ctx context.Context, req common.CompletionRequest) error {
	if a.rateLimiter == nil {
		return nil
	}

	// set up rate limiting based on input token count
	tokenCount, err := a.CountTokens(ctx, req)
	if err != nil {
		return fmt.Errorf("anthropic count tokens: %w", err)
	}
	if err = a.rateLimiter.Acquire(ctx, tokenCount.InputTokens); err != nil {
		return fmt.Errorf("rate limiter acquire failed: %w", err)
	}
	return nil
}

func (a *Client) releaseRateLimit() {
	if a.rateLimiter != nil {
		a.rateLimiter.Release()
	}
}

func (a *Client) SendSyncMessage(ctx context.Context, req common.CompletionRequest) (common.CompletionResponse, error) {
	params, err := a.buildParams(req)
	if err != nil {
		return common.CompletionResponse{}, err
	}
	params.MaxTokens = min(params.MaxTokens, maxNonStreamingTokens(req.Model))

	// enforce rate limit before sending the message
	if err := a.enforceRateLimit(ctx, req); err != nil {
		return common.CompletionResponse{}, fmt.Errorf("anthropic rate limit enforcement failed: %w", err)
	}
	defer a.releaseRateLimit()

	// send the message
	res, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return common.CompletionResponse{}, fmt.Errorf("anthropic send message: %w", err)
	}

	return fromAnthropicResponse(res), nil
}

func (a *Client) SendStreamingMessage(ctx context.Context, req common.CompletionRequest, events chan<- common.StreamEvent) error {
	defer close(events)

	params, err := a.buildParams(req)
	if err != nil {
		return err
	}

	// enforce rate limit before sending the message
	if err := a.enforceRateLimit(ctx, req); err != nil {
		return fmt.Errorf("anthropic rate limit enforcement failed: %w", err)
	}
	defer a.releaseRateLimit()

	stream := a.client.Messages.NewStreaming(ctx, params)

	var accumulated common.CompletionResponse
	// Content blocks under construction, keyed by stream index. Tool input
	// JSON arrives as partial fragments that must be buffered until
	// content_block_stop.
	blocks := map[int64]*common.ContentBlock{}
	jsonParts := map[int64][]string{}

	for stream.Next() {
		event := stream.Current()
		switch event.Type {
		case "message_start":
			accumulated.ID = event.Message.ID
			accumulated.Model = event.Message.Model
			accumulated.Usage.InputTokens = event.Message.Usage.InputTokens
			events <- common.StreamEvent{Type: common.StreamEventStart}

		case "content_block_start":
			switch event.ContentBlock.Type {
			case "text":
				blocks[event.Index] = &common.ContentBlock{Type: common.ContentTypeText}
			case "tool_use":
				blocks[event.Index] = &common.ContentBlock{
					Type:      common.ContentTypeToolUse,
					ToolUseID: event.ContentBlock.ID,
					ToolName:  event.ContentBlock.Name,
				}
			}

		case "content_block_delta":
			switch event.Delta.Type {
			case "text_delta":
				if block := blocks[event.Index]; block != nil {
					block.Text += event.Delta.Text
				}
				events <- common.StreamEvent{
					Type: common.StreamEventDelta,
					Text: event.Delta.Text,
				}
			case "input_json_delta":
				jsonParts[event.Index] = append(jsonParts[event.Index], event.Delta.PartialJSON)
			}

		case "content_block_stop":
			block := blocks[event.Index]
			if block == nil {
				continue
			}
			if block.Type == common.ContentTypeToolUse {
				input := strings.Join(jsonParts[event.Index], "")
				if input == "" {
					input = "{}"
				}
				block.ToolInput = json.RawMessage(input)
			}
			accumulated.Content = append(accumulated.Content, *block)

		case "message_delta":
			accumulated.StopReason = fromAnthropicStopReason(event.Delta.StopReason)
			accumulated.Usage.OutputTokens = event.Usage.OutputTokens
		}
	}

	if err := stream.Err(); err != nil {
		events <- common.StreamEvent{Type: common.StreamEventError, Err: err}
		return fmt.Errorf("anthropic streaming: %w", err)
	}

	// Emitted after the loop rather than on message_stop so consumers always
	// get a stop event when the stream ends cleanly.
	events <- common.StreamEvent{
		Type:     common.StreamEventStop,
		Response: &accumulated,
	}

	return nil
}

// SendMessageWithTools sends a completion request with the given tools,
// overriding any tools already set on the request.
func (a *Client) SendMessageWithTools(ctx context.Context, req common.CompletionRequest, tools []common.ToolDefinition) (common.CompletionResponse, error) {
	req.Tools = tools
	params, err := a.buildParams(req)
	if err != nil {
		return common.CompletionResponse{}, err
	}
	params.MaxTokens = min(params.MaxTokens, maxNonStreamingTokens(req.Model))

	// enforce rate limit before sending the message
	if err := a.enforceRateLimit(ctx, req); err != nil {
		return common.CompletionResponse{}, fmt.Errorf("anthropic rate limit enforcement failed: %w", err)
	}
	defer a.releaseRateLimit()

	res, err := a.client.Messages.New(ctx, params)
	if err != nil {
		return common.CompletionResponse{}, fmt.Errorf("anthropic send message with tools: %w", err)
	}

	return fromAnthropicResponse(res), nil
}

func (a *Client) GetCurrentModel() string {
	return string(a.model)
}

func (a *Client) GetContextWindowSize() int {
	if w, ok := anthropicContextWindows[a.model]; ok {
		return w
	}
	return ContextWindowDefault
}

func (a *Client) CountTokens(ctx context.Context, req common.CompletionRequest) (common.TokenCount, error) {
	params := anthropicSDK.MessageCountTokensParams{
		Model:    anthropicSDK.Model(req.Model),
		Messages: toAnthropicMessages(req.Messages),
	}

	if req.System != "" {
		params.System = anthropicSDK.MessageCountTokensParamsSystemUnion{
			OfTextBlockArray: []anthropicSDK.TextBlockParam{
				{Text: req.System},
			},
		}
	}

	res, err := a.client.Messages.CountTokens(ctx, params)
	if err != nil {
		return common.TokenCount{}, fmt.Errorf("anthropic count tokens: %w", err)
	}

	return common.TokenCount{InputTokens: res.InputTokens}, nil
}

func (a *Client) ListModels(ctx context.Context) ([]common.ModelInfo, error) {
	page, err := a.client.Models.List(ctx, anthropicSDK.ModelListParams{})
	if err != nil {
		return nil, fmt.Errorf("anthropic list models: %w", err)
	}

	models := make([]common.ModelInfo, 0, len(page.Data))
	for _, m := range page.Data {
		models = append(models, common.ModelInfo{
			ID:        m.ID,
			Name:      m.DisplayName,
			MaxTokens: m.MaxTokens,
		})
	}

	return models, nil
}

func (a *Client) buildParams(req common.CompletionRequest) (anthropicSDK.MessageNewParams, error) {
	maxTokens := req.MaxTokens
	if maxTokens == 0 {
		maxTokens = common.MaxTokensStdResponse
	}

	params := anthropicSDK.MessageNewParams{
		Model:     anthropicSDK.Model(req.Model),
		MaxTokens: maxTokens,
		Messages:  toAnthropicMessages(req.Messages),
	}

	if req.System != "" {
		params.System = []anthropicSDK.TextBlockParam{
			{Text: req.System},
		}
	}

	if req.Temperature != nil {
		params.Temperature = anthropicSDK.Float(*req.Temperature)
	}

	if len(req.Tools) > 0 {
		tools, err := toAnthropicTools(req.Tools)
		if err != nil {
			return anthropicSDK.MessageNewParams{}, err
		}
		params.Tools = tools
	}

	return params, nil
}

func toAnthropicMessages(msgs []common.Message) []anthropicSDK.MessageParam {
	result := make([]anthropicSDK.MessageParam, 0, len(msgs))
	for _, msg := range msgs {
		blocks := toAnthropicContentBlocks(msg.Content)
		switch msg.Role {
		case common.RoleUser:
			result = append(result, anthropicSDK.NewUserMessage(blocks...))
		case common.RoleAssistant:
			result = append(result, anthropicSDK.NewAssistantMessage(blocks...))
		case common.RoleTool:
			// Anthropic has no tool role; tool_result blocks ride in a
			// user message.
			result = append(result, anthropicSDK.NewUserMessage(blocks...))
		case common.RoleSystem:
			continue
		}
	}
	return result
}

func toAnthropicContentBlocks(blocks []common.ContentBlock) []anthropicSDK.ContentBlockParamUnion {
	result := make([]anthropicSDK.ContentBlockParamUnion, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case common.ContentTypeText:
			result = append(result, anthropicSDK.NewTextBlock(block.Text))
		case common.ContentTypeToolUse:
			result = append(result, anthropicSDK.NewToolUseBlock(block.ToolUseID, block.ToolInput, block.ToolName))
		case common.ContentTypeToolResult:
			result = append(result, anthropicSDK.NewToolResultBlock(block.ToolResultID, block.ToolOutput, false))
		}
	}
	return result
}

func toAnthropicTools(tools []common.ToolDefinition) ([]anthropicSDK.ToolUnionParam, error) {
	result := make([]anthropicSDK.ToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		// InputSchema is a full JSON Schema object; Anthropic's ToolInputSchemaParam
		// wants its properties and required fields split out.
		var schema struct {
			Properties json.RawMessage `json:"properties"`
			Required   []string        `json:"required"`
		}
		if tool.InputSchema != nil {
			if err := json.Unmarshal(tool.InputSchema, &schema); err != nil {
				return nil, fmt.Errorf("anthropic tool %q: parse input schema: %w", tool.Name, err)
			}
		}

		var props any
		if schema.Properties != nil {
			if err := json.Unmarshal(schema.Properties, &props); err != nil {
				return nil, fmt.Errorf("anthropic tool %q: parse schema properties: %w", tool.Name, err)
			}
		}

		result = append(result, anthropicSDK.ToolUnionParam{
			OfTool: &anthropicSDK.ToolParam{
				Name:        tool.Name,
				Description: anthropicSDK.String(tool.Description),
				InputSchema: anthropicSDK.ToolInputSchemaParam{
					Properties: props,
					Required:   schema.Required,
				},
			},
		})
	}
	return result, nil
}

func fromAnthropicResponse(res *anthropicSDK.Message) common.CompletionResponse {
	content := make([]common.ContentBlock, 0, len(res.Content))
	for _, block := range res.Content {
		switch block.Type {
		case "text":
			content = append(content, common.NewTextContent(block.Text))
		case "tool_use":
			content = append(content, common.NewToolUseContent(block.ID, block.Name, block.Input))
		}
	}

	return common.CompletionResponse{
		ID:         res.ID,
		Content:    content,
		StopReason: fromAnthropicStopReason(res.StopReason),
		Usage: common.Usage{
			InputTokens:  res.Usage.InputTokens,
			OutputTokens: res.Usage.OutputTokens,
		},
		Model: res.Model,
	}
}

func fromAnthropicStopReason(reason anthropicSDK.StopReason) common.StopReason {
	switch reason {
	case anthropicSDK.StopReasonEndTurn:
		return common.StopReasonEndTurn
	case anthropicSDK.StopReasonMaxTokens:
		return common.StopReasonMaxTokens
	case anthropicSDK.StopReasonToolUse:
		return common.StopReasonToolUse
	default:
		return common.StopReason(reason)
	}
}
