package ollama

import (
	"bytes"
	"context"
	"encoding/json"

	"fmt"
	"io"
	"net/http"
	"regexp"
	"strings"

	"github.com/tab58/llm-providers/errors"
	"github.com/tab58/llm-providers/logger"
	"github.com/tab58/llm-providers/providers/common"
	"github.com/tab58/llm-providers/utils"
)

var thinkBlockRe = regexp.MustCompile(`(?s)<think>.*?</think>\s*`)

const OLLAMA_CLOUD_BASE_URL = "https://ollama.com/"

const MAX_CONCURRENT_REQUESTS = 3

// Client implements the LLM interface using Client's native /api/* endpoints.
type Client struct {
	baseURL     string
	apiKey      string
	client      *http.Client
	rateLimiter *utils.Semaphore
	model       Model
	log         logger.Logger
}

// Config holds configuration for connecting to an Ollama server.
type Config struct {
	BaseURL     string
	APIKey      string
	Model       Model
	ContextSize int64 // Ollama num_ctx: total context window (input+output). 0 uses model default.
	// Logger receives request/response diagnostics. Nil disables them.
	Logger logger.Logger
}

// logf writes a diagnostic line when a Logger is configured.
func (o *Client) logf(format string, args ...any) {
	if o.log != nil {
		o.log.Logf(format, args...)
	}
}

// NewClient creates an Ollama LLM client using the native Ollama API.
func NewClient(cfg Config) *Client {
	model := cfg.Model
	if model == nil {
		model = Model_Gemma4_31B
	}

	url := cfg.BaseURL
	if url == "" {
		url = OLLAMA_CLOUD_BASE_URL
	}

	return &Client{
		baseURL:     strings.TrimSuffix(url, "/"),
		apiKey:      cfg.APIKey,
		client:      http.DefaultClient,
		rateLimiter: utils.NewSemaphore(MAX_CONCURRENT_REQUESTS),
		model:       model,
		log:         cfg.Logger,
	}
}

// -- internal request/response types --

type ollamaChatRequest struct {
	Model    string              `json:"model"`
	Messages []ollamaChatMessage `json:"messages"`
	Stream   bool                `json:"stream"`
	Think    *bool               `json:"think,omitempty"`
	Tools    []ollamaTool        `json:"tools,omitempty"`
	Options  map[string]any      `json:"options,omitempty"`
}

type ollamaChatResponse struct {
	Model           string            `json:"model"`
	Message         ollamaChatMessage `json:"message"`
	Done            bool              `json:"done"`
	DoneReason      string            `json:"done_reason"`
	PromptEvalCount int               `json:"prompt_eval_count"`
	EvalCount       int               `json:"eval_count"`
}

type ollamaChatMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	// ToolName labels a role-"tool" message with the tool that produced it.
	// Ollama's native API has no tool_call_id; this is its only linkage.
	ToolName string `json:"tool_name,omitempty"`
}

type ollamaTool struct {
	Type     string             `json:"type"`
	Function ollamaToolFunction `json:"function"`
}

type ollamaToolFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

type ollamaToolCall struct {
	Function ollamaToolCallFunction `json:"function"`
}

type ollamaToolCallFunction struct {
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

type ollamaModelsResponse struct {
	Models []ollamaModelEntry `json:"models"`
}

type ollamaModelEntry struct {
	Name  string `json:"name"`
	Model string `json:"model"`
}

// -- LLM interface implementation --

func (o *Client) SendSyncMessage(ctx context.Context, req common.CompletionRequest) (common.CompletionResponse, error) {
	tools, err := toOllamaTools(req.Tools, o.log)
	if err != nil {
		return common.CompletionResponse{}, err
	}

	chatReq := ollamaChatRequest{
		Model:    string(req.Model),
		Messages: toOllamaMessages(req, o.log),
		Stream:   false,
		Think:    boolPtr(false),
		Tools:    tools,
		Options:  o.ollamaOptions(req),
	}

	if err := o.rateLimiter.Acquire(ctx); err != nil {
		return common.CompletionResponse{}, errors.Wrap(err, "unable to acquire semaphore")
	}
	defer o.rateLimiter.Release()

	var chatRes ollamaChatResponse
	if err := o.postJSON(ctx, "/api/chat", chatReq, &chatRes); err != nil {
		return common.CompletionResponse{}, errors.Wrap(err, "ollama send message")
	}

	return fromOllamaResponse(chatRes, o.log), nil
}

func (o *Client) SendStreamingMessage(ctx context.Context, req common.CompletionRequest, events chan<- common.StreamEvent) error {
	defer close(events)

	tools, err := toOllamaTools(req.Tools, o.log)
	if err != nil {
		return err
	}

	if err := o.rateLimiter.Acquire(ctx); err != nil {
		return errors.Wrap(err, "unable to acquire semaphore")
	}
	defer o.rateLimiter.Release()

	chatReq := ollamaChatRequest{
		Model:    string(req.Model),
		Messages: toOllamaMessages(req, o.log),
		Stream:   true,
		Think:    boolPtr(false),
		Tools:    tools,
		Options:  o.ollamaOptions(req),
	}

	body, err := json.Marshal(chatReq)
	if err != nil {
		return fmt.Errorf("ollama marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+"/api/chat", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("ollama create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+o.apiKey)
	}

	o.logf("ollama: POST %s/api/chat stream=true model=%s messages=%d tools=%d",
		o.baseURL, chatReq.Model, len(chatReq.Messages), len(chatReq.Tools))

	resp, err := o.client.Do(httpReq)
	if err != nil {
		o.logf("ollama: request error: %v", err)
		return fmt.Errorf("ollama streaming request: %w", err)
	}
	defer resp.Body.Close()

	o.logf("ollama: response status=%d", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("ollama streaming: unexpected status %d: %s", resp.StatusCode, readErrorBody(resp.Body))
	}

	var started bool
	var accumulatedToolCalls []ollamaToolCall
	thinkParser := newThinkBlockParser(events)
	decoder := json.NewDecoder(resp.Body)
	for decoder.More() {
		var chunk ollamaChatResponse
		if err := decoder.Decode(&chunk); err != nil {
			events <- common.StreamEvent{Type: common.StreamEventError, Err: err}
			return fmt.Errorf("ollama streaming decode: %w", err)
		}

		if !started {
			events <- common.StreamEvent{Type: common.StreamEventStart}
			started = true
		}

		if chunk.Message.Content != "" {
			thinkParser.Feed(chunk.Message.Content)
		}

		if len(chunk.Message.ToolCalls) > 0 {
			o.logf("ollama: chunk has %d tool_calls", len(chunk.Message.ToolCalls))
			accumulatedToolCalls = append(accumulatedToolCalls, chunk.Message.ToolCalls...)
		}

		if chunk.Done {
			// accumulatedToolCalls already includes this chunk's tool calls,
			// so using it never drops calls seen in earlier chunks.
			if len(accumulatedToolCalls) > 0 {
				chunk.Message.ToolCalls = accumulatedToolCalls
			}
			o.logf("ollama: done chunk content=%q tool_calls=%d done_reason=%s eval_count=%d",
				chunk.Message.Content, len(chunk.Message.ToolCalls), chunk.DoneReason, chunk.EvalCount)
			res := fromOllamaResponse(chunk, o.log)
			thinkParser.Flush()
			events <- common.StreamEvent{
				Type:     common.StreamEventStop,
				Response: &res,
			}
		}
	}

	thinkParser.Flush()
	return nil
}

// SendMessageWithTools sends a completion request with the given tools,
// overriding any tools already set on the request.
func (o *Client) SendMessageWithTools(ctx context.Context, req common.CompletionRequest, tools []common.ToolDefinition) (common.CompletionResponse, error) {
	ollamaTools, err := toOllamaTools(tools, o.log)
	if err != nil {
		return common.CompletionResponse{}, err
	}

	if err := o.rateLimiter.Acquire(ctx); err != nil {
		return common.CompletionResponse{}, errors.Wrap(err, "unable to acquire semaphore")
	}
	defer o.rateLimiter.Release()

	chatReq := ollamaChatRequest{
		Model:    string(req.Model),
		Messages: toOllamaMessages(req, o.log),
		Stream:   false,
		Think:    boolPtr(false),
		Tools:    ollamaTools,
		Options:  o.ollamaOptions(req),
	}

	o.logf("ollama: POST %s/api/chat stream=false model=%s messages=%d tools=%d",
		o.baseURL, chatReq.Model, len(chatReq.Messages), len(chatReq.Tools))

	var chatRes ollamaChatResponse
	if err := o.postJSON(ctx, "/api/chat", chatReq, &chatRes); err != nil {
		o.logf("ollama: SendMessageWithTools error: %v", err)
		return common.CompletionResponse{}, fmt.Errorf("ollama send message with tools: %w", err)
	}

	o.logf("ollama: SendMessageWithTools done_reason=%s tool_calls=%d",
		chatRes.DoneReason, len(chatRes.Message.ToolCalls))

	return fromOllamaResponse(chatRes, o.log), nil
}

func (o *Client) GetCurrentModel() string {
	return o.model.GetName()
}

func (o *Client) GetContextWindowSize() int {
	return o.model.GetContextWindowSize()
}

// CountTokens is not supported by Ollama. Returns ErrNotSupported.
func (o *Client) CountTokens(_ context.Context, _ common.CompletionRequest) (common.TokenCount, error) {
	return common.TokenCount{}, fmt.Errorf("ollama count tokens: %w", common.ErrNotSupported)
}

func (o *Client) ListModels(ctx context.Context) ([]common.ModelInfo, error) {
	var modelsRes ollamaModelsResponse

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodGet, o.baseURL+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("ollama create request: %w", err)
	}

	resp, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("ollama list models: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama list models: unexpected status %d", resp.StatusCode)
	}

	if err := json.NewDecoder(resp.Body).Decode(&modelsRes); err != nil {
		return nil, fmt.Errorf("ollama list models decode: %w", err)
	}

	models := make([]common.ModelInfo, 0, len(modelsRes.Models))
	for _, m := range modelsRes.Models {
		models = append(models, common.ModelInfo{
			ID:   m.Model,
			Name: m.Name,
		})
	}

	return models, nil
}

// -- HTTP helper --

func (o *Client) postJSON(ctx context.Context, path string, input any, output any) error {
	body, err := json.Marshal(input)
	if err != nil {
		return fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, o.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if o.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+o.apiKey)
	}

	resp, err := o.client.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("unexpected status %d: %s", resp.StatusCode, readErrorBody(resp.Body))
	}

	return json.NewDecoder(resp.Body).Decode(output)
}

// readErrorBody reads a truncated error response body for inclusion in
// error messages.
func readErrorBody(r io.Reader) string {
	const maxErrorBody = 512
	body, err := io.ReadAll(io.LimitReader(r, maxErrorBody))
	if err != nil || len(body) == 0 {
		return "<no body>"
	}
	return strings.TrimSpace(string(body))
}

func boolPtr(v bool) *bool { return &v }

func (o *Client) ollamaOptions(req common.CompletionRequest) map[string]any {
	opts := map[string]any{}
	if req.MaxTokens > 0 {
		opts["num_predict"] = req.MaxTokens
	}
	contextSize := o.model.GetContextWindowSize()
	if contextSize > 0 {
		opts["num_ctx"] = contextSize
	}
	if len(opts) == 0 {
		return nil
	}
	return opts
}

// -- converters --

func toOllamaMessages(req common.CompletionRequest, log logger.Logger) []ollamaChatMessage {
	var msgs []ollamaChatMessage

	if req.System != "" {
		msgs = append(msgs, ollamaChatMessage{
			Role:    "system",
			Content: req.System,
		})
	}

	for _, msg := range req.Messages {
		switch msg.Role {
		case common.RoleSystem:
			msgs = append(msgs, ollamaChatMessage{
				Role:    "system",
				Content: common.CombinedText(msg.Content),
			})
		case common.RoleUser:
			msgs = append(msgs, ollamaChatMessage{
				Role:    "user",
				Content: common.CombinedText(msg.Content),
			})
		case common.RoleAssistant:
			assistant := ollamaChatMessage{
				Role:    "assistant",
				Content: common.CombinedText(msg.Content),
			}
			for _, block := range msg.Content {
				if block.Type != common.ContentTypeToolUse {
					continue
				}
				var args map[string]any
				if block.ToolInput != nil {
					if err := json.Unmarshal(block.ToolInput, &args); err != nil {
						logger.Logf(log, "ollama: skip malformed tool input for %s: %v", block.ToolName, err)
						continue
					}
				}
				assistant.ToolCalls = append(assistant.ToolCalls, ollamaToolCall{
					Function: ollamaToolCallFunction{
						Name:      block.ToolName,
						Arguments: args,
					},
				})
			}
			msgs = append(msgs, assistant)
		case common.RoleTool:
			for _, block := range msg.Content {
				if block.Type == common.ContentTypeToolResult {
					msgs = append(msgs, ollamaChatMessage{
						Role:     "tool",
						Content:  block.ToolOutput,
						ToolName: block.ToolName,
					})
				}
			}
		}
	}

	return msgs
}

func toOllamaTools(tools []common.ToolDefinition, log logger.Logger) ([]ollamaTool, error) {
	result := make([]ollamaTool, 0, len(tools))
	for _, tool := range tools {
		var params map[string]any
		if tool.InputSchema != nil {
			if err := json.Unmarshal(tool.InputSchema, &params); err != nil {
				return nil, fmt.Errorf("ollama tool %q: parse input schema: %w", tool.Name, err)
			}
		}

		result = append(result, ollamaTool{
			Type: "function",
			Function: ollamaToolFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  params,
			},
		})
	}
	return result, nil
}

// StripThinkBlocks removes <think>...</think> blocks that Qwen 3.5 models
// emit in thinking mode. This is a safety net in case think:false is ignored.
// Exported so callers accumulating streaming text can also strip think blocks.
func StripThinkBlocks(s string) string {
	return strings.TrimSpace(thinkBlockRe.ReplaceAllString(s, ""))
}

type thinkBlockParser struct {
	events  chan<- common.StreamEvent
	inThink bool
	buf     strings.Builder
}

const thinkOpen = "<think>"
const thinkClose = "</think>"

func newThinkBlockParser(events chan<- common.StreamEvent) *thinkBlockParser {
	return &thinkBlockParser{events: events}
}

func (p *thinkBlockParser) Feed(text string) {
	if p.buf.Len() > 0 {
		text = p.buf.String() + text
		p.buf.Reset()
	}
	for len(text) > 0 {
		if p.inThink {
			idx := strings.Index(text, thinkClose)
			if idx >= 0 {
				if idx > 0 {
					p.events <- common.StreamEvent{Type: common.StreamEventThinking, Text: text[:idx]}
				}
				p.inThink = false
				text = text[idx+len(thinkClose):]
			} else {
				n := p.matchTagPrefix(text, thinkClose)
				if n > 0 {
					if len(text)-n > 0 {
						p.events <- common.StreamEvent{Type: common.StreamEventThinking, Text: text[:len(text)-n]}
					}
					p.buf.WriteString(text[len(text)-n:])
				} else {
					p.events <- common.StreamEvent{Type: common.StreamEventThinking, Text: text}
				}
				return
			}
		} else {
			idx := strings.Index(text, thinkOpen)
			if idx >= 0 {
				if idx > 0 {
					p.events <- common.StreamEvent{Type: common.StreamEventDelta, Text: text[:idx]}
				}
				p.inThink = true
				text = text[idx+len(thinkOpen):]
			} else {
				n := p.matchTagPrefix(text, thinkOpen)
				if n > 0 {
					if len(text)-n > 0 {
						p.events <- common.StreamEvent{Type: common.StreamEventDelta, Text: text[:len(text)-n]}
					}
					p.buf.WriteString(text[len(text)-n:])
				} else {
					p.events <- common.StreamEvent{Type: common.StreamEventDelta, Text: text}
				}
				return
			}
		}
	}
}

func (p *thinkBlockParser) Flush() {
	if p.buf.Len() > 0 {
		evType := common.StreamEventDelta
		if p.inThink {
			evType = common.StreamEventThinking
		}
		p.events <- common.StreamEvent{Type: evType, Text: p.buf.String()}
		p.buf.Reset()
	}
}

func (p *thinkBlockParser) matchTagPrefix(text, tag string) int {
	for i := 1; i < len(tag) && i <= len(text); i++ {
		if text[len(text)-i:] == tag[:i] {
			return i
		}
	}
	return 0
}

func fromOllamaResponse(res ollamaChatResponse, log logger.Logger) common.CompletionResponse {
	var content []common.ContentBlock

	text := StripThinkBlocks(res.Message.Content)
	if text != "" {
		content = append(content, common.NewTextContent(text))
	}

	for i, tc := range res.Message.ToolCalls {
		args, err := json.Marshal(tc.Function.Arguments)
		if err != nil {
			logger.Logf(log, "ollama: marshal tool arguments for %s: %v", tc.Function.Name, err)
			args = json.RawMessage("{}")
		}
		// Ollama returns no tool-call IDs, so this ID is synthetic
		// (name+index, unique within one response). It only needs to pair
		// tool results with calls — don't expect provider-native formats
		// like Anthropic's "toolu_..." when debugging result matching.
		id := fmt.Sprintf("call_%s_%d", tc.Function.Name, i)
		content = append(content, common.NewToolUseContent(id, tc.Function.Name, args))
	}

	responseID := fmt.Sprintf("ollama-%s-%d", res.Model, res.PromptEvalCount+res.EvalCount)

	return common.CompletionResponse{
		ID:         responseID,
		Content:    content,
		StopReason: fromOllamaStopReason(res),
		Usage: common.Usage{
			InputTokens:  int64(res.PromptEvalCount),
			OutputTokens: int64(res.EvalCount),
		},
		Model: res.Model,
	}
}

func fromOllamaStopReason(res ollamaChatResponse) common.StopReason {
	if len(res.Message.ToolCalls) > 0 {
		return common.StopReasonToolUse
	}
	switch res.DoneReason {
	case "stop":
		return common.StopReasonStop
	case "length":
		return common.StopReasonMaxTokens
	default:
		return common.StopReason(res.DoneReason)
	}
}
