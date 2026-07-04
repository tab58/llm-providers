package common

import (
	"context"
	"errors"
)

var ErrNotSupported = errors.New("operation not supported by this provider")

const (
	MaxTokensStdResponse int64 = 32768
)

// openAICompat implements the LLM interface for any OpenAI-compatible chat
// completions API. Provider types (OpenAI, Cerebras, Lightning, OpenRouter)
// embed it and differ only in configuration.
const ContextWindowDefault = 128_000

type LLM interface {
	// SendSyncMessage sends a completion request and returns the full response.
	SendSyncMessage(ctx context.Context, req CompletionRequest) (CompletionResponse, error)

	// SendStreamingMessage sends a completion request and streams response events
	// to the provided channel. The channel is closed when the stream is complete.
	SendStreamingMessage(ctx context.Context, req CompletionRequest, events chan<- StreamEvent) error

	// SendMessageWithTools sends a completion request with tool definitions.
	// Returns tool calls in the response content when the model invokes tools.
	SendMessageWithTools(ctx context.Context, req CompletionRequest, tools []ToolDefinition) (CompletionResponse, error)

	// CountTokens estimates token count for the given request without executing it.
	CountTokens(ctx context.Context, req CompletionRequest) (TokenCount, error)

	// ListModels returns the models available from this provider.
	ListModels(ctx context.Context) ([]ModelInfo, error)

	// Gets the current model.
	GetCurrentModel() string

	// GetContextWindowSize returns the model's total context window in tokens.
	GetContextWindowSize() int
}

// Model represents a model available from a provider.
type Model interface {
	// GetName returns the model's name.
	GetName() string

	// GetMaxTokens returns the model's maximum token limit.
	GetMaxTokens() int

	// GetContextWindowSize returns the model's total context window in tokens.
	GetContextWindowSize() int
}

// ModelDefinition represents a model available from a provider.
// Implementations of the Model interface can be used to define models for specific providers.
type ModelDefinition struct {
	Name                 string
	MaxTokens            int
	ContextWindowSize    int
	DefaultContextWindow int
}

func (m ModelDefinition) GetName() string {
	return m.Name
}

func (m ModelDefinition) GetMaxTokens() int {
	return m.MaxTokens
}

func (m ModelDefinition) GetContextWindowSize() int {
	if m.ContextWindowSize > 0 {
		return m.ContextWindowSize
	}
	return m.DefaultContextWindow
}
