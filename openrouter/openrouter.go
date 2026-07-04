package openrouter

import (
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/openai_compat"
)

const (
	openRouterBaseURL = "https://openrouter.ai/api/v1"
)

// Client implements the LLM interface using Client's
// OpenAI-compatible API.
type Client struct {
	*openai_compat.Client
}

// Config holds configuration for connecting to the OpenRouter API.
type Config struct {
	APIKey string
	Model  Model
}

// NewClient creates an OpenRouter LLM client using the
// OpenAI-compatible API. OpenRouter has no default client-side rate limit;
// wrap the result with ratelimit.Wrap to add one.
func NewClient(cfg Config) common.LLM {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(openRouterBaseURL),
	)
	model := cfg.Model
	if model == nil {
		model = Model_Gemma4_31B
	}

	return &Client{&openai_compat.Client{
		Name:   "openrouter",
		Client: &client,
		Model:  model,
	}}
}
