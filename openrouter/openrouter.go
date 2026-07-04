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

type options struct {
	baseURL string
}

// Option is a functional option for configuring the OpenRouter client.
type Option func(*options)

// WithBaseURL overrides the API endpoint (e.g. a proxy or gateway).
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.baseURL = url
	}
}

// NewClient creates an OpenRouter LLM client using the
// OpenAI-compatible API. OpenRouter has no default client-side rate limit;
// wrap the result with ratelimit.Wrap to add one.
func NewClient(cfg Config, opts ...Option) common.LLM {
	model := cfg.Model
	if model == nil {
		model = Model_Gemma4_31B
	}

	// apply functional options
	o := options{
		baseURL: openRouterBaseURL,
	}
	for _, opt := range opts {
		opt(&o)
	}

	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(o.baseURL),
	)
	return &Client{&openai_compat.Client{
		Name:   "openrouter",
		Client: &client,
		Model:  model,
	}}
}

func (c *Client) ProviderName() common.Provider {
	return common.ProviderOpenRouter
}
