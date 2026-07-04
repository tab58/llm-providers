package openai

import (
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/openai_compat"
	"github.com/tab58/llm-providers/ratelimit"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	openaiBaseURL = "https://api.openai.com/v1"
)

// Client implements the LLM interface using the Client API.
type Client struct {
	*openai_compat.Client
}

// Config holds configuration for connecting to the OpenAI API.
type Config struct {
	APIKey string
	Model  Model
}

type options struct {
	noRateLimit bool
	baseURL     string
}

// Option is a functional option for configuring the OpenAI client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the OpenAI client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.noRateLimit = true
	}
}

// WithBaseURL overrides the API endpoint (e.g. a proxy or gateway).
func WithBaseURL(url string) Option {
	return func(o *options) {
		o.baseURL = url
	}
}

// NewClient creates an OpenAI LLM client.
func NewClient(cfg Config, opts ...Option) common.LLM {
	model := cfg.Model
	if model == nil {
		model = Model_GPT5_4
	}

	// apply functional options
	o := options{
		baseURL: openaiBaseURL,
	}
	for _, opt := range opts {
		opt(&o)
	}

	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(o.baseURL),
	)

	raw := &Client{&openai_compat.Client{
		Name:                   "openai",
		Client:                 &client,
		Model:                  model,
		UseMaxCompletionTokens: true,
	}}
	if o.noRateLimit {
		return raw
	}

	limiter := ratelimit.NewTokenBucket(ratelimit.TokenBucketConfig{
		Rate:           10_000.0 / 60.0, // 10K input tokens per minute
		BurstSize:      10_000,
		MaxConcurrency: 10,
	})
	return ratelimit.Wrap(raw, limiter, ratelimit.CostByTokenCount)
}

func (c *Client) ProviderName() common.Provider {
	return common.ProviderOpenAI
}
