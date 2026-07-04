package openai

import (
	"github.com/tab58/llm-providers/providers/openai_compat"
	"github.com/tab58/llm-providers/utils"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
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
	rateLimiter *utils.TokenBucket
	haveLimiter bool
}

// Option is a functional option for configuring the OpenAI client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the OpenAI client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.rateLimiter = nil
		o.haveLimiter = true
	}
}

// NewClient creates an OpenAI LLM client.
func NewClient(cfg Config, opts ...Option) *Client {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
	)
	model := cfg.Model
	if model == nil {
		model = Model_GPT5_4
	}

	var o options
	for _, opt := range opts {
		opt(&o)
	}
	if !o.haveLimiter {
		o.rateLimiter = utils.NewTokenBucket(utils.TokenBucketConfig{
			Rate:           10_000.0 / 60.0, // 10K input tokens per minute
			BurstSize:      10_000,
			MaxConcurrency: 10,
		})
	}

	return &Client{&openai_compat.Client{
		Name:                   "openai",
		Client:                 &client,
		Model:                  model,
		RateLimiter:            o.rateLimiter,
		TokenCostLimit:         true,
		UseMaxCompletionTokens: true,
	}}
}
