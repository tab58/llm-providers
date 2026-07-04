package cerebras

import (
	"github.com/tab58/llm-providers/providers/openai_compat"
	"github.com/tab58/llm-providers/utils"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	cerebrasBaseURL = "https://api.cerebras.ai/v1"
)

// Client implements the LLM interface using Client's OpenAI-compatible API.
type Client struct {
	*openai_compat.Client
}

// Config holds configuration for connecting to the Cerebras API.
type Config struct {
	APIKey string
	Model  Model
}

type options struct {
	rateLimiter *utils.TokenBucket
	haveLimiter bool
}

// Option is a functional option for configuring the Cerebras client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the Cerebras client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.rateLimiter = nil
		o.haveLimiter = true
	}
}

// NewClient creates a Cerebras LLM client using the OpenAI-compatible API.
func NewClient(cfg Config, opts ...Option) *Client {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(cerebrasBaseURL),
	)
	model := cfg.Model
	if model == nil {
		model = Model_GPTOSS_120B
	}

	var o options
	for _, opt := range opts {
		opt(&o)
	}
	if !o.haveLimiter {
		o.rateLimiter = utils.NewTokenBucket(utils.TokenBucketConfig{
			Rate:           30.0 / 60.0, // 30 requests per minute
			BurstSize:      1,
			MaxConcurrency: 10,
		})
	}

	return &Client{&openai_compat.Client{
		Name:           "cerebras",
		Client:         &client,
		Model:          openai_compat.Model(model),
		RateLimiter:    o.rateLimiter,
		RetryRateLimit: true,
	}}
}
