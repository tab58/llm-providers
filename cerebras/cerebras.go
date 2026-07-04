package cerebras

import (
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/openai_compat"
	"github.com/tab58/llm-providers/ratelimit"

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
	noRateLimit bool
}

// Option is a functional option for configuring the Cerebras client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the Cerebras client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.noRateLimit = true
	}
}

// NewClient creates a Cerebras LLM client using the OpenAI-compatible API.
func NewClient(cfg Config, opts ...Option) common.LLM {
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

	raw := &Client{&openai_compat.Client{
		Name:           "cerebras",
		Client:         &client,
		Model:          openai_compat.Model(model),
		RetryRateLimit: true,
	}}
	if o.noRateLimit {
		return raw
	}

	limiter := ratelimit.NewTokenBucket(ratelimit.TokenBucketConfig{
		Rate:           30.0 / 60.0, // 30 requests per minute
		BurstSize:      1,
		MaxConcurrency: 10,
	})
	return ratelimit.Wrap(raw, limiter, ratelimit.CostPerRequest)
}
