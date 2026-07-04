package lightning

import (
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/openai_compat"
	"github.com/tab58/llm-providers/ratelimit"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

// Client implements the LLM interface using Client.ai's OpenAI-compatible API.
type Client struct {
	*openai_compat.Client
}

// Config holds configuration for connecting to the Lightning.ai API.
// BaseURL is required: Lightning.ai endpoints are deployment-specific.
type Config struct {
	APIKey  string
	BaseURL string
	Model   Model
}

type options struct {
	noRateLimit bool
}

// Option is a functional option for configuring the Lightning client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the Lightning client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.noRateLimit = true
	}
}

// NewClient creates a Lightning.ai LLM client using the
// OpenAI-compatible API. Requests rejected with HTTP 429 are retried with
// exponential backoff.
func NewClient(cfg Config, opts ...Option) common.LLM {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(cfg.BaseURL),
	)
	model := cfg.Model
	if model == nil {
		model = Model_Gemma4_31B
	}

	var o options
	for _, opt := range opts {
		opt(&o)
	}

	raw := &Client{&openai_compat.Client{
		Name:           "lightning",
		Client:         &client,
		Model:          model,
		RetryRateLimit: true,
	}}
	if o.noRateLimit {
		return raw
	}

	limiter := ratelimit.NewTokenBucket(ratelimit.TokenBucketConfig{
		Rate:           0.25, // 15 requests per minute
		BurstSize:      3,    // small burst to avoid hitting server-side limits
		MaxConcurrency: 3,
	})
	return ratelimit.Wrap(raw, limiter, ratelimit.CostPerRequest)
}
