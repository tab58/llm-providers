package lightning

import (
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/openai_compat"
	"github.com/tab58/llm-providers/ratelimit"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

const (
	// lightningBaseURL is the shared Lightning.ai Model API endpoint.
	lightningBaseURL = "https://lightning.ai/api/v1"
)

// Client implements the LLM interface using Client.ai's OpenAI-compatible API.
type Client struct {
	*openai_compat.Client
}

// Config holds configuration for connecting to the Lightning.ai API.
type Config struct {
	APIKey string
	// BaseURL points at a deployment-specific endpoint when set. Empty uses
	// the shared Lightning.ai Model API endpoint.
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
	model := cfg.Model
	if model == nil {
		model = Model_Gemma4_31B
	}

	var o options
	for _, opt := range opts {
		opt(&o)
	}

	baseURL := cfg.BaseURL
	if baseURL == "" {
		baseURL = lightningBaseURL
	}
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(baseURL),
	)
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

func (c *Client) ProviderName() common.Provider {
	return common.ProviderLightning
}
