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
	Model  Model
}

type options struct {
	noRateLimit bool
	// BaseURL points at a deployment-specific endpoint when set. Empty uses
	// the shared Lightning.ai Model API endpoint.
	baseURL string
}

// Option is a functional option for configuring the Lightning client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the Lightning client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.noRateLimit = true
	}
}

// WithBaseURL sets a custom base URL for the Lightning client. This is useful
// for testing or for connecting to a deployment-specific endpoint.
func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		if baseURL != "" {
			o.baseURL = baseURL
		}
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

	// apply functional options
	o := options{
		baseURL: lightningBaseURL,
	}
	for _, opt := range opts {
		opt(&o)
	}

	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(o.baseURL),
	)
	raw := &Client{&openai_compat.Client{
		Name:           string(common.ProviderLightning),
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
