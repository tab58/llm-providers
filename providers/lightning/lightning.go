package lightning

import (
	"github.com/tab58/llm-providers/providers/openai_compat"
	"github.com/tab58/llm-providers/utils"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
)

type Model string

const (
	ModelGemma4_31B  = Model("lightning-ai/gemma-4-31B-it")
	ModelGPTOSS_120B = Model("lightning-ai/gpt-oss-120b")

	MaxTokensGemma4_31B int64 = 245000

	ContextWindowGemma4_31B = 131_000
	ContextWindowGPTOSS120B = 128_000
)

var lightningContextWindows = map[Model]int{
	ModelGemma4_31B:  ContextWindowGemma4_31B,
	ModelGPTOSS_120B: ContextWindowGPTOSS120B,
}

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
	rateLimiter *utils.TokenBucket
	haveLimiter bool
}

// Option is a functional option for configuring the Lightning client.
type Option func(*options)

// WithNoRateLimit disables rate limiting for the Lightning client.
func WithNoRateLimit() Option {
	return func(o *options) {
		o.rateLimiter = nil
		o.haveLimiter = true
	}
}

// NewClient creates a Lightning.ai LLM client using the
// OpenAI-compatible API. Requests rejected with HTTP 429 are retried with
// exponential backoff.
func NewClient(cfg Config, opts ...Option) *Client {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(cfg.BaseURL),
	)
	model := cfg.Model
	if model == "" {
		model = ModelGemma4_31B
	}

	var o options
	for _, opt := range opts {
		opt(&o)
	}
	if !o.haveLimiter {
		o.rateLimiter = utils.NewTokenBucket(utils.TokenBucketConfig{
			Rate:           0.25, // 15 requests per minute
			BurstSize:      3,    // small burst to avoid hitting server-side limits
			MaxConcurrency: 3,
		})
	}

	return &Client{&openai_compat.Client{
		Name:           "lightning",
		Client:         &client,
		Model:          string(model),
		ContextWindow:  lightningContextWindows[model],
		RateLimiter:    o.rateLimiter,
		RetryRateLimit: true,
	}}
}
