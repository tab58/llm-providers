package openrouter

import (
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/option"
	"github.com/tab58/llm-providers/providers/openai_compat"
)

type Model string

const (
	ModelGemma4_31B = Model("google/gemma-4-31b-it")

	openRouterBaseURL = "https://openrouter.ai/api/v1"

	ContextWindowGemma4_31B = 131_000
)

var openRouterContextWindows = map[Model]int{
	ModelGemma4_31B: ContextWindowGemma4_31B,
}

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
// OpenAI-compatible API.
func NewClient(cfg Config) *Client {
	client := openai.NewClient(
		option.WithAPIKey(cfg.APIKey),
		option.WithBaseURL(openRouterBaseURL),
	)
	model := cfg.Model
	if model == "" {
		model = ModelGemma4_31B
	}

	return &Client{&openai_compat.Client{
		Name:          "openrouter",
		Client:        &client,
		Model:         string(model),
		ContextWindow: openRouterContextWindows[model],
	}}
}
