package provider

import (
	"github.com/tab58/llm-providers/anthropic"
	"github.com/tab58/llm-providers/cerebras"
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/lightning"
	"github.com/tab58/llm-providers/ollama"
	"github.com/tab58/llm-providers/openai"
	"github.com/tab58/llm-providers/openrouter"
)

// Compile-time interface compliance checks for all providers.
var (
	_ common.LLM = (*anthropic.Client)(nil)
	_ common.LLM = (*cerebras.Client)(nil)
	_ common.LLM = (*lightning.Client)(nil)
	_ common.LLM = (*ollama.Client)(nil)
	_ common.LLM = (*openai.Client)(nil)
	_ common.LLM = (*openrouter.Client)(nil)
)

type options struct {
	baseURL string
}

type Option func(*options)

func WithBaseURL(baseURL string) Option {
	return func(o *options) {
		o.baseURL = baseURL
	}
}

func LLMFromModel(apiKey string, model common.ModelDefinition, opts ...Option) (common.LLM, error) {
	o := options{}
	for _, opt := range opts {
		opt(&o)
	}

	switch model.Provider {
	case common.ProviderAnthropic:
		return anthropic.NewClient(anthropic.Config{
			APIKey: apiKey,
			Model:  model,
		}), nil
	case common.ProviderOpenAI:
		return openai.NewClient(openai.Config{
			APIKey: apiKey,
			Model:  model,
		}), nil
	case common.ProviderCerebras:
		return cerebras.NewClient(cerebras.Config{
			APIKey: apiKey,
			Model:  model,
		}), nil
	case common.ProviderLightning:
		return lightning.NewClient(
			lightning.Config{
				APIKey: apiKey,
				Model:  model,
			},
			lightning.WithBaseURL(o.baseURL),
		), nil
	case common.ProviderOllama:
		return ollama.NewClient(
			ollama.Config{
				APIKey: apiKey,
				Model:  model,
			},
			ollama.WithBaseURL(o.baseURL),
		), nil
	case common.ProviderOpenRouter:
		return openrouter.NewClient(
			openrouter.Config{
				APIKey: apiKey,
				Model:  model,
			},
			openrouter.WithBaseURL(o.baseURL),
		), nil
	default:
		return nil, common.ErrUnknownProvider
	}
}
