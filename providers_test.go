package provider

import (
	"testing"

	"github.com/tab58/llm-providers/providers/anthropic"
	"github.com/tab58/llm-providers/providers/cerebras"
	"github.com/tab58/llm-providers/providers/common"
	"github.com/tab58/llm-providers/providers/lightning"
	"github.com/tab58/llm-providers/providers/ollama"
	"github.com/tab58/llm-providers/providers/openai"
	"github.com/tab58/llm-providers/providers/openrouter"
)

func TestConstructorDefaults(t *testing.T) {
	tests := []struct {
		name      string
		llm       common.LLM
		wantModel string
	}{
		{
			"anthropic default",
			anthropic.NewClient(anthropic.Config{APIKey: "k"}),
			string(anthropic.ModelClaudeSonnet4_6),
		},
		{
			"openai default",
			openai.NewClient(openai.Config{APIKey: "k"}),
			string(openai.ModelGPT5_4),
		},
		{
			"cerebras default",
			cerebras.NewClient(cerebras.Config{APIKey: "k"}),
			string(cerebras.ModelGPTOSS120B),
		},
		{
			"cerebras override",
			cerebras.NewClient(cerebras.Config{APIKey: "k", Model: cerebras.Model("custom")}),
			"custom",
		},
		{
			"lightning default",
			lightning.NewClient(lightning.Config{APIKey: "k", BaseURL: "https://example.test"}),
			string(lightning.ModelGemma4_31B),
		},
		{
			"openrouter default",
			openrouter.NewClient(openrouter.Config{APIKey: "k"}),
			string(openrouter.ModelGemma4_31B),
		},
		{
			"ollama default",
			ollama.NewClient(ollama.Config{}),
			string(ollama.ModelGemma4_31B),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.llm.GetCurrentModel(); got != tt.wantModel {
				t.Errorf("GetCurrentModel() = %q, want %q", got, tt.wantModel)
			}
		})
	}
}
