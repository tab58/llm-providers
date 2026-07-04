package provider

import (
	"testing"

	"github.com/tab58/llm-providers/anthropic"
	"github.com/tab58/llm-providers/cerebras"
	"github.com/tab58/llm-providers/common"
	"github.com/tab58/llm-providers/lightning"
	"github.com/tab58/llm-providers/ollama"
	"github.com/tab58/llm-providers/openai"
	"github.com/tab58/llm-providers/openrouter"
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
			string(anthropic.Model_ClaudeSonnet4_6.GetName()),
		},
		{
			"openai default",
			openai.NewClient(openai.Config{APIKey: "k"}),
			string(openai.Model_GPT5_4.GetName()),
		},
		{
			"cerebras default",
			cerebras.NewClient(cerebras.Config{APIKey: "k"}),
			string(cerebras.Model_GPTOSS_120B.GetName()),
		},
		{
			"cerebras override",
			cerebras.NewClient(cerebras.Config{APIKey: "k", Model: common.ModelDefinition{Name: "custom"}}),
			"custom",
		},
		{
			"lightning default",
			lightning.NewClient(lightning.Config{APIKey: "k", BaseURL: "https://example.test"}),
			string(lightning.Model_Gemma4_31B.GetName()),
		},
		{
			"openrouter default",
			openrouter.NewClient(openrouter.Config{APIKey: "k"}),
			string(openrouter.Model_Gemma4_31B.GetName()),
		},
		{
			"ollama default",
			ollama.NewClient(ollama.Config{}),
			string(ollama.Model_Gemma4_31B.GetName()),
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

func TestModelProviders(t *testing.T) {
	tests := []struct {
		name  string
		model common.Model
		want  common.Provider
	}{
		{"anthropic opus", anthropic.Model_ClaudeOpus4_6, common.ProviderAnthropic},
		{"anthropic sonnet", anthropic.Model_ClaudeSonnet4_6, common.ProviderAnthropic},
		{"anthropic haiku", anthropic.Model_ClaudeHaiku4_5, common.ProviderAnthropic},
		{"cerebras gpt-oss", cerebras.Model_GPTOSS_120B, common.ProviderCerebras},
		{"lightning gemma", lightning.Model_Gemma4_31B, common.ProviderLightning},
		{"lightning gpt-oss", lightning.Model_GPTOSS_120B, common.ProviderLightning},
		{"ollama qwen 9b", ollama.Model_Qwen3_5_9B, common.ProviderOllama},
		{"ollama qwen 35b", ollama.Model_Qwen3_5_35B, common.ProviderOllama},
		{"ollama qwen 122b", ollama.Model_Qwen3_5_122B, common.ProviderOllama},
		{"ollama gemma", ollama.Model_Gemma4_31B, common.ProviderOllama},
		{"openai gpt-5.4", openai.Model_GPT5_4, common.ProviderOpenAI},
		{"openai gpt-5.4-mini", openai.Model_GPT5_4Mini, common.ProviderOpenAI},
		{"openrouter gemma", openrouter.Model_Gemma4_31B, common.ProviderOpenRouter},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.model.GetProvider(); got != tt.want {
				t.Errorf("GetProvider() = %q, want %q", got, tt.want)
			}
		})
	}
}
