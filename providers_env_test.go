package provider

import (
	"errors"
	"strings"
	"testing"

	"github.com/tab58/llm-providers/common"
)

func TestLLMFromEnv(t *testing.T) {
	tests := []struct {
		name       string
		model      common.ModelDefinition
		env        map[string]string
		wantErr    bool
		wantErrSub string
		wantErrIs  error
	}{
		{
			name:  "anthropic with key",
			model: common.ModelDefinition{Name: "claude-x", Provider: common.ProviderAnthropic},
			env:   map[string]string{"ANTHROPIC_API_KEY": "test-key"},
		},
		{
			name:       "anthropic missing key",
			model:      common.ModelDefinition{Name: "claude-x", Provider: common.ProviderAnthropic},
			env:        map[string]string{},
			wantErr:    true,
			wantErrSub: "ANTHROPIC_API_KEY",
		},
		{
			name:  "openai with key",
			model: common.ModelDefinition{Name: "gpt-x", Provider: common.ProviderOpenAI},
			env:   map[string]string{"OPENAI_API_KEY": "test-key"},
		},
		{
			name:       "openrouter missing key",
			model:      common.ModelDefinition{Name: "m", Provider: common.ProviderOpenRouter},
			env:        map[string]string{},
			wantErr:    true,
			wantErrSub: "OPENROUTER_API_KEY",
		},
		{
			name:  "ollama needs no key",
			model: common.ModelDefinition{Name: "glm-5.2", Provider: common.ProviderOllama},
			env:   map[string]string{},
		},
		{
			name:       "unknown provider",
			model:      common.ModelDefinition{Name: "m", Provider: common.Provider("nope")},
			env:        map[string]string{},
			wantErr:    true,
			wantErrSub: "unknown provider",
			wantErrIs:  common.ErrUnknownProvider,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Clear all known env keys so ambient shell config can't leak in,
			// then set only what the case declares. t.Setenv also restores.
			for _, key := range []string{
				"ANTHROPIC_API_KEY", "CEREBRAS_API_KEY", "LIGHTNING_API_KEY",
				"OLLAMA_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
			} {
				t.Setenv(key, "")
			}
			for k, v := range tt.env {
				t.Setenv(k, v)
			}

			llm, err := LLMFromEnv(tt.model)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("LLMFromEnv() error = nil, want error")
				}
				if tt.wantErrIs != nil && !errors.Is(err, tt.wantErrIs) {
					t.Fatalf("LLMFromEnv() error = %v, want errors.Is(%v)", err, tt.wantErrIs)
				}
				if tt.wantErrSub != "" && !strings.Contains(err.Error(), tt.wantErrSub) {
					t.Fatalf("LLMFromEnv() error = %q, want substring %q", err.Error(), tt.wantErrSub)
				}
				return
			}
			if err != nil {
				t.Fatalf("LLMFromEnv() error: %v", err)
			}
			if llm == nil {
				t.Fatal("LLMFromEnv() returned nil LLM")
			}
			if got := llm.ProviderName(); got != tt.model.Provider {
				t.Errorf("ProviderName() = %q, want %q", got, tt.model.Provider)
			}
		})
	}
}
