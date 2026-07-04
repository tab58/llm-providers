package ollama

import (
	"testing"

	"github.com/tab58/llm-providers/common"
)

func TestOllamaOptions_NumCtx(t *testing.T) {
	tests := []struct {
		name        string
		contextSize int64
		wantNumCtx  int64
	}{
		{"defaults to model's default window, not its max", 0, int64(Model_Gemma4_31B.GetDefaultContextWindow())},
		{"explicit ContextSize overrides", 8192, 8192},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// WithNoRateLimit returns the raw client, so the assertion is safe.
			c := NewClient(Config{ContextSize: tt.contextSize}, WithNoRateLimit()).(*Client)
			opts := c.ollamaOptions(common.CompletionRequest{})
			if got := opts["num_ctx"]; got != tt.wantNumCtx {
				t.Errorf("num_ctx = %v, want %v", got, tt.wantNumCtx)
			}
		})
	}

	if max := Model_Gemma4_31B.GetContextWindowSize(); Model_Gemma4_31B.GetDefaultContextWindow() >= max {
		t.Errorf("default window %d should be below model max %d", Model_Gemma4_31B.GetDefaultContextWindow(), max)
	}
}

func TestGetContextWindowSize_EffectiveWindow(t *testing.T) {
	if got := NewClient(Config{ContextSize: 8192}).GetContextWindowSize(); got != 8192 {
		t.Errorf("GetContextWindowSize() = %d, want 8192", got)
	}
	if got := NewClient(Config{}).GetContextWindowSize(); got != Model_Gemma4_31B.GetDefaultContextWindow() {
		t.Errorf("GetContextWindowSize() = %d, want model default %d", got, Model_Gemma4_31B.GetDefaultContextWindow())
	}
}
