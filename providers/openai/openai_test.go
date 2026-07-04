package openai

import "testing"

func TestNewOpenAIClient_SetsModel(t *testing.T) {
	client := NewClient(Config{APIKey: "test"})
	if got := client.GetCurrentModel(); got != string(ModelGPT5_4) {
		t.Errorf("GetCurrentModel() = %q, want %q", got, ModelGPT5_4)
	}

	client = NewClient(Config{APIKey: "test", Model: ModelGPT5_4Mini})
	if got := client.GetCurrentModel(); got != string(ModelGPT5_4Mini) {
		t.Errorf("GetCurrentModel() = %q, want %q", got, ModelGPT5_4Mini)
	}
}
