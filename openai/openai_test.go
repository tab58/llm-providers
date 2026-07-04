package openai

import "testing"

func TestNewOpenAIClient_SetsModel(t *testing.T) {
	client := NewClient(Config{APIKey: "test"})
	if got := client.GetCurrentModel(); got != Model_GPT5_4.GetName() {
		t.Errorf("GetCurrentModel() = %q, want %q", got, Model_GPT5_4)
	}

	client = NewClient(Config{APIKey: "test", Model: Model_GPT5_4Mini})
	if got := client.GetCurrentModel(); got != Model_GPT5_4Mini.GetName() {
		t.Errorf("GetCurrentModel() = %q, want %q", got, Model_GPT5_4Mini)
	}
}
