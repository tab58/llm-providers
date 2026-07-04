package openrouter

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/tab58/llm-providers/common"
)

func TestOpenRouter_BaseURLOverride(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{
			"id": "chatcmpl-1", "object": "chat.completion", "model": "test-model",
			"choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
		}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	}))
	t.Cleanup(srv.Close)

	client := NewClient(Config{APIKey: "test"}, WithBaseURL(srv.URL))
	res, err := client.SendSyncMessage(context.Background(), common.CompletionRequest{
		Model:    "test-model",
		Messages: []common.Message{common.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("SendSyncMessage: %v", err)
	}
	if res.Text() != "hi" {
		t.Errorf("text = %q, want hi", res.Text())
	}
}
