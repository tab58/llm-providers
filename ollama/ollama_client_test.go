package ollama

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/tab58/llm-providers/common"
)

func newOllamaTestClient(t *testing.T, handler http.HandlerFunc) common.LLM {
	t.Helper()
	srv := httptest.NewServer(handler)
	t.Cleanup(srv.Close)
	return NewClient(Config{}, WithBaseURL(srv.URL))
}

func TestOllama_SendSyncMessage(t *testing.T) {
	client := newOllamaTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{
			"model": "test-model",
			"message": {"role": "assistant", "content": "<think>hmm</think>hello"},
			"done": true, "done_reason": "stop",
			"prompt_eval_count": 8, "eval_count": 4
		}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	})

	res, err := client.SendSyncMessage(context.Background(), common.CompletionRequest{
		Model:    "test-model",
		Messages: []common.Message{common.NewUserMessage("hi")},
	})
	if err != nil {
		t.Fatalf("SendSyncMessage: %v", err)
	}

	if res.Text() != "hello" {
		t.Errorf("text = %q, want hello (think block stripped)", res.Text())
	}
	if res.StopReason != common.StopReasonStop {
		t.Errorf("stop reason = %q, want %q", res.StopReason, common.StopReasonStop)
	}
	if res.Usage.InputTokens != 8 || res.Usage.OutputTokens != 4 {
		t.Errorf("usage = %+v, want 8/4", res.Usage)
	}
}

func TestOllama_SendMessageWithTools(t *testing.T) {
	client := newOllamaTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{
			"model": "test-model",
			"message": {"role": "assistant", "content": "", "tool_calls": [
				{"function": {"name": "read_file", "arguments": {"path": "main.go"}}}
			]},
			"done": true, "done_reason": "stop",
			"prompt_eval_count": 8, "eval_count": 4
		}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	})

	res, err := client.SendMessageWithTools(context.Background(), common.CompletionRequest{
		Model:    "test-model",
		Messages: []common.Message{common.NewUserMessage("read main.go")},
	}, []common.ToolDefinition{{
		Name:        "read_file",
		Description: "Reads a file",
		InputSchema: []byte(`{"type":"object","properties":{"path":{"type":"string"}}}`),
	}})
	if err != nil {
		t.Fatalf("SendMessageWithTools: %v", err)
	}

	calls := res.ToolCalls()
	if len(calls) != 1 || calls[0].ToolName != "read_file" {
		t.Fatalf("tool calls = %+v, want one read_file call", calls)
	}
	if res.StopReason != common.StopReasonToolUse {
		t.Errorf("stop reason = %q, want %q", res.StopReason, common.StopReasonToolUse)
	}
}

func TestOllama_ErrorIncludesResponseBody(t *testing.T) {
	client := newOllamaTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error": "model not found"}`, http.StatusNotFound)
	})

	_, err := client.SendSyncMessage(context.Background(), common.CompletionRequest{
		Model:    "missing-model",
		Messages: []common.Message{common.NewUserMessage("hi")},
	})
	if err == nil {
		t.Fatal("want error, got nil")
	}
	if !strings.Contains(err.Error(), "404") || !strings.Contains(err.Error(), "model not found") {
		t.Errorf("error should include status and body: %v", err)
	}
}

func TestOllama_ListModels(t *testing.T) {
	client := newOllamaTestClient(t, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"models": [
			{"name": "qwen3.5:9b", "model": "qwen3.5:9b"},
			{"name": "gemma4:31b", "model": "gemma4:31b"}
		]}`)); err != nil {
			t.Errorf("write response: %v", err)
		}
	})

	models, err := client.ListModels(context.Background())
	if err != nil {
		t.Fatalf("ListModels: %v", err)
	}
	if len(models) != 2 || models[0].ID != "qwen3.5:9b" || models[1].ID != "gemma4:31b" {
		t.Errorf("models = %+v, want qwen3.5:9b and gemma4:31b", models)
	}
}

func TestOllama_CountTokensNotSupported(t *testing.T) {
	client := NewClient(Config{})
	_, err := client.CountTokens(context.Background(), common.CompletionRequest{})
	if !errors.Is(err, common.ErrNotSupported) {
		t.Fatalf("want ErrNotSupported, got %v", err)
	}
}

func TestOllama_GetCurrentModel(t *testing.T) {
	client := NewClient(Config{})
	if got := client.GetCurrentModel(); got != Model_Gemma4_31B.GetName() {
		t.Errorf("GetCurrentModel() = %q, want default %q", got, Model_Gemma4_31B.GetName())
	}
}

// Ollama streams assistant text spread across many chunks; the final done
// chunk usually carries empty content. The terminal StreamEventStop response
// must contain the full accumulated text, or agent loops see an empty final
// answer (and retry pointlessly) whenever the model streams.
func TestOllama_SendStreamingMessage_AccumulatesTextAcrossChunks(t *testing.T) {
	client := newOllamaTestClient(t, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		chunks := []string{
			`{"model":"glm-5.2:cloud","message":{"role":"assistant","content":"The root "},"done":false}`,
			`{"model":"glm-5.2:cloud","message":{"role":"assistant","content":"cause is a "},"done":false}`,
			`{"model":"glm-5.2:cloud","message":{"role":"assistant","content":"deadlock."},"done":false}`,
			`{"model":"glm-5.2:cloud","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","eval_count":12}`,
		}
		for _, c := range chunks {
			_, _ = w.Write([]byte(c + "\n"))
		}
	})

	events := make(chan common.StreamEvent, 64)
	go func() {
		if err := client.SendStreamingMessage(context.Background(), common.CompletionRequest{}, events); err != nil {
			t.Errorf("SendStreamingMessage: %v", err)
		}
	}()

	var stop *common.CompletionResponse
	for ev := range events {
		if ev.Type == common.StreamEventStop {
			stop = ev.Response
		}
	}
	if stop == nil {
		t.Fatal("no StreamEventStop received")
	}
	var text string
	for _, block := range stop.Content {
		if block.Type == common.ContentTypeText {
			text += block.Text
		}
	}
	if text != "The root cause is a deadlock." {
		t.Errorf("final response text = %q, want full accumulated text", text)
	}
}
