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
