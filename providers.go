package provider

import (
	"github.com/tab58/llm-providers/providers/anthropic"
	"github.com/tab58/llm-providers/providers/cerebras"
	"github.com/tab58/llm-providers/providers/common"
	"github.com/tab58/llm-providers/providers/lightning"
	"github.com/tab58/llm-providers/providers/ollama"
	"github.com/tab58/llm-providers/providers/openai"
	"github.com/tab58/llm-providers/providers/openrouter"
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
