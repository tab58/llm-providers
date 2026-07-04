package openrouter

import "github.com/tab58/llm-providers/providers/common"

type Model = common.Model

var (
	Model_Gemma4_31B Model = common.ModelDefinition{
		Name:                 "google/gemma-4-31b-it",
		MaxTokens:            32_768,
		ContextWindowSize:    262_144,
		DefaultContextWindow: 262_144,
	}
)
