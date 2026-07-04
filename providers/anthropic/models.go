package anthropic

import (
	anthropicSDK "github.com/anthropics/anthropic-sdk-go"
	"github.com/tab58/llm-providers/providers/common"
)

type Model = common.Model

var (
	Model_ClaudeOpus4_6 Model = common.ModelDefinition{
		Name:              anthropicSDK.ModelClaudeOpus4_6,
		MaxTokens:         128_000,
		ContextWindowSize: 1_000_000,
	}

	Model_ClaudeSonnet4_6 Model = common.ModelDefinition{
		Name:              anthropicSDK.ModelClaudeSonnet4_6,
		MaxTokens:         128_000,
		ContextWindowSize: 1_000_000,
	}

	Model_ClaudeHaiku4_5 Model = common.ModelDefinition{
		Name:              anthropicSDK.ModelClaudeHaiku4_5,
		MaxTokens:         64_000,
		ContextWindowSize: 200_000,
	}
)
