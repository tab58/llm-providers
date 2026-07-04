package cerebras

import "github.com/tab58/llm-providers/providers/common"

type Model = common.Model

var (
	// Paid-tier limits; Cerebras free tier caps at 65K context / 32K output.
	Model_GPTOSS_120B Model = common.ModelDefinition{
		Name:                 "gpt-oss-120b",
		MaxTokens:            40_960,
		ContextWindowSize:    131_072,
		DefaultContextWindow: common.ContextWindowDefault,
	}
)
