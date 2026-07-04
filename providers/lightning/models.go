package lightning

import "github.com/tab58/llm-providers/providers/common"

type Model = common.Model

var (
	// Gemma 4 publishes no output cap; 32K is a conservative ceiling.
	Model_Gemma4_31B Model = common.ModelDefinition{
		Name:              "lightning-ai/gemma-4-31B-it",
		MaxTokens:         32_768,
		ContextWindowSize: 262_144,
	}

	Model_GPTOSS_120B Model = common.ModelDefinition{
		Name:              "lightning-ai/gpt-oss-120b",
		MaxTokens:         32_768,
		ContextWindowSize: 131_072,
	}
)
