package openai

import (
	"github.com/openai/openai-go/v3"
	"github.com/tab58/llm-providers/common"
)

type Model = common.Model

var (
	Model_GPT5_4 Model = common.ModelDefinition{
		Name:              openai.ChatModelGPT5_4,
		MaxTokens:         128_000,
		ContextWindowSize: 1_050_000,
	}

	Model_GPT5_4Mini Model = common.ModelDefinition{
		Name:              openai.ChatModelGPT5_4Mini,
		MaxTokens:         128_000,
		ContextWindowSize: 400_000,
	}
)
