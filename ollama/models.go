package ollama

import "github.com/tab58/llm-providers/common"

type Model = common.Model

var (
	// Model_GLM5_2_Cloud is Z.ai's GLM-5.2 served via Ollama Cloud — inference
	// runs on Ollama/Z.ai infrastructure, not locally. Requires an Ollama API key.
	Model_GLM5_2_Cloud Model = common.ModelDefinition{
		Name:                 "glm-5.2:cloud",
		MaxTokens:            32_768,
		ContextWindowSize:    131_072,
		DefaultContextWindow: 32_768,
		Provider:             common.ProviderOllama,
	}

	Model_Qwen3_5_9B Model = common.ModelDefinition{
		Name:                 "qwen3.5:9b",
		MaxTokens:            32_768,
		ContextWindowSize:    262_144,
		DefaultContextWindow: 32_768,
		Provider:             common.ProviderOllama,
	}

	Model_Qwen3_5_35B Model = common.ModelDefinition{
		Name:                 "qwen3.5:35b",
		MaxTokens:            32_768,
		ContextWindowSize:    262_144,
		DefaultContextWindow: 32_768,
		Provider:             common.ProviderOllama,
	}

	Model_Qwen3_5_122B Model = common.ModelDefinition{
		Name:                 "qwen3.5:122b",
		MaxTokens:            32_768,
		ContextWindowSize:    262_144,
		DefaultContextWindow: 32_768,
		Provider:             common.ProviderOllama,
	}

	Model_Gemma4_31B Model = common.ModelDefinition{
		Name:                 "gemma4:31b",
		MaxTokens:            32_768,
		ContextWindowSize:    262_144,
		DefaultContextWindow: 32_768,
		Provider:             common.ProviderOllama,
	}
)
