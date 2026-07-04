package openai_compat

// Converters between the provider-agnostic request/response types and the
// OpenAI SDK's wire types, shared by every OpenAI-compatible provider.

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/packages/param"
	"github.com/openai/openai-go/v3/shared"
	"github.com/tab58/llm-providers/providers/common"
)

func toOpenAIMessages(msgs []common.Message) []openai.ChatCompletionMessageParamUnion {
	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(msgs))
	for _, msg := range msgs {
		switch msg.Role {
		case common.RoleUser:
			result = append(result, openai.UserMessage(common.CombinedText(msg.Content)))
		case common.RoleAssistant:
			result = append(result, toOpenAIAssistantMessage(msg))
		case common.RoleSystem:
			result = append(result, openai.SystemMessage(common.CombinedText(msg.Content)))
		case common.RoleTool:
			for _, block := range msg.Content {
				if block.Type == common.ContentTypeToolResult {
					result = append(result, openai.ToolMessage(block.ToolOutput, block.ToolResultID))
				}
			}
		}
	}
	return result
}

func toOpenAIAssistantMessage(msg common.Message) openai.ChatCompletionMessageParamUnion {
	assistant := openai.ChatCompletionAssistantMessageParam{}

	if text := common.CombinedText(msg.Content); text != "" {
		assistant.Content.OfString = param.NewOpt(text)
	}

	for _, block := range msg.Content {
		if block.Type != common.ContentTypeToolUse {
			continue
		}
		assistant.ToolCalls = append(assistant.ToolCalls, openai.ChatCompletionMessageToolCallUnionParam{
			OfFunction: &openai.ChatCompletionMessageFunctionToolCallParam{
				ID: block.ToolUseID,
				Function: openai.ChatCompletionMessageFunctionToolCallFunctionParam{
					Name:      block.ToolName,
					Arguments: string(block.ToolInput),
				},
			},
		})
	}

	return openai.ChatCompletionMessageParamUnion{OfAssistant: &assistant}
}

func toOpenAITools(tools []common.ToolDefinition) ([]openai.ChatCompletionToolUnionParam, error) {
	result := make([]openai.ChatCompletionToolUnionParam, 0, len(tools))
	for _, tool := range tools {
		var params shared.FunctionParameters
		if tool.InputSchema != nil {
			if err := json.Unmarshal(tool.InputSchema, &params); err != nil {
				return nil, fmt.Errorf("tool %q: parse input schema: %w", tool.Name, err)
			}
		}

		result = append(result, openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{
			Name:        tool.Name,
			Description: param.NewOpt(tool.Description),
			Parameters:  params,
		}))
	}
	return result, nil
}

func fromOpenAIResponse(res *openai.ChatCompletion) common.CompletionResponse {
	var content []common.ContentBlock
	var stopReason common.StopReason

	if len(res.Choices) > 0 {
		choice := res.Choices[0]
		stopReason = fromOpenAIFinishReason(choice.FinishReason)

		if choice.Message.Content != "" {
			content = append(content, common.NewTextContent(choice.Message.Content))
		}

		for _, tc := range choice.Message.ToolCalls {
			if tc.Type == "function" {
				content = append(content, common.NewToolUseContent(
					tc.ID,
					tc.Function.Name,
					json.RawMessage(tc.Function.Arguments),
				))
			}
		}
	}

	return common.CompletionResponse{
		ID:         res.ID,
		Content:    content,
		StopReason: stopReason,
		Usage: common.Usage{
			InputTokens:  res.Usage.PromptTokens,
			OutputTokens: res.Usage.CompletionTokens,
		},
		Model: res.Model,
	}
}

func fromOpenAIFinishReason(reason string) common.StopReason {
	switch reason {
	case "stop":
		return common.StopReasonStop
	case "length":
		return common.StopReasonMaxTokens
	case "tool_calls":
		return common.StopReasonToolUse
	default:
		return common.StopReason(reason)
	}
}
