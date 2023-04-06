package openaichat

import (
	"context"
	"errors"
	"fmt"
	langchain "github.com/speakeasy-api/langchain-go/llms/openai"
	gpt "github.com/speakeasy-sdks/openai-go-sdk"
	"github.com/speakeasy-sdks/openai-go-sdk/pkg/models/shared"
	"math"
	"net"
	"net/http"
	"os"
	"time"

	"github.com/speakeasy-api/langchain-go/llms"
)

// Default Params for Open AI model
const (
	temperature      float64 = 1
	topP             float64 = 1
	frequencyPenalty float64 = 0
	presencePenalty  float64 = 0
	n                int64   = 1
	modelName        string  = "gpt-3.5-turbo"
	maxRetries       int     = 3
)

type OpenAIChat struct {
	temperature      float64
	maxTokens        int64
	topP             float64
	frequencyPenalty float64
	presencePenalty  float64
	n                int64
	logitBias        map[string]interface{}
	streaming        bool // Streaming Unsupported Right Now
	modelName        string
	modelKwargs      map[string]interface{}
	maxRetries       int
	stop             []string
	prefixMessages   []ChatMessage
	timeout          *time.Duration
	client           *gpt.Gpt
}

func New(args ...OpenAIChatInput) (*OpenAIChat, error) {
	if len(args) > 1 {
		return nil, errors.New("more than one config argument not supported")
	}

	input := OpenAIChatInput{}
	if len(args) > 0 {
		input = args[0]
	}

	openai := OpenAIChat{
		temperature:      temperature,
		topP:             topP,
		frequencyPenalty: frequencyPenalty,
		presencePenalty:  presencePenalty,
		n:                n,
		logitBias:        input.LogitBias,
		streaming:        input.Streaming,
		modelName:        modelName,
		modelKwargs:      input.ModelKwargs,
		stop:             input.Stop,
		timeout:          input.Timeout,
		maxRetries:       maxRetries,
		prefixMessages:   input.PrefixMessages,
	}

	apiKey := os.Getenv("OPENAI_API_KEY")

	if input.OpenAIApiKey != nil {
		apiKey = *input.OpenAIApiKey
	}

	if apiKey == "" {
		return nil, errors.New("OpenAI API key not found")
	}

	if input.Temperature != nil {
		openai.temperature = *input.Temperature
	}

	if input.MaxTokens != nil {
		openai.maxTokens = *input.MaxTokens
	}

	if input.TopP != nil {
		openai.topP = *input.TopP
	}

	if input.FrequencyPenalty != nil {
		openai.frequencyPenalty = *input.FrequencyPenalty
	}

	if input.PresencePenalty != nil {
		openai.presencePenalty = *input.PresencePenalty
	}

	if input.N != nil {
		openai.n = *input.N
	}

	if input.ModelName != nil {
		openai.modelName = *input.ModelName
	}

	if input.MaxRetries != nil {
		openai.maxRetries = *input.MaxRetries
	}

	httpClient := http.Client{Transport: &langchain.AuthorizeTransport{ApiKey: apiKey}}

	if openai.timeout != nil {
		httpClient.Timeout = *openai.timeout
	}

	client := gpt.New(gpt.WithClient(&httpClient))
	openai.client = client

	return &openai, nil
}

func (openai *OpenAIChat) Name() string {
	return "openai-chat"
}

func (openai *OpenAIChat) Call(ctx context.Context, prompt string, stop []string) (string, error) {
	if len(stop) == 0 {
		stop = openai.stop
	}

	data, err := openai.chatCompletionWithRetry(ctx, prompt, openai.maxTokens, stop)
	if err != nil {
		return "", err
	}

	message := ""
	if len(data.Choices) > 0 && data.Choices[0].Message != nil {
		message = data.Choices[0].Message.Content
	}

	return message, nil
}

func (openai *OpenAIChat) Generate(ctx context.Context, prompts []string, stop []string) (*llms.LLMResult, error) {
	// Not Implemented for OpenAIChat
	return nil, nil
}

func (openai *OpenAIChat) chatCompletionWithRetry(ctx context.Context, prompt string, maxTokens int64, stop []string) (*shared.CreateChatCompletionResponse, error) {
	request := shared.CreateChatCompletionRequest{
		Model:            openai.modelName,
		Messages:         formatMessages(openai.prefixMessages, prompt),
		Temperature:      &openai.temperature,
		TopP:             &openai.topP,
		N:                &openai.n,
		LogitBias:        openai.logitBias,
		PresencePenalty:  &openai.presencePenalty,
		FrequencyPenalty: &openai.frequencyPenalty,
	}
	if openai.maxTokens != 0 {
		request.MaxTokens = &openai.maxTokens
	}

	if len(stop) != 0 {
		stopRequest := shared.CreateCreateChatCompletionRequestStopArrayOfstr(stop)
		request.Stop = &stopRequest
	}

	var finalResult *shared.CreateChatCompletionResponse
	var finalErr error

	// wait 2^x second between each retry starting with
	// max 10 seconds
	for i := 0; i < openai.maxRetries; i++ {
		lastTry := i == openai.maxRetries-1
		sleep := int(math.Min(math.Pow(2, float64(i)), float64(10)))
		res, err := openai.client.OpenAI.CreateChatCompletion(ctx, request)
		if err != nil {
			var netErr net.Error
			if errors.As(err, &netErr) {
				// retry on client timeout
				if netErr.Timeout() && !lastTry {
					time.Sleep(time.Duration(sleep) * time.Second)
					continue
				}
			}

			return nil, err
		}

		if res.StatusCode == http.StatusOK {
			finalResult = res.CreateChatCompletionResponse
			break
		} else {
			if lastTry || !langchain.StatusCodeIsRetryable(res.StatusCode) {
				// TODO: Improve Error Parsing
				finalErr = errors.New(fmt.Sprintf("error in call to openai with status %s", res.RawResponse.Status))
				break
			}
		}

		time.Sleep(time.Duration(sleep) * time.Second)
	}

	return finalResult, finalErr
}

func formatMessages(previous []ChatMessage, message string) []shared.ChatCompletionRequestMessage {
	var result []shared.ChatCompletionRequestMessage
	for _, message := range previous {
		result = append(result, shared.ChatCompletionRequestMessage{
			Content: message.Content,
			Role:    convertRoleEnum(message.Role),
		})
	}
	result = append(result, shared.ChatCompletionRequestMessage{
		Content: message,
		Role:    shared.ChatCompletionRequestMessageRoleEnumUser,
	})
	return result
}

func convertRoleEnum(enum ChatMessageRoleEnum) shared.ChatCompletionRequestMessageRoleEnum {
	switch enum {
	case ChatMessageRoleEnumSystem:
		return shared.ChatCompletionRequestMessageRoleEnumSystem
	case ChatMessageRoleEnumUser:
		return shared.ChatCompletionRequestMessageRoleEnumUser
	case ChatMessageRoleEnumAssistant:
		return shared.ChatCompletionRequestMessageRoleEnumAssistant
	default:
		return ""
	}
}
