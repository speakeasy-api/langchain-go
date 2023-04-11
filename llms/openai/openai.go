package openai

import (
	"context"
	"errors"
	llms_shared "github.com/speakeasy-api/langchain-go/llms/shared"

	openai_shared "github.com/speakeasy-api/langchain-go/llms/shared/openai"
	gpt "github.com/speakeasy-sdks/openai-go-sdk"
	"github.com/speakeasy-sdks/openai-go-sdk/pkg/models/shared"
	"math"
	"net"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/speakeasy-api/langchain-go/llms"
)

// Default Params for Open AI model
const (
	temperature      float64 = 0.7
	maxTokens        int64   = 256
	topP             float64 = 1
	frequencyPenalty float64 = 0
	presencePenalty  float64 = 0
	n                int64   = 1
	bestOf           int64   = 1
	modelName        string  = "text-davinci-003"
	batchSize        int64   = 20
	maxRetries       int     = 3
)

type OpenAI struct {
	temperature      float64
	maxTokens        int64
	topP             float64
	frequencyPenalty float64
	presencePenalty  float64
	n                int64
	bestOf           int64
	logitBias        map[string]interface{}
	streaming        bool // Streaming Unsupported Right Now
	modelName        string
	modelKwargs      map[string]interface{}
	maxRetries       int
	batchSize        int64
	stop             []string
	timeout          *time.Duration
	client           *gpt.Gpt
}

func New(args ...OpenAIInput) (*OpenAI, error) {
	if len(args) > 1 {
		return nil, errors.New("more than one config argument not supported")
	}

	input := OpenAIInput{}
	if len(args) > 0 {
		input = args[0]
	}

	openai := OpenAI{
		temperature:      temperature,
		maxTokens:        maxTokens,
		topP:             topP,
		frequencyPenalty: frequencyPenalty,
		presencePenalty:  presencePenalty,
		n:                n,
		bestOf:           bestOf,
		logitBias:        input.LogitBias,
		streaming:        input.Streaming,
		modelName:        modelName,
		modelKwargs:      input.ModelKwargs,
		batchSize:        batchSize,
		stop:             input.Stop,
		timeout:          input.Timeout,
		maxRetries:       maxRetries,
	}

	apiKey := os.Getenv("OPENAI_API_KEY")

	if input.OpenAIApiKey != nil {
		apiKey = *input.OpenAIApiKey
	}

	if apiKey == "" {
		return nil, errors.New("OpenAI API key not found")
	}

	if input.ModelName != nil {
		openai.modelName = *input.ModelName
	}

	if strings.HasPrefix(openai.modelName, "gpt-3.5-turbo") || strings.HasPrefix(openai.modelName, "gpt-4") {
		return nil, errors.New("use OpenAIChat for these models")
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

	if input.BestOf != nil {
		openai.bestOf = *input.BestOf
	}

	if input.BatchSize != nil {
		openai.batchSize = *input.BatchSize
	}

	if input.MaxRetries != nil {
		openai.maxRetries = *input.MaxRetries
	}

	httpClient := openai_shared.OpenAIAuthenticatedClient(apiKey)

	if openai.timeout != nil {
		httpClient.Timeout = *openai.timeout
	}

	client := gpt.New(gpt.WithClient(&httpClient))
	openai.client = client

	return &openai, nil
}

func (openai *OpenAI) Name() string {
	return "openai"
}

func (openai *OpenAI) Call(ctx context.Context, prompt string, stop []string) (string, error) {
	generations, err := openai.Generate(ctx, []string{prompt}, stop)
	if err != nil {
		return "", err
	}

	return generations.Generations[0][0].Text, nil
}

func (openai *OpenAI) Generate(ctx context.Context, prompts []string, stop []string) (*llms.LLMResult, error) {
	subPrompts := llms_shared.BatchSlice[string](prompts, openai.batchSize)
	maxTokens := openai.maxTokens
	var completionTokens, promptTokens, totalTokens int64
	var choices []shared.CreateCompletionResponseChoices

	if openai.maxTokens == -1 {
		if len(prompts) != 1 {
			return nil, errors.New("max_tokens set to -1 not supported for multiple inputs")
		}

		maxTokens = llms_shared.CalculateMaxTokens(prompts[0], openai.modelName)
	}

	if len(stop) == 0 {
		stop = openai.stop
	}

	for _, prompts := range subPrompts {
		data, err := openai.completionWithRetry(ctx, prompts, maxTokens, stop)
		if err != nil {
			return nil, err
		}

		choices = append(choices, data.Choices...)
		if data.Usage != nil {
			completionTokens += data.Usage.CompletionTokens
			promptTokens += data.Usage.PromptTokens
			totalTokens += data.Usage.TotalTokens
		}
	}
	var generations [][]llms.Generation
	batchedChoices := llms_shared.BatchSlice[shared.CreateCompletionResponseChoices](choices, openai.n)
	for _, batch := range batchedChoices {
		var generationBatch []llms.Generation
		for _, choice := range batch {
			generationBatch = append(generationBatch, llms.Generation{
				Text: *choice.Text,
				GenerationInfo: map[string]interface{}{
					"finishReason": choice.FinishReason,
					"logprobs":     choice.Logprobs,
				},
			})
		}
		generations = append(generations, generationBatch)
	}

	return &llms.LLMResult{
		Generations: generations,
		LLMOutput: map[string]interface{}{
			"completionTokens": completionTokens,
			"promptTokens":     promptTokens,
			"totalTokens":      totalTokens,
		},
	}, nil
}

func (openai *OpenAI) completionWithRetry(ctx context.Context, prompts []string, maxTokens int64, stop []string) (*shared.CreateCompletionResponse, error) {
	promptRequest := shared.CreateCreateCompletionRequestPromptArrayOfstr(prompts)
	request := shared.CreateCompletionRequest{
		Model:            openai.modelName,
		Prompt:           &promptRequest,
		MaxTokens:        &maxTokens,
		Temperature:      &openai.temperature,
		TopP:             &openai.topP,
		N:                &openai.n,
		BestOf:           &openai.bestOf,
		LogitBias:        openai.logitBias,
		PresencePenalty:  &openai.presencePenalty,
		FrequencyPenalty: &openai.frequencyPenalty,
	}
	if len(stop) != 0 {
		stopRequest := shared.CreateCreateCompletionRequestStopArrayOfstr(stop)
		request.Stop = &stopRequest
	}

	var finalResult *shared.CreateCompletionResponse
	var finalErr error

	// wait 2^x second between each retry starting with
	// max 10 seconds
	for i := 0; i < openai.maxRetries; i++ {
		lastTry := i == openai.maxRetries-1
		sleep := int(math.Min(math.Pow(2, float64(i)), float64(10)))
		res, err := openai.client.OpenAI.CreateCompletion(ctx, request)
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
			finalResult = res.CreateCompletionResponse
			break
		} else {
			openAIError := openai_shared.CreateOpenAIError(res.StatusCode, res.RawResponse.Status)
			if lastTry || !openAIError.IsRetryable() {
				finalErr = openAIError
				break
			}
		}

		time.Sleep(time.Duration(sleep) * time.Second)
	}

	return finalResult, finalErr
}
