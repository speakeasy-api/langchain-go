package openai

import (
	"context"
	"errors"
	"os"
	"time"

	goopenai "github.com/sashabaranov/go-openai"
	"github.com/speakeasy-api/langchain-go/llms"
)

// Default Params for Open AI model
const (
	temperature      float32 = 0.7
	maxTokens        int     = 256
	topP             float32 = 1
	frequencyPenalty float32 = 0
	presencePenalty  float32 = 0
	n                int     = 1
	bestOf           int     = 1
	modelName        string  = "text-davinci-003"
	batchSize        int     = 20
)

type OpenAI struct {
	apiKey           string
	temperature      float32
	maxTokens        int
	topP             float32
	frequencyPenalty float32
	presencePenalty  float32
	n                int
	bestOf           int
	logitBias        map[string]int
	streaming        bool // Streaming Unsupported Right Now
	modelName        string
	modelKwargs      map[string]interface{}
	batchSize        int
	stop             []string
	timeout          *time.Duration
	client           *goopenai.Client
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
		apiKey:           os.Getenv("OPENAI_API_KEY"),
		temperature:      temperature,
		maxTokens:        maxTokens,
		topP:             topP,
		frequencyPenalty: frequencyPenalty,
		presencePenalty:  presencePenalty,
		n:                n,
		bestOf:           bestOf,
		logitBias:        input.logitBias,
		streaming:        input.streaming,
		modelName:        modelName,
		modelKwargs:      input.modelKwargs,
		batchSize:        batchSize,
		stop:             input.stop,
		timeout:          input.timeout,
	}

	if input.openAIApiKey != nil {
		openai.apiKey = *input.openAIApiKey
	}

	if openai.apiKey == "" {
		return nil, errors.New("OpenAI API key not found")
	}

	if input.temperature != nil {
		openai.temperature = *input.temperature
	}

	if input.maxTokens != nil {
		openai.maxTokens = *input.maxTokens
	}

	if input.topP != nil {
		openai.topP = *input.topP
	}

	if input.frequencyPenalty != nil {
		openai.frequencyPenalty = *input.frequencyPenalty
	}

	if input.presencePenalty != nil {
		openai.presencePenalty = *input.presencePenalty
	}

	if input.n != nil {
		openai.n = *input.n
	}

	if input.bestOf != nil {
		openai.bestOf = *input.bestOf
	}

	if input.modelName != nil {
		openai.modelName = *input.modelName
	}

	if input.batchSize != nil {
		openai.batchSize = *input.batchSize
	}

	openai.client = goopenai.NewClient(openai.apiKey)

	return &openai, nil
}

func (openai *OpenAI) Name() string {
	return "openapi"
}

func (openai *OpenAI) Call(ctx context.Context, prompt string, stop []string) (string, error) {
	generations, err := openai.Generate(ctx, []string{prompt}, stop)
	if err != nil {
		return "", err
	}

	return generations.Generations[0][0].Text, nil
}

func (openai *OpenAI) Generate(ctx context.Context, prompts []string, stop []string) (*llms.LLMResult, error) {
	subPrompts := llms.BatchSlice[string](prompts, openai.batchSize)
	maxTokens := openai.maxTokens
	var completionTokens, promptTokens, totalTokens int
	var choices []goopenai.CompletionChoice

	if openai.maxTokens == -1 {
		if len(prompts) != 1 {
			return nil, errors.New("max_tokens set to -1 not supported for multiple inputs")
		}

		maxTokens = llms.CalculateMaxTokens(prompts[0], openai.modelName)
	}

	if len(stop) == 0 {
		stop = openai.stop
	}

	for _, prompts := range subPrompts {
		data, err := openai.completionWithRetry(ctx, prompts, maxTokens, stop)
		if err != nil {
			// TODO: Wrap Into Informative Errors
			return nil, err
		}

		choices = append(choices, data.Choices...)
		completionTokens += data.Usage.CompletionTokens
		promptTokens += data.Usage.PromptTokens
		totalTokens += data.Usage.TotalTokens
	}
	var generations [][]llms.Generation
	batchedChoices := llms.BatchSlice[goopenai.CompletionChoice](choices, openai.n)
	for _, batch := range batchedChoices {
		var generationBatch []llms.Generation
		for _, choice := range batch {
			generationBatch = append(generationBatch, llms.Generation{
				Text: choice.Text,
				GenerationInfo: map[string]interface{}{
					"finishReason": choice.FinishReason,
					"logprobs":     choice.LogProbs,
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

func (openai *OpenAI) completionWithRetry(ctx context.Context, prompts []string, maxTokens int, stop []string) (*goopenai.CompletionResponse, error) {
	request := goopenai.CompletionRequest{
		Model:            openai.modelName,
		Prompt:           prompts[0], // TODO: Implement Support for Batching on Next Release
		MaxTokens:        maxTokens,
		Temperature:      openai.temperature,
		TopP:             openai.topP,
		N:                openai.n,
		BestOf:           openai.bestOf,
		LogitBias:        openai.logitBias,
		PresencePenalty:  openai.presencePenalty,
		FrequencyPenalty: openai.frequencyPenalty,
		Stop:             stop,
	}

	if openai.timeout != nil {
		ctx, _ = context.WithTimeout(ctx, *openai.timeout)
	}

	// TODO: Implement Retries
	resp, err := openai.client.CreateCompletion(
		ctx,
		request,
	)

	return &resp, err
}
