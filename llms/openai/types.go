package openai

import "time"

type OpenAIInput struct {
	// Model name to use
	ModelName *string // TODO: Make into Enum
	// Holds any additional parameters that are valid to pass to https://platform.openai.com/docs/api-reference/completions/create
	ModelKwargs map[string]interface{}
	// Batch size to use when passing multiple documents to generate
	BatchSize *int64
	// List of stop words to use when generating
	Stop []string
	// Timeout to use when making a http request to OpenAI
	Timeout *time.Duration
	// Number of retry attempts for a single request to OpenAI
	MaxRetries *int
	// OpenAI API Key
	OpenAIApiKey *string
	ModelParams
}

type ModelParams struct {
	// Sampling temperature to use
	Temperature *float64
	// Maximum number of tokens to generate in the completion. -1 returns as many
	// tokens as possible given the prompt and the model's maximum context size.
	MaxTokens *int64
	// Total probability mass of tokens to consider at each step
	TopP *float64
	// Penalizes repeated tokens according to frequency
	FrequencyPenalty *float64
	// Penalizes repeated tokens
	PresencePenalty *float64
	// Number of completions to generate for each prompt
	N *int64
	// Generates `bestOf` completions server side and returns the "best"
	BestOf *int64
	// Dictionary used to adjust the probability of specific tokens being generated
	LogitBias map[string]interface{}
	// Whether to stream the results or not. Enabling disables tokenUsage reporting
	Streaming bool
}
