package openai

import "time"

type OpenAIInput struct {
	// Model name to use
	modelName *string // TODO: Make into Enum
	// Holds any additional parameters that are valid to pass to https://platform.openai.com/docs/api-reference/completions/create
	modelKwargs map[string]interface{}
	// Batch size to use when passing multiple documents to generate
	batchSize *int
	// List of stop words to use when generating
	stop []string
	// Timeout to use when making requests to OpenAI
	timeout *time.Duration
	// OpenAI API Key
	openAIApiKey *string
	ModelParams
}

type ModelParams struct {
	// Sampling temperature to use
	temperature *float32
	// Maximum number of tokens to generate in the completion. -1 returns as many
	// tokens as possible given the prompt and the model's maximum context size.
	maxTokens *int
	// Total probability mass of tokens to consider at each step
	topP *float32
	// Penalizes repeated tokens according to frequency
	frequencyPenalty *float32
	// Penalizes repeated tokens
	presencePenalty *float32
	// Number of completions to generate for each prompt
	n *int
	// Generates `bestOf` completions server side and returns the "best"
	bestOf *int
	// Dictionary used to adjust the probability of specific tokens being generated
	logitBias map[string]int
	// Whether to stream the results or not. Enabling disables tokenUsage reporting
	streaming bool
}
