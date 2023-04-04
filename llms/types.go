package llms

type LLMResult struct {
	Generations [][]Generation
	LLMOutput   map[string]interface{}
}

type Generation struct {
	Text           string
	GenerationInfo map[string]interface{}
}

type BaseLLMParams struct {
	// TODO: Implement when relevant embedded cache and streaming implemented
	cache           interface{}
	concurrency     interface{}
	callbackManager interface{}
}
