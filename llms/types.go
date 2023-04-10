package llms

type LLMResult struct {
	Generations [][]Generation
	LLMOutput   map[string]interface{}
}

type Generation struct {
	Text           string
	GenerationInfo map[string]interface{}
}
