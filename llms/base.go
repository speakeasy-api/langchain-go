package llms

import "context"

type LLM interface {
	Generate(ctx context.Context, prompts []string, stop []string) (*LLMResult, error)
	Call(ctx context.Context, prompt string, stop []string) (string, error)
	Name() string
}
