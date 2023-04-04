package openai

import (
	"context"
	"fmt"
	openai "github.com/speakeasy-api/langchain-go/llms/openapi"
	"log"
	"testing"
)

// To Execute Set OPENAI_API_KEY
func TestRun(t *testing.T) {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	completion, err := llm.Call(context.Background(), "Question, what kind of bear is best?", []string{})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}
