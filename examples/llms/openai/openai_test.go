package openai

import (
	"context"
	"fmt"
	openai "github.com/speakeasy-api/langchain-go/llms/openai"
	"log"
	"net"
	"testing"
)

// To Execute EXPORT OPENAI_API_KEY=...

func TestBasicCompletion(t *testing.T) {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	completion, err := llm.Call(context.Background(), "Question, what kind of bear is best?", []string{})
	if err != nil {
		if err, ok := err.(net.Error); ok && err.Timeout() {
			log.Fatal(err)
		}
	}

	fmt.Println(completion)
}

func TestBasicCompletionWithStop(t *testing.T) {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	completion, err := llm.Call(context.Background(), "Question, what kind of bear is best?", []string{"bear"})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}

func TestBatchCompletion(t *testing.T) {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	completion, err := llm.Generate(context.Background(), []string{
		"Question, what kind of bear is best?",
		"How tall is mount everest?",
	}, []string{})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}
