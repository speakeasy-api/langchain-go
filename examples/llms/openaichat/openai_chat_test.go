package openai

import (
	"context"
	"fmt"
	"github.com/speakeasy-api/langchain-go/llms/openaichat"
	"log"
	"testing"
)

// To Execute EXPORT OPENAI_API_KEY=...

func TestFirstMessageChat(t *testing.T) {
	llm, err := openaichat.New()
	if err != nil {
		log.Fatal(err)
	}
	completion, err := llm.Call(context.Background(), "Hi, how are you?", []string{})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}

func TestMultiMessageChat(t *testing.T) {
	llm, err := openaichat.New(openaichat.OpenAIChatInput{
		PrefixMessages: []openaichat.ChatMessage{
			{
				Content: "Mount Everest is the tallest mountain in the world.",
				Role:    openaichat.ChatMessageRoleEnumAssistant,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	completion, err := llm.Call(context.Background(), "How tall is it?", []string{})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}
