package openai

import (
	"fmt"
	"net/http"
)

type OpenAIError struct {
	error

	statusCode int
	status     string
}

func (e *OpenAIError) Error() string {
	return fmt.Sprintf("error in call to openai with status %s", e.status)
}

func (e *OpenAIError) GetStatusCode() int {
	return e.statusCode
}

func (e *OpenAIError) IsRetryable() bool {
	return e.statusCode == http.StatusTooManyRequests || e.statusCode == http.StatusInternalServerError
}

func CreateOpenAIError(statusCode int, status string) *OpenAIError {
	err := OpenAIError{
		statusCode: statusCode,
		status:     status,
	}

	return &err
}
