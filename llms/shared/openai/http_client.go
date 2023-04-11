package openai

import (
	"fmt"
	"net/http"
)

type authorizeTransport struct {
	ApiKey string
}

func (t *authorizeTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	req.Header.Add("Authorization", fmt.Sprintf("Bearer %s", t.ApiKey))
	return http.DefaultTransport.RoundTrip(req)
}

func OpenAIAuthenticatedClient(apiKey string) http.Client {
	return http.Client{Transport: &authorizeTransport{ApiKey: apiKey}}
}
