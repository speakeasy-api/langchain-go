package llms

func BatchSlice[T any](slice []T, batchSize int) [][]T {
	var chunks [][]T
	for i := 0; i < len(slice); i += batchSize {
		end := i + batchSize
		if end > len(slice) {
			end = len(slice)
		}

		chunks = append(chunks, slice[i:end])
	}

	return chunks
}

// TODO: Implement Max Token Inference
func CalculateMaxTokens(prompt string, modelName string) int {
	return 1
}
