package shared

func BatchSlice[T any](slice []T, batchSize int64) [][]T {
	var chunks [][]T
	for i := int64(0); i < int64(len(slice)); i += batchSize {
		end := i + batchSize
		if end > int64(len(slice)) {
			end = int64(len(slice))
		}

		chunks = append(chunks, slice[i:end])
	}

	return chunks
}

// TODO: Implement Max Token Inference
func CalculateMaxTokens(prompt string, modelName string) int64 {
	return 1
}
