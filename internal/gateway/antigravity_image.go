package gateway

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/yetone/magpie/internal/provider"
)

// validateImageURL validates that u is a non-empty string.
// For data:image/ URLs, it validates raw payload structure and verifies base64 data decodes cleanly
// without trimming the raw data segment, ensuring fidelity with downstream imagePart parsing.
// Non-data URLs are kept for existing fileData passthrough without remote fetching or connectivity checks.
func validateImageURL(u string, path string) error {
	if strings.TrimSpace(u) == "" {
		return fmt.Errorf("%s cannot be empty", path)
	}
	if !strings.HasPrefix(u, "data:") {
		return nil
	}
	meta, data, ok := strings.Cut(strings.TrimPrefix(u, "data:"), ",")
	if !ok {
		return fmt.Errorf("%s invalid data URL: missing comma", path)
	}
	if !strings.HasPrefix(meta, "image/") || strings.TrimPrefix(strings.TrimSuffix(meta, ";base64"), "image/") == "" {
		return fmt.Errorf("%s invalid data URL: must have image/* MIME", path)
	}
	if !strings.HasSuffix(meta, ";base64") {
		return fmt.Errorf("%s invalid data URL: must specify ;base64", path)
	}
	if len(data) == 0 {
		return fmt.Errorf("%s invalid data URL: empty data", path)
	}
	decoded, err := base64.StdEncoding.DecodeString(data)
	if err != nil || len(decoded) == 0 {
		return fmt.Errorf("%s invalid data URL: bad or empty base64 payload", path)
	}
	return nil
}

// validateMessagesImage validates the source object of an Anthropic Messages image block.
// It decodes fields on-demand via RawMessage to pinpoint exact leaf field paths on type errors.
func validateMessagesImage(sourceRaw json.RawMessage, path string) error {
	if len(sourceRaw) == 0 {
		return fmt.Errorf("%s is required", path)
	}
	trimmed := bytes.TrimSpace(sourceRaw)
	if bytes.Equal(trimmed, []byte("null")) {
		return fmt.Errorf("%s cannot be null", path)
	}
	var srcMap map[string]json.RawMessage
	if err := json.Unmarshal(trimmed, &srcMap); err != nil {
		return fmt.Errorf("%s must be an object", path)
	}
	typeRaw, hasType := srcMap["type"]
	typePath := path + ".type"
	if !hasType {
		return fmt.Errorf("%s is required", typePath)
	}
	var typ string
	if err := json.Unmarshal(typeRaw, &typ); err != nil {
		return fmt.Errorf("%s must be a string", typePath)
	}
	if typ != "base64" && typ != "url" {
		return fmt.Errorf("%s must be 'base64' or 'url'", typePath)
	}
	if typ == "base64" {
		mediaTypePath := path + ".media_type"
		mtRaw, hasMT := srcMap["media_type"]
		if !hasMT {
			return fmt.Errorf("%s is required", mediaTypePath)
		}
		var mt string
		if err := json.Unmarshal(mtRaw, &mt); err != nil {
			return fmt.Errorf("%s must be a string", mediaTypePath)
		}
		if !strings.HasPrefix(mt, "image/") || strings.TrimPrefix(mt, "image/") == "" {
			return fmt.Errorf("%s must have a non-empty image/* subtype", mediaTypePath)
		}
		dataPath := path + ".data"
		dataRaw, hasData := srcMap["data"]
		if !hasData {
			return fmt.Errorf("%s is required", dataPath)
		}
		var dataStr string
		if err := json.Unmarshal(dataRaw, &dataStr); err != nil {
			return fmt.Errorf("%s must be a string", dataPath)
		}
		if strings.TrimSpace(dataStr) == "" {
			return fmt.Errorf("%s cannot be empty", dataPath)
		}
		decoded, err := base64.StdEncoding.DecodeString(dataStr)
		if err != nil || len(decoded) == 0 {
			return fmt.Errorf("%s invalid or empty base64 data", dataPath)
		}
	} else if typ == "url" {
		urlPath := path + ".url"
		uRaw, hasURL := srcMap["url"]
		if !hasURL {
			return fmt.Errorf("%s is required", urlPath)
		}
		var u string
		if err := json.Unmarshal(uRaw, &u); err != nil {
			return fmt.Errorf("%s must be a string", urlPath)
		}
		if strings.TrimSpace(u) == "" {
			return fmt.Errorf("%s cannot be empty", urlPath)
		}
	}
	return nil
}

// validateAntigravityImages validates user image nodes for Antigravity target requests before parse.
// Reason: Validating user image payloads upfront prevents missing or malformed image sources from being
// silently dropped into nil parts and prevents corrupted data URLs from being converted to fileData,
// ensuring strict payload fidelity and exact error path indexing without touching assistant or tool images.
func validateAntigravityImages(from provider.Protocol, body []byte) error {
	switch from {
	case provider.Anthropic:
		var req struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			return err
		}
		for i, m := range req.Messages {
			if m.Role != "user" || len(m.Content) == 0 {
				continue
			}
			trimmed := bytes.TrimSpace(m.Content)
			if bytes.HasPrefix(trimmed, []byte("[")) {
				var blocks []struct {
					Type   string          `json:"type"`
					Source json.RawMessage `json:"source"`
				}
				if err := json.Unmarshal(trimmed, &blocks); err != nil {
					return err
				}
				for j, b := range blocks {
					if b.Type == "image" {
						path := fmt.Sprintf("messages[%d].content[%d].source", i, j)
						if err := validateMessagesImage(b.Source, path); err != nil {
							return err
						}
					}
				}
			}
		}
	case provider.Chat:
		var req struct {
			Messages []struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			return err
		}
		for i, m := range req.Messages {
			if m.Role != "user" || len(m.Content) == 0 {
				continue
			}
			trimmed := bytes.TrimSpace(m.Content)
			if bytes.HasPrefix(trimmed, []byte("[")) {
				var items []struct {
					Type     string          `json:"type"`
					ImageURL json.RawMessage `json:"image_url"`
				}
				if err := json.Unmarshal(trimmed, &items); err != nil {
					return err
				}
				for j, it := range items {
					if it.Type == "image_url" {
						path := fmt.Sprintf("messages[%d].content[%d].image_url", i, j)
						if len(it.ImageURL) == 0 {
							return fmt.Errorf("%s is required", path)
						}
						trimmedImg := bytes.TrimSpace(it.ImageURL)
						if bytes.Equal(trimmedImg, []byte("null")) {
							return fmt.Errorf("%s cannot be null", path)
						}
						var imgMap map[string]json.RawMessage
						if err := json.Unmarshal(trimmedImg, &imgMap); err != nil {
							return fmt.Errorf("%s must be an object", path)
						}
						urlRaw, hasURL := imgMap["url"]
						urlPath := path + ".url"
						if !hasURL {
							return fmt.Errorf("%s is required", urlPath)
						}
						var u string
						if err := json.Unmarshal(urlRaw, &u); err != nil {
							return fmt.Errorf("%s must be a string", urlPath)
						}
						if err := validateImageURL(u, urlPath); err != nil {
							return err
						}
					}
				}
			}
		}
	case provider.Responses:
		var req struct {
			Input json.RawMessage `json:"input"`
		}
		if err := json.Unmarshal(body, &req); err != nil {
			return err
		}
		if len(req.Input) > 0 {
			trimmed := bytes.TrimSpace(req.Input)
			if bytes.HasPrefix(trimmed, []byte("[")) {
				var items []struct {
					Type    string          `json:"type"`
					Role    string          `json:"role"`
					Content json.RawMessage `json:"content"`
				}
				if err := json.Unmarshal(trimmed, &items); err != nil {
					return err
				}
				for i, it := range items {
					// Only validate genuine user message items; skip function_call_output and other item types.
					if (it.Type != "" && it.Type != "message") || it.Role != "user" || len(it.Content) == 0 {
						continue
					}
					trimmedContent := bytes.TrimSpace(it.Content)
					if bytes.HasPrefix(trimmedContent, []byte("[")) {
						var blocks []struct {
							Type     string          `json:"type"`
							ImageURL json.RawMessage `json:"image_url"`
						}
						if err := json.Unmarshal(trimmedContent, &blocks); err != nil {
							return err
						}
						for j, b := range blocks {
							if b.Type == "input_image" {
								path := fmt.Sprintf("input[%d].content[%d].image_url", i, j)
								if len(b.ImageURL) == 0 {
									return fmt.Errorf("%s is required", path)
								}
								trimmedImg := bytes.TrimSpace(b.ImageURL)
								if bytes.Equal(trimmedImg, []byte("null")) {
									return fmt.Errorf("%s cannot be null", path)
								}
								var u string
								if err := json.Unmarshal(trimmedImg, &u); err != nil {
									return fmt.Errorf("%s must be a string", path)
								}
								if err := validateImageURL(u, path); err != nil {
									return err
								}
							}
						}
					}
				}
			}
		}
	}
	return nil
}
