package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/yetone/magpie/internal/provider"
)

// validateBlockType checks that typeRaw contains a valid, supported string type in the allowed set.
func validateBlockType(typeRaw json.RawMessage, hasType bool, allowed map[string]bool, path string) error {
	if !hasType {
		return fmt.Errorf("%s is required", path)
	}
	trimmed := bytes.TrimSpace(typeRaw)
	if bytes.Equal(trimmed, []byte("null")) {
		return fmt.Errorf("%s cannot be null", path)
	}
	var typ string
	if err := json.Unmarshal(trimmed, &typ); err != nil {
		return fmt.Errorf("%s must be a string", path)
	}
	if strings.TrimSpace(typ) == "" {
		return fmt.Errorf("%s cannot be empty", path)
	}
	if !allowed[typ] {
		return fmt.Errorf("%s %q has no supported mapping in current Antigravity bridge", path, typ)
	}
	return nil
}

// validateAntigravityBlockTypes validates that user content array items have supported, mapped block types.
// Reason: Validating user block types prevents unmapped types (e.g. audio, document) from being silently
// dropped by parser switch statements without error, ensuring calling intent is not lost.
func validateAntigravityBlockTypes(from provider.Protocol, body []byte) error {
	switch from {
	case provider.Anthropic:
		allowed := map[string]bool{
			"text":        true,
			"image":       true,
			"tool_use":    true,
			"tool_result": true,
			"thinking":    true,
		}
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
				var blocks []map[string]json.RawMessage
				if err := json.Unmarshal(trimmed, &blocks); err != nil {
					return err
				}
				for j, b := range blocks {
					path := fmt.Sprintf("messages[%d].content[%d].type", i, j)
					tRaw, has := b["type"]
					if err := validateBlockType(tRaw, has, allowed, path); err != nil {
						return err
					}
				}
			}
		}
	case provider.Chat:
		allowed := map[string]bool{
			"text":      true,
			"image_url": true,
		}
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
				var items []map[string]json.RawMessage
				if err := json.Unmarshal(trimmed, &items); err != nil {
					return err
				}
				for j, it := range items {
					path := fmt.Sprintf("messages[%d].content[%d].type", i, j)
					tRaw, has := it["type"]
					if err := validateBlockType(tRaw, has, allowed, path); err != nil {
						return err
					}
				}
			}
		}
	case provider.Responses:
		allowed := map[string]bool{
			"input_text":  true,
			"output_text": true,
			"text":        true,
			"input_image": true,
		}
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
					if (it.Type != "" && it.Type != "message") || it.Role != "user" || len(it.Content) == 0 {
						continue
					}
					trimmedContent := bytes.TrimSpace(it.Content)
					if bytes.HasPrefix(trimmedContent, []byte("[")) {
						var blocks []map[string]json.RawMessage
						if err := json.Unmarshal(trimmedContent, &blocks); err != nil {
							return err
						}
						for j, b := range blocks {
							path := fmt.Sprintf("input[%d].content[%d].type", i, j)
							tRaw, has := b["type"]
							if err := validateBlockType(tRaw, has, allowed, path); err != nil {
								return err
							}
						}
					}
				}
			}
		}
	}
	return nil
}
