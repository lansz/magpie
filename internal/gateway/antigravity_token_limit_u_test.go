package gateway

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03OutputTokenLimit is the unified entrypoint for B03a scenarios.
func testB03OutputTokenLimit(t *testing.T) {
	t.Run("Unit", testB03OutputTokenLimitU)
	t.Run("HLayer", testB03OutputTokenLimitH)
	t.Run("NonTarget", testB03OutputTokenLimitNonTarget)
}

// testB03OutputTokenLimitU covers all U-layer B03a scenarios.
func testB03OutputTokenLimitU(t *testing.T) {
	t.Run("ExplicitTokenLimitPreserved", testB03UExplicitTokenLimitPreserved)
	t.Run("AbsentTokenLimitNotDefaulted", testB03UAbsentTokenLimitNotDefaulted)
	t.Run("ChatDualFieldPriority", testB03UChatDualFieldPriority)
	t.Run("NonTargetGeminiCLIUnchanged", testB03UNonTargetGeminiCLIUnchanged)
}

type typedGenConfigEnv struct {
	Request struct {
		GenerationConfig struct {
			MaxOutputTokens *int `json:"maxOutputTokens"`
		} `json:"generationConfig"`
	} `json:"request"`
}

type rawGenConfigEnv struct {
	Request struct {
		GenerationConfig map[string]json.RawMessage `json:"generationConfig"`
	} `json:"request"`
}

// 1. Explicit token limits across three protocols and two target models.
func testB03UExplicitTokenLimitPreserved(t *testing.T) {
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}
	limits := []int{128, 2048}

	for _, model := range models {
		for _, limit := range limits {
			// (a) Messages: max_tokens
			t.Run(fmt.Sprintf("Messages_%s_%d", model, limit), func(t *testing.T) {
				body := fmt.Sprintf(`{"model":%q,"max_tokens":%d,"messages":[{"role":"user","content":"hi"}]}`, model, limit)
				req, err := parse(provider.Anthropic, []byte(body))
				if err != nil {
					t.Fatalf("parse failed: %v", err)
				}
				built := buildCodeAssist(req, model, "antigravity")
				var typed typedGenConfigEnv
				var raw rawGenConfigEnv
				if err := json.Unmarshal(built, &typed); err != nil {
					t.Fatalf("unmarshal typed failed: %v", err)
				}
				if err := json.Unmarshal(built, &raw); err != nil {
					t.Fatalf("unmarshal raw failed: %v", err)
				}
				rawVal, exists := raw.Request.GenerationConfig["maxOutputTokens"]
				if !exists || string(rawVal) == "null" || typed.Request.GenerationConfig.MaxOutputTokens == nil {
					t.Fatalf("maxOutputTokens absent or null for %s limit %d", model, limit)
				}
				if *typed.Request.GenerationConfig.MaxOutputTokens != limit {
					t.Errorf("maxOutputTokens mismatch: got %d, want %d", *typed.Request.GenerationConfig.MaxOutputTokens, limit)
				}
			})

			// (b) Chat: max_completion_tokens
			t.Run(fmt.Sprintf("Chat_MaxCompletionTokens_%s_%d", model, limit), func(t *testing.T) {
				body := fmt.Sprintf(`{"model":%q,"max_completion_tokens":%d,"messages":[{"role":"user","content":"hi"}]}`, model, limit)
				req, err := parse(provider.Chat, []byte(body))
				if err != nil {
					t.Fatalf("parse failed: %v", err)
				}
				built := buildCodeAssist(req, model, "antigravity")
				var typed typedGenConfigEnv
				var raw rawGenConfigEnv
				if err := json.Unmarshal(built, &typed); err != nil {
					t.Fatalf("unmarshal typed failed: %v", err)
				}
				if err := json.Unmarshal(built, &raw); err != nil {
					t.Fatalf("unmarshal raw failed: %v", err)
				}
				rawVal, exists := raw.Request.GenerationConfig["maxOutputTokens"]
				if !exists || string(rawVal) == "null" || typed.Request.GenerationConfig.MaxOutputTokens == nil {
					t.Fatalf("maxOutputTokens absent or null for %s limit %d", model, limit)
				}
				if *typed.Request.GenerationConfig.MaxOutputTokens != limit {
					t.Errorf("maxOutputTokens mismatch: got %d, want %d", *typed.Request.GenerationConfig.MaxOutputTokens, limit)
				}
			})

			// (c) Chat: single max_tokens field
			t.Run(fmt.Sprintf("Chat_MaxTokens_%s_%d", model, limit), func(t *testing.T) {
				body := fmt.Sprintf(`{"model":%q,"max_tokens":%d,"messages":[{"role":"user","content":"hi"}]}`, model, limit)
				req, err := parse(provider.Chat, []byte(body))
				if err != nil {
					t.Fatalf("parse failed: %v", err)
				}
				built := buildCodeAssist(req, model, "antigravity")
				var typed typedGenConfigEnv
				var raw rawGenConfigEnv
				if err := json.Unmarshal(built, &typed); err != nil {
					t.Fatalf("unmarshal typed failed: %v", err)
				}
				if err := json.Unmarshal(built, &raw); err != nil {
					t.Fatalf("unmarshal raw failed: %v", err)
				}
				rawVal, exists := raw.Request.GenerationConfig["maxOutputTokens"]
				if !exists || string(rawVal) == "null" || typed.Request.GenerationConfig.MaxOutputTokens == nil {
					t.Fatalf("maxOutputTokens absent or null for %s limit %d", model, limit)
				}
				if *typed.Request.GenerationConfig.MaxOutputTokens != limit {
					t.Errorf("maxOutputTokens mismatch: got %d, want %d", *typed.Request.GenerationConfig.MaxOutputTokens, limit)
				}
			})

			// (d) Responses: max_output_tokens
			t.Run(fmt.Sprintf("Responses_%s_%d", model, limit), func(t *testing.T) {
				body := fmt.Sprintf(`{"model":%q,"max_output_tokens":%d,"input":[{"role":"user","content":"hi"}]}`, model, limit)
				req, err := parse(provider.Responses, []byte(body))
				if err != nil {
					t.Fatalf("parse failed: %v", err)
				}
				built := buildCodeAssist(req, model, "antigravity")
				var typed typedGenConfigEnv
				var raw rawGenConfigEnv
				if err := json.Unmarshal(built, &typed); err != nil {
					t.Fatalf("unmarshal typed failed: %v", err)
				}
				if err := json.Unmarshal(built, &raw); err != nil {
					t.Fatalf("unmarshal raw failed: %v", err)
				}
				rawVal, exists := raw.Request.GenerationConfig["maxOutputTokens"]
				if !exists || string(rawVal) == "null" || typed.Request.GenerationConfig.MaxOutputTokens == nil {
					t.Fatalf("maxOutputTokens absent or null for %s limit %d", model, limit)
				}
				if *typed.Request.GenerationConfig.MaxOutputTokens != limit {
					t.Errorf("maxOutputTokens mismatch: got %d, want %d", *typed.Request.GenerationConfig.MaxOutputTokens, limit)
				}
			})
		}
	}
}

// 2. Chat dual field priority: max_completion_tokens wins over max_tokens.
func testB03UChatDualFieldPriority(t *testing.T) {
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":%q,"max_completion_tokens":128,"max_tokens":2048,"messages":[{"role":"user","content":"hi"}]}`, model)
			req, err := parse(provider.Chat, []byte(body))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			built := buildCodeAssist(req, model, "antigravity")
			var typed typedGenConfigEnv
			if err := json.Unmarshal(built, &typed); err != nil {
				t.Fatalf("unmarshal typed failed: %v", err)
			}
			if typed.Request.GenerationConfig.MaxOutputTokens == nil {
				t.Fatalf("maxOutputTokens missing from generationConfig for %s dual fields", model)
			}
			if *typed.Request.GenerationConfig.MaxOutputTokens != 128 {
				t.Errorf("maxOutputTokens priority mismatch: got %d, want 128", *typed.Request.GenerationConfig.MaxOutputTokens)
			}
		})
	}
}

// 3. Absent token limits must truly be absent from wire JSON (not defaulted, not null).
func testB03UAbsentTokenLimitNotDefaulted(t *testing.T) {
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}
	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"hi"}]}`, model)
			req, err := parse(provider.Anthropic, []byte(body))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			built := buildCodeAssist(req, model, "antigravity")
			var raw rawGenConfigEnv
			if err := json.Unmarshal(built, &raw); err != nil {
				t.Fatalf("unmarshal raw failed: %v", err)
			}
			if raw.Request.GenerationConfig != nil {
				if _, exists := raw.Request.GenerationConfig["maxOutputTokens"]; exists {
					t.Errorf("expected maxOutputTokens to be completely absent from generationConfig for %s, but key exists", model)
				}
			}
		})
	}
}

// 4. Non-target Gemini CLI behavior unchanged: positive and absent.
func testB03UNonTargetGeminiCLIUnchanged(t *testing.T) {
	t.Run("PositivePreserved", func(t *testing.T) {
		req := &Request{
			Model:     "gemini-2.5-pro",
			MaxTokens: 2000,
			Messages:  []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}},
		}
		built := buildCodeAssist(req, "gemini-2.5-pro", "gemini")
		var typed typedGenConfigEnv
		if err := json.Unmarshal(built, &typed); err != nil {
			t.Fatalf("unmarshal typed failed: %v", err)
		}
		if typed.Request.GenerationConfig.MaxOutputTokens == nil || *typed.Request.GenerationConfig.MaxOutputTokens != 2000 {
			t.Errorf("gemini CLI positive limit mismatch: got %v, want 2000", typed.Request.GenerationConfig.MaxOutputTokens)
		}
	})

	t.Run("AbsentUnchanged", func(t *testing.T) {
		req := &Request{
			Model:    "gemini-2.5-pro",
			Messages: []Message{{Role: "user", Parts: []Part{{Kind: Text, Text: "hi"}}}},
		}
		built := buildCodeAssist(req, "gemini-2.5-pro", "gemini")
		var raw rawGenConfigEnv
		if err := json.Unmarshal(built, &raw); err != nil {
			t.Fatalf("unmarshal raw failed: %v", err)
		}
		if raw.Request.GenerationConfig != nil {
			if _, exists := raw.Request.GenerationConfig["maxOutputTokens"]; exists {
				t.Errorf("expected absent maxOutputTokens for gemini CLI absent request, but key exists")
			}
		}
	})
}
