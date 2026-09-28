package gateway

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03dBlockTypeValidation is the unified entrypoint for B03d scenarios.
func testB03dBlockTypeValidation(t *testing.T) {
	t.Run("Unit", testB03dBlockTypeValidationU)
	t.Run("HLayer_Valid", testB03dBlockTypeValidationHValid)
	t.Run("HLayer_Scope", testB03dBlockTypeValidationHScope)
	t.Run("HLayer_Invalid", testB03dBlockTypeValidationHInvalid)
}

// testB03dBlockTypeValidationU covers U-layer parse mapping checks for acknowledged block types across both models.
func testB03dBlockTypeValidationU(t *testing.T) {
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}

	for _, model := range models {
		// (a) Messages: acknowledged 5 block types (text, image, tool_use, tool_result, thinking)
		// Accurately assert parsed Request Kinds and key values rather than merely non-empty contents.
		t.Run(fmt.Sprintf("Messages_AcknowledgedTypes_%s", model), func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "start"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": %q}}
					]},
					{"role": "assistant", "content": [
						{"type": "thinking", "thinking": "plan", "signature": "sig_xyz"},
						{"type": "tool_use", "id": "c1", "name": "fn", "input": {"k": "v"}}
					]},
					{"role": "user", "content": [
						{"type": "tool_result", "tool_use_id": "c1", "content": "done"},
						{"type": "text", "text": "end"}
					]}
				]
			}`, model, samplePNGBase64)

			req, err := parse(provider.Anthropic, []byte(payload))
			if err != nil {
				t.Fatalf("parse Messages failed: %v", err)
			}
			if len(req.Messages) != 3 {
				t.Fatalf("expected 3 messages, got %d", len(req.Messages))
			}

			// Message 0: user [Text, Image]
			m0 := req.Messages[0]
			if len(m0.Parts) != 2 || m0.Parts[0].Kind != Text || m0.Parts[0].Text != "start" || m0.Parts[1].Kind != Image || m0.Parts[1].MediaType != "image/png" {
				t.Errorf("m0 parts mismatch: %+v", m0.Parts)
			}

			// Message 1: assistant [Thinking, ToolCall]
			m1 := req.Messages[1]
			if len(m1.Parts) != 2 || m1.Parts[0].Kind != Thinking || m1.Parts[0].Text != "plan" || m1.Parts[1].Kind != ToolCall || m1.Parts[1].Name != "fn" || m1.Parts[1].ID != "c1" {
				t.Errorf("m1 parts mismatch: %+v", m1.Parts)
			}

			// Message 2: user [ToolResult, Text]
			m2 := req.Messages[2]
			if len(m2.Parts) != 2 || m2.Parts[0].Kind != ToolResult || m2.Parts[0].CallID != "c1" || m2.Parts[0].Text != "done" || m2.Parts[1].Kind != Text || m2.Parts[1].Text != "end" {
				t.Errorf("m2 parts mismatch: %+v", m2.Parts)
			}

			// Build CodeAssist wire check
			built := buildCodeAssist(req, model, "antigravity")
			var env codeAssistWireEnvelope
			if err := json.Unmarshal(built, &env); err != nil {
				t.Fatalf("unmarshal CodeAssist wire failed: %v", err)
			}
			if len(env.Request.Contents) == 0 {
				t.Fatalf("expected non-empty wire contents")
			}
		})

		// (b) Chat: acknowledged 2 block types (text, image_url)
		t.Run(fmt.Sprintf("Chat_AcknowledgedTypes_%s", model), func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "start"},
						{"type": "image_url", "image_url": {"url": %q}},
						{"type": "text", "text": "end"}
					]}
				]
			}`, model, samplePNGDataURL)

			req, err := parse(provider.Chat, []byte(payload))
			if err != nil {
				t.Fatalf("parse Chat failed: %v", err)
			}
			if len(req.Messages) != 1 {
				t.Fatalf("expected 1 message, got %d", len(req.Messages))
			}
			m := req.Messages[0]
			if len(m.Parts) != 3 || m.Parts[0].Kind != Text || m.Parts[0].Text != "start" || m.Parts[1].Kind != Image || m.Parts[2].Kind != Text || m.Parts[2].Text != "end" {
				t.Errorf("chat parts mismatch: %+v", m.Parts)
			}
		})

		// (c) Responses: acknowledged 4 block types (input_text, output_text, text, input_image)
		t.Run(fmt.Sprintf("Responses_AcknowledgedTypes_%s", model), func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"input": [
					{"role": "user", "content": [
						{"type": "input_text", "text": "p1"},
						{"type": "text", "text": "p2"},
						{"type": "output_text", "text": "p3"},
						{"type": "input_image", "image_url": %q}
					]}
				]
			}`, model, samplePNGDataURL)

			req, err := parse(provider.Responses, []byte(payload))
			if err != nil {
				t.Fatalf("parse Responses failed: %v", err)
			}
			if len(req.Messages) != 1 {
				t.Fatalf("expected 1 message, got %d", len(req.Messages))
			}
			m := req.Messages[0]
			if len(m.Parts) != 4 || m.Parts[0].Kind != Text || m.Parts[1].Kind != Text || m.Parts[2].Kind != Text || m.Parts[3].Kind != Image {
				t.Errorf("responses parts mismatch: %+v", m.Parts)
			}
		})
	}
}
