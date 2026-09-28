package gateway

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03cImageValidation is the unified entrypoint for all B03c image validation BDD scenarios.
func testB03cImageValidation(t *testing.T) {
	t.Run("Unit", testB03cImageValidationU)
	t.Run("HLayer_Valid", testB03cImageValidationHValid)
	t.Run("HLayer_Context", testB03cImageValidationHContext)
	t.Run("HLayer_Invalid", testB03cImageValidationHInvalid)
}

// testB03cImageValidationU covers all U-layer B03c image and text ordering fidelity scenarios.
func testB03cImageValidationU(t *testing.T) {
	t.Run("AlternatingTextAndInlineImageFidelity", testB03cUAlternatingTextAndInlineImageFidelity)
	t.Run("RemoteURLPreservesFileData", testB03cURemoteURLPreservesFileData)
}

const (
	samplePNGBase64   = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	samplePNGDataURL  = "data:image/png;base64," + samplePNGBase64
	sampleJPEGBase64  = "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="
	sampleJPEGDataURL = "data:image/jpeg;base64," + sampleJPEGBase64
)

type codeAssistWirePart struct {
	Text       string `json:"text,omitempty"`
	InlineData *struct {
		MimeType string `json:"mimeType"`
		Data     string `json:"data"`
	} `json:"inlineData,omitempty"`
	FileData *struct {
		MimeType string `json:"mimeType"`
		FileURI  string `json:"fileUri"`
	} `json:"fileData,omitempty"`
}

type codeAssistWireEnvelope struct {
	Request struct {
		Contents []struct {
			Role  string               `json:"role"`
			Parts []codeAssistWirePart `json:"parts"`
		} `json:"contents"`
	} `json:"request"`
}

// 1. Alternating [text, img, text, img, text] order, type, MIME, and bytes fidelity across 3 protocols and 2 models.
func testB03cUAlternatingTextAndInlineImageFidelity(t *testing.T) {
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}

	for _, model := range models {
		// (a) Anthropic Messages: alternating content blocks
		t.Run(fmt.Sprintf("Messages_%s", model), func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "t1"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": %q}},
						{"type": "text", "text": "t2"},
						{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": %q}},
						{"type": "text", "text": "t3"}
					]}
				]
			}`, model, samplePNGBase64, sampleJPEGBase64)

			req, err := parse(provider.Anthropic, []byte(payload))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			built := buildCodeAssist(req, model, "antigravity")
			var env codeAssistWireEnvelope
			if err := json.Unmarshal(built, &env); err != nil {
				t.Fatalf("unmarshal built request failed: %v\nBody: %s", err, string(built))
			}
			if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
				t.Fatalf("expected 1 user content in built request, got: %+v", env.Request.Contents)
			}
			parts := env.Request.Contents[0].Parts
			if len(parts) != 5 {
				t.Fatalf("expected 5 alternating parts on wire, got %d", len(parts))
			}
			assertAlternatingWireParts(t, parts)
		})

		// (b) Chat: alternating content parts with data URLs
		t.Run(fmt.Sprintf("Chat_%s", model), func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"messages": [
					{"role": "user", "content": [
						{"type": "text", "text": "t1"},
						{"type": "image_url", "image_url": {"url": %q}},
						{"type": "text", "text": "t2"},
						{"type": "image_url", "image_url": {"url": %q}},
						{"type": "text", "text": "t3"}
					]}
				]
			}`, model, samplePNGDataURL, sampleJPEGDataURL)

			req, err := parse(provider.Chat, []byte(payload))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			built := buildCodeAssist(req, model, "antigravity")
			var env codeAssistWireEnvelope
			if err := json.Unmarshal(built, &env); err != nil {
				t.Fatalf("unmarshal built request failed: %v\nBody: %s", err, string(built))
			}
			if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
				t.Fatalf("expected 1 user content in built request, got: %+v", env.Request.Contents)
			}
			parts := env.Request.Contents[0].Parts
			if len(parts) != 5 {
				t.Fatalf("expected 5 alternating parts on wire, got %d", len(parts))
			}
			assertAlternatingWireParts(t, parts)
		})

		// (c) Responses: user message with alternating content parts
		t.Run(fmt.Sprintf("Responses_%s", model), func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"input": [
					{"role": "user", "content": [
						{"type": "input_text", "text": "t1"},
						{"type": "input_image", "image_url": %q},
						{"type": "input_text", "text": "t2"},
						{"type": "input_image", "image_url": %q},
						{"type": "input_text", "text": "t3"}
					]}
				]
			}`, model, samplePNGDataURL, sampleJPEGDataURL)

			req, err := parse(provider.Responses, []byte(payload))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			built := buildCodeAssist(req, model, "antigravity")
			var env codeAssistWireEnvelope
			if err := json.Unmarshal(built, &env); err != nil {
				t.Fatalf("unmarshal built request failed: %v\nBody: %s", err, string(built))
			}
			if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
				t.Fatalf("expected 1 user content in built request, got: %+v", env.Request.Contents)
			}
			parts := env.Request.Contents[0].Parts
			if len(parts) != 5 {
				t.Fatalf("expected 5 alternating parts on wire, got %d", len(parts))
			}
			assertAlternatingWireParts(t, parts)
		})
	}
}

// assertAlternatingWireParts validates [text, img, text, img, text] types, exclusivity, and exact payloads.
func assertAlternatingWireParts(t *testing.T, parts []codeAssistWirePart) {
	t.Helper()
	// Part 0: Text "t1"
	if parts[0].Text != "t1" || parts[0].InlineData != nil || parts[0].FileData != nil {
		t.Errorf("part 0 mismatch: text=%q inlineData=%v fileData=%v", parts[0].Text, parts[0].InlineData, parts[0].FileData)
	}

	// Part 1: PNG Image (type exclusivity: Text must be empty, FileData must be nil)
	if parts[1].Text != "" || parts[1].FileData != nil || parts[1].InlineData == nil {
		t.Fatalf("part 1 type exclusivity violated: text=%q fileData=%v inlineData=%v", parts[1].Text, parts[1].FileData, parts[1].InlineData)
	}
	if parts[1].InlineData.MimeType != "image/png" {
		t.Errorf("part 1 mimeType mismatch: got %q, want image/png", parts[1].InlineData.MimeType)
	}
	if parts[1].InlineData.Data != samplePNGBase64 {
		t.Errorf("part 1 raw base64 string mismatch: got %q, want %q", parts[1].InlineData.Data, samplePNGBase64)
	}
	rawPNG, err1 := base64.StdEncoding.DecodeString(parts[1].InlineData.Data)
	expectedPNG, err2 := base64.StdEncoding.DecodeString(samplePNGBase64)
	if err1 != nil || err2 != nil || !bytes.Equal(rawPNG, expectedPNG) {
		t.Errorf("part 1 data byte mismatch (err1=%v, err2=%v)", err1, err2)
	}

	// Part 2: Text "t2"
	if parts[2].Text != "t2" || parts[2].InlineData != nil || parts[2].FileData != nil {
		t.Errorf("part 2 mismatch: text=%q inlineData=%v fileData=%v", parts[2].Text, parts[2].InlineData, parts[2].FileData)
	}

	// Part 3: JPEG Image (type exclusivity: Text must be empty, FileData must be nil)
	if parts[3].Text != "" || parts[3].FileData != nil || parts[3].InlineData == nil {
		t.Fatalf("part 3 type exclusivity violated: text=%q fileData=%v inlineData=%v", parts[3].Text, parts[3].FileData, parts[3].InlineData)
	}
	if parts[3].InlineData.MimeType != "image/jpeg" {
		t.Errorf("part 3 mimeType mismatch: got %q, want image/jpeg", parts[3].InlineData.MimeType)
	}
	if parts[3].InlineData.Data != sampleJPEGBase64 {
		t.Errorf("part 3 raw base64 string mismatch: got %q, want %q", parts[3].InlineData.Data, sampleJPEGBase64)
	}
	rawJPEG, err3 := base64.StdEncoding.DecodeString(parts[3].InlineData.Data)
	expectedJPEG, err4 := base64.StdEncoding.DecodeString(sampleJPEGBase64)
	if err3 != nil || err4 != nil || !bytes.Equal(rawJPEG, expectedJPEG) {
		t.Errorf("part 3 data byte mismatch (err3=%v, err4=%v)", err3, err4)
	}

	// Part 4: Text "t3"
	if parts[4].Text != "t3" || parts[4].InlineData != nil || parts[4].FileData != nil {
		t.Errorf("part 4 mismatch: text=%q inlineData=%v fileData=%v", parts[4].Text, parts[4].InlineData, parts[4].FileData)
	}
}

// 2. Non-empty ordinary remote URL preserves existing fileData behavior without network fetch.
func testB03cURemoteURLPreservesFileData(t *testing.T) {
	const remoteURL = "https://example.com/photo.png"
	models := []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}

	for _, model := range models {
		t.Run(model, func(t *testing.T) {
			payload := fmt.Sprintf(`{
				"model": %q,
				"max_tokens": 1024,
				"messages": [
					{"role": "user", "content": [
						{"type": "image", "source": {"type": "url", "url": %q}}
					]}
				]
			}`, model, remoteURL)

			req, err := parse(provider.Anthropic, []byte(payload))
			if err != nil {
				t.Fatalf("parse failed: %v", err)
			}
			built := buildCodeAssist(req, model, "antigravity")
			var env codeAssistWireEnvelope
			if err := json.Unmarshal(built, &env); err != nil {
				t.Fatalf("unmarshal failed: %v", err)
			}
			if len(env.Request.Contents) != 1 || env.Request.Contents[0].Role != "user" {
				t.Fatalf("expected 1 user content in built request, got: %+v", env.Request.Contents)
			}
			if len(env.Request.Contents[0].Parts) != 1 {
				t.Fatalf("expected exactly 1 part on wire, got %d", len(env.Request.Contents[0].Parts))
			}
			part := env.Request.Contents[0].Parts[0]
			if part.Text != "" || part.InlineData != nil || part.FileData == nil {
				t.Fatalf("expected exclusive fileData for remote url, got: %+v", part)
			}
			if part.FileData.FileURI != remoteURL {
				t.Errorf("fileUri mismatch: got %q, want %q", part.FileData.FileURI, remoteURL)
			}
			if part.FileData.MimeType != "" {
				t.Errorf("mimeType mismatch: got %q, want empty", part.FileData.MimeType)
			}
		})
	}
}
