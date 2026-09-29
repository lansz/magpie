package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB05EnvelopeProfile covers B05: The final upstream request strictly matches
// the selected Antigravity account, project, model, and profile; non-Antigravity
// target paths are not rewritten with Antigravity-specific envelope fields;
// downstream provider must not overwrite fields already established by adapters;
// and distinct accounts under the same provider ID maintain isolated signing scopes.
func testB05EnvelopeProfile(t *testing.T) {
	t.Run("ThroughputMatchesAccountAndProfile", testB05ThroughputMatchesAccountAndProfile)
	t.Run("NonTargetIsolation", testB05NonTargetIsolation)
	t.Run("ProviderDoesNotOverwritePredefinedFields", testB05ProviderDoesNotOverwritePredefinedFields)
	t.Run("ScopeIsolationSameProviderID", testB05ScopeIsolationSameProviderID)
}

type fullCaptureTransport struct {
	mu           sync.Mutex
	capturedReq  *http.Request
	capturedBody []byte
	callCount    int64
	sseReply     string
}

func (f *fullCaptureTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	atomic.AddInt64(&f.callCount, 1)
	f.mu.Lock()
	f.capturedReq = req
	if req.Body != nil {
		body, err := io.ReadAll(req.Body)
		if err != nil {
			f.mu.Unlock()
			return nil, err
		}
		f.capturedBody = body
		req.Body = io.NopCloser(bytes.NewReader(body))
	}
	f.mu.Unlock()

	resp := &http.Response{
		StatusCode: http.StatusOK,
		Header:     make(http.Header),
		Body:       io.NopCloser(strings.NewReader(f.sseReply)),
	}
	resp.Header.Set("Content-Type", "text/event-stream")
	return resp, nil
}

// testB05ThroughputMatchesAccountAndProfile: Intercept the real wire request after
// build and provider sign for both official models, asserting envelope and headers.
func testB05ThroughputMatchesAccountAndProfile(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	sessionPattern := regexp.MustCompile(`^-[0-9]+$`)
	requestIDPattern := regexp.MustCompile(`^agent-[a-zA-Z0-9_-]+$`)

	payload := `{
		"model": "m",
		"max_tokens": 2048,
		"tools": [{"name":"read_file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
		"messages": [{"role":"user","content":"inspect project structure"}]
	}`

	for _, model := range antigravityModels {
		t.Run(model, func(t *testing.T) {
			s := New()
			tr := &fullCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"structure ok"}]},"finishReason":"STOP"}]}}`)}
			s.client = &http.Client{Transport: tr}

			const (
				testUser    = "agent-user@example.com"
				testProject = "ag-corp-proj-999"
				testToken   = "ya29.antigravity-real-token-val"
			)

			p := provider.AntigravityTestProvider("antigravity", testUser, testProject, testToken)

			rec := httptest.NewRecorder()
			httpReq := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(payload)))
			httpReq.Header.Set("Content-Type", "application/json")

			var u Usage
			status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, model, []byte(payload), &u)
			if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
				t.Fatalf("translate returned status %d/%d msg=%q", status, rec.Code, msg)
			}
			if atomic.LoadInt64(&tr.callCount) != 1 {
				t.Fatalf("callCount = %d, want 1", tr.callCount)
			}

			// 1. Assert Headers
			req := tr.capturedReq
			if auth := req.Header.Get("Authorization"); auth != "Bearer "+testToken {
				t.Errorf("Authorization header mismatch: got %q, want Bearer %s", auth, testToken)
			}
			if ua := req.Header.Get("User-Agent"); !strings.HasPrefix(ua, "antigravity/hub/") {
				t.Errorf("User-Agent header must have 'antigravity/hub/' prefix, got %q", ua)
			}

			// 2. Assert Envelope Body
			var env map[string]any
			if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
				t.Fatalf("unmarshal wire body failed: %v\nbody: %s", err, tr.capturedBody)
			}

			if env["model"] != model {
				t.Errorf("envelope.model = %v, want %s", env["model"], model)
			}
			if env["project"] != testProject {
				t.Errorf("envelope.project = %v, want %s", env["project"], testProject)
			}
			if env["userAgent"] != "antigravity" {
				t.Errorf("envelope.userAgent = %v, want 'antigravity'", env["userAgent"])
			}
			if env["requestType"] != "agent" {
				t.Errorf("envelope.requestType = %v, want 'agent'", env["requestType"])
			}
			reqID, _ := env["requestId"].(string)
			if !requestIDPattern.MatchString(reqID) {
				t.Errorf("envelope.requestId = %q, want matching 'agent-[a-zA-Z0-9]+'", reqID)
			}

			// 3. Assert Inner Request (sessionId, generationConfig, tools)
			innerReq, ok := env["request"].(map[string]any)
			if !ok {
				t.Fatalf("envelope.request missing or invalid: %v", env["request"])
			}
			sessID, _ := innerReq["sessionId"].(string)
			if !sessionPattern.MatchString(sessID) {
				t.Errorf("inner.sessionId = %q, want negative integer string", sessID)
			}

			tools, ok := innerReq["tools"].([]any)
			if !ok || len(tools) != 1 {
				t.Fatalf("inner.tools mismatch: %v", innerReq["tools"])
			}
			decl := tools[0].(map[string]any)["functionDeclarations"].([]any)
			if len(decl) != 1 || decl[0].(map[string]any)["name"] != "read_file" {
				t.Errorf("functionDeclarations mismatch: %v", decl)
			}

			genCfg, ok := innerReq["generationConfig"].(map[string]any)
			if !ok {
				t.Fatalf("inner.generationConfig missing: %v", innerReq["generationConfig"])
			}
			if maxTok, _ := genCfg["maxOutputTokens"].(float64); int(maxTok) != 2048 {
				t.Errorf("maxOutputTokens = %v, want 2048", genCfg["maxOutputTokens"])
			}
		})
	}
}

// testB05NonTargetIsolation: Control cases — non-Antigravity providers must not be
// rewritten with Antigravity envelope fields.
func testB05NonTargetIsolation(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	payload := `{"model":"gemini-2.5-pro","max_tokens":100,"messages":[{"role":"user","content":"hi non-ag"}]}`

	// 1. Control A: Gemini CodeAssist account (Account.Agent == "gemini")
	t.Run("GeminiAccountControl", func(t *testing.T) {
		s := New()
		tr := &fullCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"gemini ok"}]},"finishReason":"STOP"}]}}`)}
		s.client = &http.Client{Transport: tr}

		p := provider.GeminiTestProvider("gemini-cli", "gemini-user@example.com", "gemini-proj-1", "gemini-token-1")

		rec := httptest.NewRecorder()
		httpReq := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(payload)))
		var u Usage
		status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "gemini-2.5-pro", []byte(payload), &u)
		if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
			t.Fatalf("translate returned status %d/%d msg=%q", status, rec.Code, msg)
		}

		ua := tr.capturedReq.Header.Get("User-Agent")
		if strings.Contains(ua, "antigravity") {
			t.Errorf("Gemini CLI User-Agent must not contain antigravity, got %q", ua)
		}

		var env map[string]any
		if err := json.Unmarshal(tr.capturedBody, &env); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if env["userAgent"] == "antigravity" || env["requestType"] == "agent" {
			t.Errorf("Gemini wire envelope contaminated with Antigravity fields: %v", env)
		}
		if _, hasPromptID := env["user_prompt_id"]; !hasPromptID {
			t.Errorf("Gemini wire envelope expected user_prompt_id, got: %v", env)
		}
		if r, ok := env["request"].(map[string]any); ok && r["sessionId"] != nil {
			t.Errorf("Gemini wire request must not have sessionId: %v", r["sessionId"])
		}
	})

	// 2. Control B: Plain non-CodeAssist provider (Anthropic)
	t.Run("PlainAnthropicControl", func(t *testing.T) {
		s := New()
		tr := &fullCaptureTransport{sseReply: sse(`data: {"type":"message_stop"}`)}
		s.client = &http.Client{Transport: tr}

		p := provider.Provider{
			ID:        "plain-anthropic",
			Name:      "Anthropic",
			Key:       "sk-ant-test-key",
			Anthropic: "http://example.com/v1",
		}

		rec := httptest.NewRecorder()
		anthropicPayload := `{"model":"claude-3-haiku","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
		httpReq := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(anthropicPayload)))
		var u Usage
		status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.Anthropic, "claude-3-haiku", []byte(anthropicPayload), &u)
		if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
			t.Fatalf("translate returned status %d/%d msg=%q", status, rec.Code, msg)
		}

		// Must be standard Anthropic request, completely free of CodeAssist envelopes
		var antReq map[string]any
		if err := json.Unmarshal(tr.capturedBody, &antReq); err != nil {
			t.Fatalf("unmarshal error: %v", err)
		}
		if antReq["project"] != nil || antReq["userAgent"] != nil || antReq["requestType"] != nil {
			t.Errorf("Plain Anthropic request contaminated with CodeAssist fields: %v", antReq)
		}
	})
}

// testB05ProviderDoesNotOverwritePredefinedFields: Antigravity provider must NOT overwrite
// envelope fields (project, userAgent, requestType, requestId, sessionId) already established.
func testB05ProviderDoesNotOverwritePredefinedFields(t *testing.T) {
	const (
		fixedProject   = "adapter-explicit-project"
		fixedUserAgent = "adapter-explicit-agent"
		fixedReqType   = "web_search"
		fixedReqID     = "agent-explicit-request-id-99"
		fixedSessionID = "-987654321012345"
	)

	predefinedBody := map[string]any{
		"model":       "claude-sonnet-4-6",
		"project":     fixedProject,
		"userAgent":   fixedUserAgent,
		"requestType": fixedReqType,
		"requestId":   fixedReqID,
		"request": map[string]any{
			"sessionId": fixedSessionID,
			"contents": []any{
				map[string]any{
					"role":  "user",
					"parts": []any{map[string]any{"text": "test predefined"}},
				},
			},
		},
	}
	bodyBytes, err := json.Marshal(predefinedBody)
	if err != nil {
		t.Fatalf("marshal error: %v", err)
	}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "default-fallback-proj", "tok-123")
	req, _ := http.NewRequestWithContext(context.Background(), "POST", "http://example.com", bytes.NewReader(bodyBytes))

	if err := p.Sign(context.Background(), req, provider.CodeAssist, bodyBytes); err != nil {
		t.Fatalf("Sign error: %v", err)
	}

	var env map[string]any
	b, _ := io.ReadAll(req.Body)
	if err := json.Unmarshal(b, &env); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}

	if env["project"] != fixedProject {
		t.Errorf("project was overwritten: got %v, want %s", env["project"], fixedProject)
	}
	if env["userAgent"] != fixedUserAgent {
		t.Errorf("userAgent was overwritten: got %v, want %s", env["userAgent"], fixedUserAgent)
	}
	if env["requestType"] != fixedReqType {
		t.Errorf("requestType was overwritten: got %v, want %s", env["requestType"], fixedReqType)
	}
	if env["requestId"] != fixedReqID {
		t.Errorf("requestId was overwritten: got %v, want %s", env["requestId"], fixedReqID)
	}
	r, ok := env["request"].(map[string]any)
	if !ok || r["sessionId"] != fixedSessionID {
		t.Errorf("sessionId was overwritten: got %v, want %s", r["sessionId"], fixedSessionID)
	}
}

// testB05ScopeIsolationSameProviderID: Distinct accounts under the same Provider.ID
// must not share or cross-contaminate signing scopes (tokens and projects).
func testB05ScopeIsolationSameProviderID(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	const (
		commonProviderID = "antigravity"
		userAlice        = "alice@example.com"
		projectAlice     = "proj-alice-111"
		tokenAlice       = "ya29.alice-token-val"

		userBob    = "bob@example.com"
		projectBob = "proj-bob-222"
		tokenBob   = "ya29.bob-token-val"
	)

	pAlice := provider.AntigravityTestProvider(commonProviderID, userAlice, projectAlice, tokenAlice)
	pBob := provider.AntigravityTestProvider(commonProviderID, userBob, projectBob, tokenBob)

	payloadAlice := `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"request from alice"}]}`
	payloadBob := `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"request from bob"}]}`

	// Alice request
	trAlice := &fullCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"alice ok"}]},"finishReason":"STOP"}]}}`)}
	sAlice := New()
	sAlice.client = &http.Client{Transport: trAlice}
	recAlice := httptest.NewRecorder()
	reqAlice := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(payloadAlice))
	var uAlice Usage
	if status, _ := sAlice.translate(recAlice, reqAlice, pAlice, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(payloadAlice), &uAlice); status != http.StatusOK {
		t.Fatalf("Alice translate failed: %d", status)
	}

	// Bob request
	trBob := &fullCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"bob ok"}]},"finishReason":"STOP"}]}}`)}
	sBob := New()
	sBob.client = &http.Client{Transport: trBob}
	recBob := httptest.NewRecorder()
	reqBob := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(payloadBob))
	var uBob Usage
	if status, _ := sBob.translate(recBob, reqBob, pBob, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(payloadBob), &uBob); status != http.StatusOK {
		t.Fatalf("Bob translate failed: %d", status)
	}

	// Assert Alice request
	if trAlice.capturedReq.Header.Get("Authorization") != "Bearer "+tokenAlice {
		t.Errorf("Alice Authorization header mismatch: got %q, want Bearer %s", trAlice.capturedReq.Header.Get("Authorization"), tokenAlice)
	}
	var envAlice map[string]any
	json.Unmarshal(trAlice.capturedBody, &envAlice)
	if envAlice["project"] != projectAlice {
		t.Errorf("Alice project mismatch: got %v, want %s", envAlice["project"], projectAlice)
	}

	// Assert Bob request
	if trBob.capturedReq.Header.Get("Authorization") != "Bearer "+tokenBob {
		t.Errorf("Bob Authorization header mismatch: got %q, want Bearer %s", trBob.capturedReq.Header.Get("Authorization"), tokenBob)
	}
	var envBob map[string]any
	json.Unmarshal(trBob.capturedBody, &envBob)
	if envBob["project"] != projectBob {
		t.Errorf("Bob project mismatch: got %v, want %s", envBob["project"], projectBob)
	}
}
