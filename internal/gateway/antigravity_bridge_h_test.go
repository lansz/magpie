package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/yetone/magpie/internal/provider"
)

// testB02AgentBridgeH covers all H-layer integration B02 BDD scenarios.
func testB02AgentBridgeH(t *testing.T) {
	t.Run("UnknownProfile_Isolation_HLayer", testB02UnknownProfileIsolationHLayer)
	t.Run("TranslateAcrossProtocols", testB02TranslateAcrossProtocols)
	t.Run("AskTranslated_MultiTurnRound", testB02AskTranslatedMultiTurnRound)
}

// 1. Unknown Profile isolation: HTTP entry test.
// Note: Blackbox HTTP output verifies declared tools are unchanged and unpolluted;
// internal clientProfile field storage is verified at unit level and by code review.
func testB02UnknownProfileIsolationHLayer(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	expectedTool := Tool{
		Name:        "user_defined_calculator",
		Description: "Calculates mathematical expressions",
		Schema:      json.RawMessage([]byte(`{"properties":{"expr":{"type":"string"}},"required":["expr"],"type":"object"}`)),
	}

	var mu sync.Mutex
	var capturedUpstreamBody []byte

	upChat := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("upstream ReadAll failed: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		mu.Lock()
		capturedUpstreamBody = body
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		respPayload := sse(
			`data: {"id":"c_iso","choices":[{"delta":{"content":"calc ok"},"finish_reason":null}]}`,
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: [DONE]`,
		)
		if _, writeErr := io.WriteString(w, respPayload); writeErr != nil {
			t.Errorf("upstream WriteString failed: %v", writeErr)
		}
	}))
	t.Cleanup(upChat.Close)

	testP := provider.Provider{
		ID:     "iso-prov",
		Name:   "IsoProvider",
		Key:    "k",
		Models: []string{"iso-model"},
		Chat:   upChat.URL + "/v1", // Force translate: client Anthropic -> upstream Chat
	}
	if err := provider.Save(testP); err != nil {
		t.Fatalf("save provider failed: %v", err)
	}

	rawPayload := fmt.Sprintf(`{
		"model": "iso-model",
		"max_tokens": 100,
		"messages": [{"role":"user","content":"do math"}],
		"tools": [{
			"name": %q,
			"description": %q,
			"input_schema": {"type":"object","properties":{"expr":{"type":"string"}},"required":["expr"]}
		}]
	}`, expectedTool.Name, expectedTool.Description)

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", strings.NewReader(rawPayload))
	httpReq.Header.Set("Content-Type", "application/json")
	// Claude-specific hints in headers
	httpReq.Header.Set("x-claude-code-session-id", "session_cl_99999")
	httpReq.Header.Set("User-Agent", "Claude-Code/1.0.0")

	New().Handler().ServeHTTP(rec, httpReq)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d: %s", rec.Code, rec.Body.String())
	}

	// Verify client received a valid Anthropic JSON response with expected content and finish reason
	var clientReply struct {
		ID      string `json:"id"`
		Content []struct {
			Type string `json:"type"`
			Text string `json:"text"`
		} `json:"content"`
		StopReason string `json:"stop_reason"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &clientReply); err != nil {
		t.Fatalf("failed to unmarshal client response: %v\nBody: %s", err, rec.Body.String())
	}
	if len(clientReply.Content) == 0 || clientReply.Content[0].Text != "calc ok" {
		t.Errorf("client reply text mismatch: %v", clientReply.Content)
	}
	if clientReply.StopReason != "end_turn" {
		t.Errorf("client stop_reason mismatch: got %q, want 'end_turn'", clientReply.StopReason)
	}

	// Verify upstream Chat request has strictly the declared tool and no Claude private injections
	mu.Lock()
	upBody := capturedUpstreamBody
	mu.Unlock()

	var upChatPayload struct {
		Tools []struct {
			Type     string `json:"type"`
			Function struct {
				Name        string          `json:"name"`
				Description string          `json:"description"`
				Parameters  json.RawMessage `json:"parameters"`
			} `json:"function"`
		} `json:"tools"`
	}
	if err := json.Unmarshal(upBody, &upChatPayload); err != nil {
		t.Fatalf("failed to unmarshal upstream body: %v\nBody: %s", err, string(upBody))
	}
	if len(upChatPayload.Tools) != 1 {
		t.Fatalf("upstream tools count mismatch: got %d, want 1", len(upChatPayload.Tools))
	}
	upTool := upChatPayload.Tools[0].Function
	if upTool.Name != expectedTool.Name {
		t.Errorf("upstream tool name mismatch: got %q, want %q", upTool.Name, expectedTool.Name)
	}
	if upTool.Description != expectedTool.Description {
		t.Errorf("upstream tool description mismatch: got %q, want %q", upTool.Description, expectedTool.Description)
	}
	eq, err := jsonEqualExact(upTool.Parameters, expectedTool.Schema)
	if err != nil || !eq {
		t.Errorf("upstream tool schema mismatch: got %s, want %s (err: %v)", string(upTool.Parameters), string(expectedTool.Schema), err)
	}
}

// 2. Protocol translation across three paths with complete client/upstream parsing (table-driven).
func testB02TranslateAcrossProtocols(t *testing.T) {
	cases := []struct {
		name           string
		endpoint       string
		clientPayload  string
		upstreamProto  provider.Protocol
		upstreamSSE    string
		assertClient   func(t *testing.T, body string)
		assertUpstream func(t *testing.T, raw []byte)
	}{
		{
			name:          "AnthropicToChat",
			endpoint:      "/v1/messages",
			clientPayload: `{"model":"m1","max_tokens":50,"messages":[{"role":"user","content":"hello"}]}`,
			upstreamProto: provider.Chat,
			upstreamSSE: sse(
				`data: {"id":"c1","choices":[{"delta":{"content":"chat reply ok"},"finish_reason":null}]}`,
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
				`data: [DONE]`,
			),
			assertClient: func(t *testing.T, body string) {
				var r struct {
					Content    []struct{ Text string } `json:"content"`
					StopReason string                  `json:"stop_reason"`
				}
				if err := json.Unmarshal([]byte(body), &r); err != nil {
					t.Fatalf("unmarshal client reply failed: %v", err)
				}
				if len(r.Content) == 0 || r.Content[0].Text != "chat reply ok" || r.StopReason != "end_turn" {
					t.Errorf("client reply mismatch: %+v", r)
				}
			},
			assertUpstream: func(t *testing.T, raw []byte) {
				var u struct {
					Model    string `json:"model"`
					Stream   bool   `json:"stream"`
					Messages []struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"messages"`
				}
				if err := json.Unmarshal(raw, &u); err != nil {
					t.Fatalf("unmarshal upstream body failed: %v", err)
				}
				if u.Model != "m1" || !u.Stream || len(u.Messages) == 0 || u.Messages[0].Role != "user" || u.Messages[0].Content != "hello" {
					t.Errorf("upstream chat payload mismatch: %+v", u)
				}
			},
		},
		{
			name:          "ChatToResponses",
			endpoint:      "/chat/completions",
			clientPayload: `{"model":"m1","messages":[{"role":"user","content":"hello"}]}`,
			upstreamProto: provider.Responses,
			upstreamSSE: sse(
				`event: response.created`+"\n"+`data: {"type":"response.created","response":{"id":"r1","model":"m1","usage":{"input_tokens":0,"output_tokens":0}}}`,
				`event: response.output_text.delta`+"\n"+`data: {"type":"response.output_text.delta","delta":"responses reply ok"}`,
				`event: response.completed`+"\n"+`data: {"type":"response.completed","response":{"id":"r1","model":"m1","status":"completed","usage":{"input_tokens":2,"output_tokens":2}}}`,
			),
			assertClient: func(t *testing.T, body string) {
				var r struct {
					Choices []struct {
						Message      struct{ Content string } `json:"message"`
						FinishReason string                   `json:"finish_reason"`
					} `json:"choices"`
				}
				if err := json.Unmarshal([]byte(body), &r); err != nil {
					t.Fatalf("unmarshal client reply failed: %v", err)
				}
				if len(r.Choices) == 0 || r.Choices[0].Message.Content != "responses reply ok" || r.Choices[0].FinishReason != "stop" {
					t.Errorf("client reply mismatch: %+v", r)
				}
			},
			assertUpstream: func(t *testing.T, raw []byte) {
				var u struct {
					Model  string `json:"model"`
					Stream bool   `json:"stream"`
					Input  []struct {
						Role    string                  `json:"role"`
						Content []struct{ Text string } `json:"content"`
					} `json:"input"`
				}
				if err := json.Unmarshal(raw, &u); err != nil {
					t.Fatalf("unmarshal upstream body failed: %v", err)
				}
				if u.Model != "m1" || !u.Stream || len(u.Input) == 0 || u.Input[0].Role != "user" || len(u.Input[0].Content) == 0 || u.Input[0].Content[0].Text != "hello" {
					t.Errorf("upstream responses payload mismatch: %+v", u)
				}
			},
		},
		{
			name:          "ResponsesToChat",
			endpoint:      "/responses",
			clientPayload: `{"model":"m1","input":[{"role":"user","content":"hello"}]}`,
			upstreamProto: provider.Chat,
			upstreamSSE: sse(
				`data: {"id":"c2","choices":[{"delta":{"content":"chat to responses reply"},"finish_reason":null}]}`,
				`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
				`data: [DONE]`,
			),
			assertClient: func(t *testing.T, body string) {
				var r struct {
					Status string `json:"status"`
					Output []struct {
						Content []struct{ Text string } `json:"content"`
					} `json:"output"`
				}
				if err := json.Unmarshal([]byte(body), &r); err != nil {
					t.Fatalf("unmarshal client reply failed: %v", err)
				}
				if r.Status != "completed" || len(r.Output) == 0 || len(r.Output[0].Content) == 0 || r.Output[0].Content[0].Text != "chat to responses reply" {
					t.Errorf("client reply mismatch: %+v", r)
				}
			},
			assertUpstream: func(t *testing.T, raw []byte) {
				var u struct {
					Model    string `json:"model"`
					Stream   bool   `json:"stream"`
					Messages []struct {
						Role    string `json:"role"`
						Content string `json:"content"`
					} `json:"messages"`
				}
				if err := json.Unmarshal(raw, &u); err != nil {
					t.Fatalf("unmarshal upstream body failed: %v", err)
				}
				if u.Model != "m1" || !u.Stream || len(u.Messages) == 0 || u.Messages[0].Role != "user" || u.Messages[0].Content != "hello" {
					t.Errorf("upstream chat payload mismatch: %+v", u)
				}
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Isolate configuration per case so providers do not accumulate
			t.Setenv("XDG_CONFIG_HOME", t.TempDir())
			t.Setenv("XDG_CACHE_HOME", t.TempDir())

			var mu sync.Mutex
			var captured []byte

			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				b, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("upstream ReadAll failed: %v", err)
					http.Error(w, err.Error(), http.StatusInternalServerError)
					return
				}
				mu.Lock()
				captured = b
				mu.Unlock()

				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				if _, writeErr := io.WriteString(w, tc.upstreamSSE); writeErr != nil {
					t.Errorf("upstream WriteString failed: %v", writeErr)
				}
			}))
			defer up.Close()

			prov := provider.Provider{ID: "p-" + strings.ToLower(tc.name), Name: "P", Key: "k", Models: []string{"m1"}}
			switch tc.upstreamProto {
			case provider.Chat:
				prov.Chat = up.URL + "/v1"
			case provider.Responses:
				prov.Responses = up.URL + "/v1"
			case provider.Anthropic:
				prov.Anthropic = up.URL
			}
			if err := provider.Save(prov); err != nil {
				t.Fatal(err)
			}

			code, body := post(t, tc.endpoint, tc.clientPayload)
			if code != http.StatusOK {
				t.Fatalf("[%s] expected HTTP 200, got %d: %s", tc.name, code, body)
			}

			tc.assertClient(t, body)

			mu.Lock()
			upRaw := captured
			mu.Unlock()
			tc.assertUpstream(t, upRaw)
		})
	}
}

// 3. askTranslated multi-turn throughput: uses current turn's IR, asserts events, and handles round status (0 means success in search.go).
func testB02AskTranslatedMultiTurnRound(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())

	var mu sync.Mutex
	var capturedUpstreamBodies [][]byte

	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("upstream ReadAll failed: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}
		mu.Lock()
		capturedUpstreamBodies = append(capturedUpstreamBodies, b)
		callCount := len(capturedUpstreamBodies)
		mu.Unlock()

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		content := fmt.Sprintf("round ok turn %d", callCount)
		chunk := sse(
			fmt.Sprintf(`data: {"id":"c%d","choices":[{"delta":{"content":%q},"finish_reason":null}]}`, callCount, content),
			`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}`,
			`data: [DONE]`,
		)
		if _, writeErr := io.WriteString(w, chunk); writeErr != nil {
			t.Errorf("upstream WriteString failed: %v", writeErr)
		}
	}))
	t.Cleanup(up.Close)

	p := provider.Provider{ID: "p-ask", Name: "PAsk", Key: "k", Models: []string{"m_ask"}, Chat: up.URL + "/v1"}
	s := New()

	inHdr := http.Header{}
	replyHdr := http.Header{}

	// Call askTranslated with 7 parameters as required by the final contract:
	// (p, to, model, in, reply, sourceProto, clientProfile)
	rnd := s.askTranslated(p, provider.Chat, "m_ask", inHdr, replyHdr, provider.Anthropic, "test_profile")
	if rnd == nil {
		t.Fatalf("askTranslated returned nil round")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Turn 1: In search.go round contract, status==0 and errStr=="" indicates success
	req1 := &Request{
		Model: "m_ask",
		Messages: []Message{
			{Role: "user", Parts: []Part{{Kind: Text, Text: "question one"}}},
		},
	}
	evCh1, status1, errStr1 := rnd(ctx, req1)
	if evCh1 == nil || status1 != 0 || errStr1 != "" {
		t.Fatalf("turn 1 failed: status=%d err=%s", status1, errStr1)
	}

	var turn1Text string
	var turn1Stop string
	for ev := range evCh1 {
		if ev.Kind == KError {
			t.Fatalf("turn 1 stream emitted error event: %s", ev.Text)
		}
		if ev.Kind == KText {
			turn1Text += ev.Text
		}
		if ev.Kind == KStop {
			turn1Stop = ev.Stop
		}
	}
	if !strings.Contains(turn1Text, "round ok turn 1") {
		t.Errorf("turn 1 text mismatch: got %q, want containing 'round ok turn 1'", turn1Text)
	}
	if turn1Stop != "stop" {
		t.Errorf("turn 1 stop mismatch: got %q, want 'stop'", turn1Stop)
	}

	// Turn 2: different Request pointer and distinct question text
	req2 := &Request{
		Model: "m_ask",
		Messages: []Message{
			{Role: "user", Parts: []Part{{Kind: Text, Text: "question two"}}},
		},
	}
	evCh2, status2, errStr2 := rnd(ctx, req2)
	if evCh2 == nil || status2 != 0 || errStr2 != "" {
		t.Fatalf("turn 2 failed: status=%d err=%s", status2, errStr2)
	}

	var turn2Text string
	var turn2Stop string
	for ev := range evCh2 {
		if ev.Kind == KError {
			t.Fatalf("turn 2 stream emitted error event: %s", ev.Text)
		}
		if ev.Kind == KText {
			turn2Text += ev.Text
		}
		if ev.Kind == KStop {
			turn2Stop = ev.Stop
		}
	}
	if !strings.Contains(turn2Text, "round ok turn 2") {
		t.Errorf("turn 2 text mismatch: got %q, want containing 'round ok turn 2'", turn2Text)
	}
	if turn2Stop != "stop" {
		t.Errorf("turn 2 stop mismatch: got %q, want 'stop'", turn2Stop)
	}

	// Assert exactly 2 distinct upstream calls were made with respective turn IRs
	mu.Lock()
	defer mu.Unlock()
	if len(capturedUpstreamBodies) != 2 {
		t.Fatalf("expected 2 upstream calls, got %d", len(capturedUpstreamBodies))
	}
	if !strings.Contains(string(capturedUpstreamBodies[0]), "question one") {
		t.Errorf("turn 1 upstream body did not contain 'question one': %s", string(capturedUpstreamBodies[0]))
	}
	if !strings.Contains(string(capturedUpstreamBodies[1]), "question two") {
		t.Errorf("turn 2 upstream body did not contain 'question two': %s", string(capturedUpstreamBodies[1]))
	}
}
