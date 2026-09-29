package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB08ToolBinding covers B08: Recovering complete native tool calls (id, name, full args)
// from binding records after mapping to client tools; never falling back to name="tool";
// handling multiple calls with the same name without collision; and ensuring atomic registration
// with zero half-state residue on validation failure.
func testB08ToolBinding(t *testing.T) {
	t.Run("Unit_RestoreNativeNameAndArgs", testB08RestoreNativeNameAndArgsU)
	t.Run("Unit_MultipleSameNameCalls", testB08MultipleSameNameCallsU)
	t.Run("Unit_AtomicFailureLeavesNoHalfState", testB08AtomicFailureLeavesNoHalfStateU)
	t.Run("HLayer_NeverFallbackToGenericToolName", testB08NeverFallbackToGenericToolNameH)
}

// 1. Recover original native ID, native name, and complete original arguments.
func testB08RestoreNativeNameAndArgsU(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	const (
		nativeCallID = "call_view_file_001"
		nativeName   = "view_file"
		fullArgsJSON = `{"path":"/path/to/main.go","toolAction":"view","toolSummary":"View File"}`

		clientCallID = "toolu_client_view_01"
		clientName   = "Read"
		clientArgs   = `{"file_path":"/path/to/main.go"}`
	)

	binding := AntigravityToolBinding{
		NativeID:   nativeCallID,
		NativeName: nativeName,
		NativeArgs: json.RawMessage(fullArgsJSON),
		ClientID:   clientCallID,
		ClientName: clientName,
		ClientArgs: json.RawMessage(clientArgs),
	}

	if err := defaultToolBindingStore.Bind(binding); err != nil {
		t.Fatalf("Bind error: %v", err)
	}

	// Client sends back turn with its simplified/mapped tool use & result
	clientBody := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role":"user","content":"read file"},
			{"role":"assistant","content":[{"type":"tool_use","id":"` + clientCallID + `","name":"` + clientName + `","input":` + clientArgs + `}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + clientCallID + `","content":"file content A"}]}
		]
	}`

	req, err := parseAnthropic([]byte(clientBody))
	if err != nil {
		t.Fatalf("parseAnthropic: %v", err)
	}

	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")
	var env struct {
		Request struct {
			Contents []struct {
				Role  string `json:"role"`
				Parts []struct {
					FunctionCall *struct {
						ID   string          `json:"id"`
						Name string          `json:"name"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
					FunctionResponse *struct {
						ID       string `json:"id"`
						Name     string `json:"name"`
						Response struct {
							Output string `json:"output"`
						} `json:"response"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}

	if err := json.Unmarshal(wireBytes, &env); err != nil {
		t.Fatalf("unmarshal: %v\nwire: %s", err, wireBytes)
	}

	// Locate functionCall and functionResponse
	var fcID, fcName, frID, frName string
	var fcArgs json.RawMessage
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				fcID = p.FunctionCall.ID
				fcName = p.FunctionCall.Name
				fcArgs = p.FunctionCall.Args
			}
			if p.FunctionResponse != nil {
				frID = p.FunctionResponse.ID
				frName = p.FunctionResponse.Name
			}
		}
	}

	// 1. Assert functionCall has restored native id, name, and complete args
	if fcID != nativeCallID {
		t.Errorf("functionCall.id = %q, want native ID %q", fcID, nativeCallID)
	}
	if fcName != nativeName {
		t.Errorf("functionCall.name = %q, want native name %q", fcName, nativeName)
	}
	eq, err := jsonEqualExact(fcArgs, json.RawMessage(fullArgsJSON))
	if err != nil || !eq {
		t.Errorf("functionCall.args did not restore full native arguments: got %s, want %s", string(fcArgs), fullArgsJSON)
	}

	// 2. Assert functionResponse has restored native name (never fallback to "tool")
	if frName != nativeName {
		t.Errorf("functionResponse.name = %q, want %q (must not fallback to 'tool')", frName, nativeName)
	}
	if frID != nativeCallID {
		t.Errorf("functionResponse.id = %q, want native ID %q", frID, nativeCallID)
	}
}

// 2. Multiple calls with the same name in the same turn maintain separate identities and args.
func testB08MultipleSameNameCallsU(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	b1 := AntigravityToolBinding{
		NativeID:   "native_call_1",
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{"path":"/fileA.go","toolAction":"view","toolSummary":"A"}`),
		ClientID:   "client_call_1",
		ClientName: "Read",
	}
	b2 := AntigravityToolBinding{
		NativeID:   "native_call_2",
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{"path":"/fileB.go","toolAction":"view","toolSummary":"B"}`),
		ClientID:   "client_call_2",
		ClientName: "Read",
	}

	if err := defaultToolBindingStore.Bind(b1, b2); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	clientBody := `{
		"model": "claude-sonnet-4-6",
		"messages": [
			{"role":"user","content":"read both"},
			{"role":"assistant","content":[
				{"type":"tool_use","id":"client_call_1","name":"Read","input":{"file_path":"/fileA.go"}},
				{"type":"tool_use","id":"client_call_2","name":"Read","input":{"file_path":"/fileB.go"}}
			]},
			{"role":"user","content":[
				{"type":"tool_result","tool_use_id":"client_call_1","content":"content A"},
				{"type":"tool_result","tool_use_id":"client_call_2","content":"content B"}
			]}
		]
	}`

	req, _ := parseAnthropic([]byte(clientBody))
	wireBytes := buildCodeAssist(req, "claude-sonnet-4-6", "antigravity")

	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionCall *struct {
						ID   string          `json:"id"`
						Args json.RawMessage `json:"args"`
					} `json:"functionCall"`
					FunctionResponse *struct {
						ID   string `json:"id"`
						Name string `json:"name"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	json.Unmarshal(wireBytes, &env)

	var calls []struct{ id, path string }
	var resps []struct{ id, name string }
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionCall != nil {
				var a struct{ Path string }
				json.Unmarshal(p.FunctionCall.Args, &a)
				calls = append(calls, struct{ id, path string }{p.FunctionCall.ID, a.Path})
			}
			if p.FunctionResponse != nil {
				resps = append(resps, struct{ id, name string }{p.FunctionResponse.ID, p.FunctionResponse.Name})
			}
		}
	}

	if len(calls) != 2 || len(resps) != 2 {
		t.Fatalf("expected 2 calls and 2 responses, got calls=%d resps=%d", len(calls), len(resps))
	}

	if calls[0].id != "native_call_1" || calls[0].path != "/fileA.go" {
		t.Errorf("call 0 mismatch: %v", calls[0])
	}
	if calls[1].id != "native_call_2" || calls[1].path != "/fileB.go" {
		t.Errorf("call 1 mismatch: %v", calls[1])
	}

	if resps[0].id != "native_call_1" || resps[0].name != "view_file" {
		t.Errorf("resp 0 mismatch: %v", resps[0])
	}
	if resps[1].id != "native_call_2" || resps[1].name != "view_file" {
		t.Errorf("resp 1 mismatch: %v", resps[1])
	}
}

// 3. Atomic registration: if any binding fails validation, no partial state remains in store.
func testB08AtomicFailureLeavesNoHalfStateU(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	validItem := AntigravityToolBinding{
		NativeID:   "call_valid_1",
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{"path":"/a.go"}`),
		ClientID:   "client_valid_1",
		ClientName: "Read",
	}

	// Invalid item with broken JSON
	invalidItem := AntigravityToolBinding{
		NativeID:   "call_invalid_2",
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{broken-json:`),
		ClientID:   "client_invalid_2",
		ClientName: "Read",
	}

	err := defaultToolBindingStore.Bind(validItem, invalidItem)
	if err == nil {
		t.Fatalf("expected Bind to fail for broken JSON, got nil")
	}

	// Verify validItem was rolled back and is NOT in store
	if _, ok := defaultToolBindingStore.LookupByClientID("client_valid_1"); ok {
		t.Errorf("atomic rollback failed: client_valid_1 still exists in store")
	}
	if _, ok := defaultToolBindingStore.LookupByNativeID("call_valid_1"); ok {
		t.Errorf("atomic rollback failed: call_valid_1 still exists in store")
	}
}

// 4. H-Layer: When client sends tool_result without previous assistant tool_use in the turn,
// the gateway recovers the native name from binding instead of defaulting to name="tool".
func testB08NeverFallbackToGenericToolNameH(t *testing.T) {
	defaultToolBindingStore.clear()
	t.Cleanup(defaultToolBindingStore.clear)

	const (
		nativeCallID = "call_isolated_result_01"
		clientCallID = "toolu_isolated_result_01"
	)

	binding := AntigravityToolBinding{
		NativeID:   nativeCallID,
		NativeName: "view_file",
		NativeArgs: json.RawMessage(`{"path":"/isolated.go","toolAction":"view","toolSummary":"view"}`),
		ClientID:   clientCallID,
		ClientName: "Read",
	}
	if err := defaultToolBindingStore.Bind(binding); err != nil {
		t.Fatalf("Bind: %v", err)
	}

	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"done"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}

	p := provider.AntigravityTestProvider("antigravity", "user@test.com", "proj-test", "tok-test")

	// Notice: This message history contains ONLY the tool_result, NO prior assistant tool_use!
	// Previously this caused name := names[p.CallID] to be empty and fallback to name="tool".
	body := `{
		"model": "claude-sonnet-4-6",
		"max_tokens": 100,
		"messages": [
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"` + clientCallID + `","content":"read ok"}]}
		]
	}`

	rec := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/messages", bytes.NewReader([]byte(body)))
	var u Usage
	status, msg := s.translate(rec, httpReq, p, provider.Anthropic, provider.CodeAssist, "claude-sonnet-4-6", []byte(body), &u)

	if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
		t.Fatalf("translate returned status %d/%d msg=%q", status, rec.Code, msg)
	}
	if atomic.LoadInt64(&tr.callCount) != 1 {
		t.Fatalf("callCount = %d, want 1", tr.callCount)
	}

	var env struct {
		Request struct {
			Contents []struct {
				Parts []struct {
					FunctionResponse *struct {
						ID   string `json:"id"`
						Name string `json:"name"`
					} `json:"functionResponse"`
				} `json:"parts"`
			} `json:"contents"`
		} `json:"request"`
	}
	json.Unmarshal(tr.capturedBody, &env)

	var frName, frID string
	for _, c := range env.Request.Contents {
		for _, p := range c.Parts {
			if p.FunctionResponse != nil {
				frName = p.FunctionResponse.Name
				frID = p.FunctionResponse.ID
			}
		}
	}

	if frName == "tool" || frName != "view_file" {
		t.Errorf("functionResponse.name = %q, want 'view_file' (never fallback to 'tool')", frName)
	}
	if frID != nativeCallID {
		t.Errorf("functionResponse.id = %q, want native ID %q", frID, nativeCallID)
	}
}
