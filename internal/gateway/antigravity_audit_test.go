package gateway

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// auditTransport confines all gateway requests to synthetic in-memory responses.
type auditTransport struct {
	reply string
	body  []byte
	calls int
}

func (a *auditTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	b, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}
	a.body = b
	a.calls++
	return &http.Response{StatusCode: 200, Header: http.Header{"Content-Type": {"text/event-stream"}}, Body: io.NopCloser(strings.NewReader(a.reply))}, nil
}

// auditPersister observes production commits without writing user state.
type auditPersister struct {
	rounds   [][]canonicalPart
	bindings [][]AntigravityToolBinding
	fail     error
	before   func()
}

func (p *auditPersister) CommitRound(s antigravitySessionScope, id string, c []canonicalPart, b []AntigravityToolBinding) error {
	if p.before != nil {
		p.before()
	}
	if p.fail != nil {
		return p.fail
	}
	p.rounds = append(p.rounds, c)
	p.bindings = append(p.bindings, b)
	return nil
}

func (p *auditPersister) CommitRoundWithRevision(s antigravitySessionScope, id string, c []canonicalPart, b []AntigravityToolBinding, rev int) error {
	return p.CommitRound(s, id, c, b)
}

func (p *auditPersister) LoadRound(s antigravitySessionScope, id string) (*persistedRound, error) {
	return nil, errors.New("audit: no record")
}

// auditRun exercises translate with synthetic credentials; no real signer or account discovery is used.
func auditRun(t *testing.T, from provider.Protocol, body, reply string, persist *auditPersister) (*httptest.ResponseRecorder, *auditTransport, int) {
	t.Helper()
	old := currentAntigravityPersister
	currentAntigravityPersister = persist
	defer func() { currentAntigravityPersister = old }()
	tr := &auditTransport{reply: reply}
	s := &Server{client: &http.Client{Transport: tr}, unfit: map[string]bool{}}
	p := provider.Provider{ID: "audit", Name: "audit", Key: "synthetic", Account: &provider.Account{Agent: "antigravity", User: "synthetic-user", Project: "synthetic-project"}}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/dummy", strings.NewReader(body))
	req.Header.Set("X-Magpie-Session", "audit-session-"+t.Name())
	var usage Usage
	status, _ := s.translate(rec, req, p, from, provider.CodeAssist, "gemini-3.8-flash-high", []byte(body), &usage)
	return rec, tr, status
}

func auditSSE(parts, finish string) string {
	tail := ""
	if finish != "" {
		tail = `,"finishReason":"` + finish + `"`
	}
	return `data: {"response":{"candidates":[{"content":{"role":"model","parts":` + parts + `}` + tail + `}]}}` + "\n\n"
}

const auditTextBody = `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
const auditToolBody = `{"model":"m","tools":[{"name":"read","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`

func testB22AuditFindings(t *testing.T) {
	t.Run("ControlValidText", func(t *testing.T) {
		r, tr, status := auditRun(t, provider.Anthropic, auditTextBody, auditSSE(`[{"text":"ok"}]`, "STOP"), &auditPersister{})
		if status != 200 || tr.calls != 1 || !strings.Contains(r.Body.String(), `"text":"ok"`) {
			t.Fatal("valid baseline failed")
		}
	})
	t.Run("ControlUndeclaredRejected", func(t *testing.T) {
		_, _, status := auditRun(t, provider.Anthropic, auditToolBody, auditSSE(`[{"functionCall":{"id":"x","name":"unknown","args":{}}}]`, "STOP"), &auditPersister{})
		if status != 502 {
			t.Fatalf("status=%d", status)
		}
	})
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("MissingStop_stream_%t", stream), func(t *testing.T) {
			body := strings.Replace(auditTextBody, `"model":"m"`, fmt.Sprintf(`"model":"m","stream":%t`, stream), 1)
			r, _, status := auditRun(t, provider.Anthropic, body, auditSSE(`[{"text":"partial"}]`, ""), &auditPersister{})
			if status == 200 && (strings.Contains(r.Body.String(), "message_stop") || strings.Contains(r.Body.String(), `"stop_reason":"end_turn"`)) {
				t.Errorf("EOF without finishReason produced normal completion (status=%d)", status)
			}
		})
	}
	t.Run("PartialThenError", func(t *testing.T) {
		p := &auditPersister{}
		r, _, status := auditRun(t, provider.Anthropic, auditTextBody, auditSSE(`[{"text":"partial"}]`, "")+"data: {\"error\":{\"message\":\"synthetic upstream failure\"}}\n\n", p)
		if status != 502 || len(p.rounds) != 0 {
			t.Errorf("partial+error: status=%d commits=%d success=%t", status, len(p.rounds), strings.Contains(r.Body.String(), `"stop_reason":"end_turn"`))
		}
	})
	t.Run("SignedToolActuallyCommitted", func(t *testing.T) {
		p := &auditPersister{}
		_, _, status := auditRun(t, provider.Anthropic, auditToolBody, auditSSE(`[{"functionCall":{"id":"native-1","name":"read","args":{"path":"synthetic.txt"}},"thoughtSignature":"synthetic-opaque-S"}]`, "STOP"), p)
		if status != 200 || len(p.rounds) != 1 {
			t.Fatalf("unexpected setup status %d", status)
		}
		c := p.rounds[0]
		if len(c) != 1 || c[0].CallID != "native-1" || c[0].ThoughtSignature != "synthetic-opaque-S" || c[0].Args == "" || len(p.bindings[0]) == 0 {
			t.Errorf("production commit loses native identity/args/signature/bindings: parts=%+v bindings=%d", c, len(p.bindings[0]))
		}
	})
	t.Run("TailSignatureActuallyDecoded", func(t *testing.T) {
		var c collector
		d := &codeAssistDecoder{}
		err := readSSE(strings.NewReader(auditSSE(`[{"text":"hello"},{"text":"","thoughtSignature":"synthetic-tail"}]`, "STOP")), func(_, v string) error { return d.decode(v, c.add) })
		if err != nil {
			t.Fatal(err)
		}
		found := false
		for _, p := range c.finish().Parts {
			if p.Signature == "synthetic-tail" {
				found = true
			}
		}
		if !found {
			t.Error("actual decoder/collector discarded Gemini tail signature")
		}
	})
	t.Run("NoToolExposureBeforeCommit", func(t *testing.T) {
		var outputBefore string
		rec := httptest.NewRecorder()
		p := &auditPersister{fail: errors.New("synthetic disk failure"), before: func() { outputBefore = rec.Body.String() }}
		old := currentAntigravityPersister
		currentAntigravityPersister = p
		defer func() { currentAntigravityPersister = old }()
		tr := &auditTransport{reply: auditSSE(`[{"functionCall":{"id":"x","name":"read","args":{}}}]`, "STOP")}
		s := &Server{client: &http.Client{Transport: tr}, unfit: map[string]bool{}}
		body := strings.Replace(auditToolBody, `"model":"m"`, `"model":"m","stream":true`, 1)
		req := httptest.NewRequest("POST", "/dummy", strings.NewReader(body))
		req.Header.Set("X-Magpie-Session", "audit-session-"+t.Name())
		var u Usage
		s.translate(rec, req, provider.Provider{ID: "audit", Key: "synthetic", Account: &provider.Account{Agent: "antigravity"}}, provider.Anthropic, provider.CodeAssist, "gemini-3.8-flash-high", []byte(body), &u)
		if strings.Contains(outputBefore, `"type":"tool_use"`) {
			t.Error("client tool_use was already exposed before failed commit")
		}
	})
	t.Run("OwnResponseCanBeContinued", func(t *testing.T) {
		p := &auditPersister{}
		r, _, status := auditRun(t, provider.Responses, `{"model":"m","input":"hi"}`, auditSSE(`[{"text":"hello"}]`, "STOP"), p)
		if status != 200 {
			t.Fatal(status)
		}
		var v struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(r.Body.Bytes(), &v); err != nil {
			t.Fatal(err)
		}
		body := fmt.Sprintf(`{"model":"m","input":"next","previous_response_id":%q}`, v.ID)
		r, _, status = auditRun(t, provider.Responses, body, auditSSE(`[{"text":"next answer"}]`, "STOP"), p)
		if status != 200 {
			t.Errorf("freshly returned own response_id cannot resume: status=%d body=%s", status, r.Body.String())
		}
	})
	t.Run("SchemaConstraintsNotDropped", func(t *testing.T) {
		raw := json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{"n":{"type":"integer","enum":[1,2]}}}`)
		out := string(plainSchema(raw))
		if !strings.Contains(out, `"additionalProperties":false`) || !strings.Contains(out, `"enum":[1,2]`) {
			t.Errorf("constraints silently erased: %s", out)
		}
	})
	t.Run("NamedChoiceMatchesDeclaration", func(t *testing.T) {
		defaultToolBindingStore.clear()
		defer defaultToolBindingStore.clear()
		if err := defaultToolBindingStore.Bind(AntigravityToolBinding{NativeID: "n", NativeName: "view_file", NativeArgs: json.RawMessage(`{}`), ClientID: "c", ClientName: "Read"}); err != nil {
			t.Fatal(err)
		}
		req := &Request{Tools: []Tool{{Name: "Read", Schema: json.RawMessage(`{"type":"object"}`)}}, ToolChoice: "name:Read"}
		wire := string(buildCodeAssist(req, "claude-sonnet-4-6", "antigravity"))
		if strings.Contains(wire, `"allowedFunctionNames":["view_file"]`) && !strings.Contains(wire, `"name":"view_file"`) {
			t.Error("allowedFunctionNames=view_file, but only Read is declared")
		}
	})
	t.Run("ToolChoiceNoneEnforcedOnResponse", func(t *testing.T) {
		body := strings.Replace(auditToolBody, `"model":"m"`, `"model":"m","tool_choice":{"type":"none"}`, 1)
		_, _, status := auditRun(t, provider.Anthropic, body, auditSSE(`[{"functionCall":{"id":"x","name":"read","args":{}}}]`, "STOP"), &auditPersister{})
		if status != 502 {
			t.Errorf("tool_choice=none still accepted tool result: status=%d", status)
		}
	})
	t.Run("OrphanResultRejected", func(t *testing.T) {
		body := `{"model":"m","messages":[{"role":"user","content":[{"type":"tool_result","tool_use_id":"missing","content":"output"}]}]}`
		_, tr, status := auditRun(t, provider.Anthropic, body, auditSSE(`[{"text":"ok"}]`, "STOP"), &auditPersister{})
		if status != 400 || tr.calls != 0 {
			t.Errorf("orphan result sent upstream: status=%d calls=%d genericName=%t", status, tr.calls, strings.Contains(string(tr.body), `"name":"tool"`))
		}
	})
	t.Run("UserTextNotRelabeledModel", func(t *testing.T) {
		req := &Request{Messages: []Message{{Role: "assistant", Parts: []Part{{Kind: ToolCall, ID: "c", Name: "read", Args: json.RawMessage(`{}`)}}}, {Role: "user", Parts: []Part{{Kind: ToolResult, CallID: "c", Text: "out"}, {Kind: Text, Text: "NEW USER INSTRUCTION"}}}}}
		var env struct {
			Request struct {
				Contents []struct {
					Role  string
					Parts []struct{ Text string }
				}
			}
		}
		if err := json.Unmarshal(buildCodeAssist(req, "gemini-3.8-flash-high", "antigravity"), &env); err != nil {
			t.Fatal(err)
		}
		for _, c := range env.Request.Contents {
			for _, p := range c.Parts {
				if p.Text == "NEW USER INSTRUCTION" && c.Role != "user" {
					t.Errorf("user text emitted as role=%s", c.Role)
				}
			}
		}
	})
	t.Run("BoundArgsTamperDetected", func(t *testing.T) {
		defaultToolBindingStore.clear()
		defer defaultToolBindingStore.clear()
		b := AntigravityToolBinding{NativeID: "n", NativeName: "read", NativeArgs: json.RawMessage(`{"path":"safe"}`), ClientID: "c", ClientName: "read", ClientArgs: json.RawMessage(`{"path":"safe"}`)}
		if err := defaultToolBindingStore.Bind(b); err != nil {
			t.Fatal(err)
		}
		req := &Request{Messages: []Message{{Role: "assistant", Parts: []Part{{Kind: ToolCall, ID: "c", Name: "read", Args: json.RawMessage(`{"path":"edited"}`)}}}}}
		if err := verifyCommittedHistory(antigravitySessionScope{}, []canonicalPart{{Kind: ToolCall, CallID: "n", Name: "read", Args: `{"path":"safe"}`}}, req); err == nil {
			t.Error("edited client args overwritten by stored NativeArgs before comparison; tampering accepted")
		}
	})
	t.Run("BindingCollisionRejected", func(t *testing.T) {
		defaultToolBindingStore.clear()
		defer defaultToolBindingStore.clear()
		a := AntigravityToolBinding{NativeID: "n1", NativeName: "read", NativeArgs: json.RawMessage(`{}`), ClientID: "shared", ClientName: "read"}
		b := a
		b.NativeID = "n2"
		if err := defaultToolBindingStore.Bind(a); err != nil {
			t.Fatal(err)
		}
		if err := defaultToolBindingStore.Bind(b); err == nil {
			t.Error("duplicate client ID silently replaces previous binding")
		}
	})
	t.Run("DiskNamespacesCannotCollide", func(t *testing.T) {
		p := newFileAntigravityPersister(t.TempDir())
		a := antigravitySessionScope{Caller: "unknown", Account: "user.a", Project: "p", Model: "m"}
		b := a
		b.Account = "user_a"
		if err := p.CommitRound(a, "s", []canonicalPart{{Kind: Text, Text: "A"}}, nil); err != nil {
			t.Fatal(err)
		}
		if _, err := p.LoadRound(b, "s"); err == nil {
			t.Error("different account loaded A's disk record due to lossy filename and no scope check")
		}
	})
	t.Run("ProductionCommitRejectsStaleOverwrite", func(t *testing.T) {
		p := newFileAntigravityPersister(t.TempDir())
		s := antigravitySessionScope{Caller: "unknown", Account: "u", Project: "p", Model: "m"}
		if err := p.CommitRoundWithRevision(s, "s", []canonicalPart{{Kind: Text, Text: "new"}}, nil, 2); err != nil {
			t.Fatal(err)
		}
		if err := p.CommitRound(s, "s", []canonicalPart{{Kind: Text, Text: "stale"}}, nil); err == nil {
			t.Error("production CommitRound(revision=0) bypassed revision conflict and overwrote revision 2")
		}
	})
	t.Run("RollbackPreservesPreviousBinding", func(t *testing.T) {
		defaultToolBindingStore.clear()
		defer defaultToolBindingStore.clear()
		b := AntigravityToolBinding{NativeID: "n1", NativeName: "read", NativeArgs: json.RawMessage(`{}`), ClientID: "c", ClientName: "read"}
		if err := defaultToolBindingStore.Bind(b); err != nil {
			t.Fatal(err)
		}
		b.NativeID = "n2"
		err := commitAntigravityRoundWithPersister(antigravitySessionScope{}, "s", nil, []AntigravityToolBinding{b}, &auditPersister{fail: errors.New("synthetic")})
		if err == nil {
			t.Fatal("expected failure")
		}
		if _, ok := defaultToolBindingStore.LookupByClientID("c"); !ok {
			t.Error("rollback deleted the previously committed binding rather than restoring it")
		}
	})
	t.Run("UsageZeroCorrectionApplied", func(t *testing.T) {
		d := &codeAssistDecoder{}
		var c collector
		frames := []string{`{"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"cachedContentTokenCount":4,"thoughtsTokenCount":3}}}`, `{"response":{"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"cachedContentTokenCount":0,"thoughtsTokenCount":0}}}`}
		for _, f := range frames {
			if err := d.decode(f, c.add); err != nil {
				t.Fatal(err)
			}
		}
		u := c.finish().Usage
		if u.CacheRead != 0 || u.Reasoning != 0 {
			t.Errorf("final zero snapshot not applied: %+v", u)
		}
	})
	t.Run("ExplicitThinkingBudgetPreserved", func(t *testing.T) {
		r, err := parseAnthropic([]byte(`{"model":"m","max_tokens":32000,"thinking":{"type":"enabled","budget_tokens":2000},"messages":[{"role":"user","content":"hi"}]}`))
		if err != nil {
			t.Fatal(err)
		}
		wire := string(buildCodeAssist(r, "claude-sonnet-4-6", "antigravity"))
		if !strings.Contains(wire, `"thinkingBudget":2000`) {
			t.Errorf("explicit 2000 budget not preserved: %s", wire)
		}
	})
	t.Run("StrictCannotBeDisabledByCallerHeader", func(t *testing.T) {
		old := GetAntigravityMode()
		SetAntigravityMode(AntigravityModeStrict)
		defer SetAntigravityMode(old)
		h := http.Header{}
		h.Set("X-Antigravity-Mode", "off")
		if effectiveAntigravityMode(h) != AntigravityModeStrict {
			t.Error("caller header downgrades server strict to off")
		}
	})
}
