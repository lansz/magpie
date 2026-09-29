package gateway

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// antigravityModels are the two model families whose tool contract was captured from the official client.
var antigravityModels = []string{"gemini-3.8-flash-high", "claude-sonnet-4-6"}

// testB03eToolContract covers the Antigravity tool wire contract evidenced by official captures:
// results travel in response.output, auto mode omits toolConfig, and a parallel ban it cannot express is refused.
func testB03eToolContract(t *testing.T) {
	t.Run("ResultUsesOutput", testB03eResultUsesOutput)
	t.Run("ToolConfigModes", testB03eToolConfigModes)
	t.Run("ParallelBanRejected", testB03eParallelBanRejected)
	t.Run("ParallelAllowedPasses", testB03eParallelAllowedPasses)
}

// codeAssistRequestOf builds the Antigravity wire request for a Messages body and returns its request object.
func codeAssistRequestOf(t *testing.T, body, model string) map[string]any {
	t.Helper()
	r, err := parseAnthropic([]byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var env struct {
		Request map[string]any `json:"request"`
	}
	if err := json.Unmarshal(buildCodeAssist(r, model, "antigravity"), &env); err != nil {
		t.Fatalf("unmarshal wire: %v", err)
	}
	return env.Request
}

// functionResponses collects every functionResponse object on the wire in order.
func functionResponses(t *testing.T, req map[string]any) []map[string]any {
	t.Helper()
	var out []map[string]any
	contents, _ := req["contents"].([]any)
	for _, c := range contents {
		parts, _ := c.(map[string]any)["parts"].([]any)
		for _, p := range parts {
			if fr, ok := p.(map[string]any)["functionResponse"].(map[string]any); ok {
				out = append(out, fr)
			}
		}
	}
	return out
}

const toolTurnBody = `{"model":"m","max_tokens":100,
	"tools":[{"name":"read","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],
	"messages":[
		{"role":"user","content":"read both"},
		{"role":"assistant","content":[
			{"type":"tool_use","id":"c1","name":"read","input":{"path":"a"}},
			{"type":"tool_use","id":"c2","name":"read","input":{"path":"b"}}]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"c1","content":"A!"},
			{"type":"tool_result","tool_use_id":"c2","content":"no such file","is_error":true}]}]}`

// testB03eResultUsesOutput: success and error results both carry the real text in response.output only.
func testB03eResultUsesOutput(t *testing.T) {
	for _, model := range antigravityModels {
		t.Run(model, func(t *testing.T) {
			frs := functionResponses(t, codeAssistRequestOf(t, toolTurnBody, model))
			if len(frs) != 2 {
				t.Fatalf("want 2 functionResponses, got %d", len(frs))
			}
			want := []struct{ id, output string }{{"c1", "A!"}, {"c2", "no such file"}}
			for i, w := range want {
				resp, _ := frs[i]["response"].(map[string]any)
				if frs[i]["id"] != w.id || frs[i]["name"] != "read" {
					t.Errorf("result %d identity = %v", i, frs[i])
				}
				if len(resp) != 1 || resp["output"] != w.output {
					t.Errorf("result %d response = %v, want only output=%q", i, resp, w.output)
				}
			}
		})
	}
}

// testB03eToolConfigModes: auto/absent omits toolConfig; any and a named tool keep their existing ANY mapping.
func testB03eToolConfigModes(t *testing.T) {
	cases := []struct {
		name, choice string
		wantConfig   string // "" means toolConfig must be absent
	}{
		{"Absent", ``, ""},
		{"Auto", `,"tool_choice":{"type":"auto"}`, ""},
		{"Any", `,"tool_choice":{"type":"any"}`, `{"functionCallingConfig":{"mode":"ANY"}}`},
		{"Named", `,"tool_choice":{"type":"tool","name":"read"}`, `{"functionCallingConfig":{"allowedFunctionNames":["read"],"mode":"ANY"}}`},
	}
	for _, model := range antigravityModels {
		for _, tc := range cases {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				body := `{"model":"m","max_tokens":100` + tc.choice + `,
					"tools":[{"name":"read","input_schema":{"type":"object"}}],
					"messages":[{"role":"user","content":"hi"}]}`
				req := codeAssistRequestOf(t, body, model)
				if _, ok := req["tools"]; !ok {
					t.Fatalf("tools missing from wire")
				}
				cfg, has := req["toolConfig"]
				if tc.wantConfig == "" {
					if has {
						t.Errorf("toolConfig should be omitted, got %v", cfg)
					}
					return
				}
				got, err := json.Marshal(cfg)
				if err != nil {
					t.Fatalf("marshal toolConfig: %v", err)
				}
				if string(got) != tc.wantConfig {
					t.Errorf("toolConfig = %s, want %s", got, tc.wantConfig)
				}
			})
		}
	}
}

// runAntigravity sends body through translate to an Antigravity provider backed by an in-memory transport.
func runAntigravity(t *testing.T, from provider.Protocol, model, body string) (int, string, *httptest.ResponseRecorder, *mockCaptureTransport) {
	t.Helper()
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	s := New()
	tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}}`)}
	s.client = &http.Client{Transport: tr}
	p := provider.Provider{ID: "ag", Name: "AG", Key: "k", Models: []string{model}, Account: &provider.Account{Agent: "antigravity"}}
	rec := httptest.NewRecorder()
	req := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(body)))
	var u Usage
	status, msg := s.translate(rec, req, p, from, provider.CodeAssist, model, []byte(body), &u)
	return status, msg, rec, tr
}

// parallelCases pair each protocol with its way of saying tools must not run in parallel.
var parallelCases = []struct {
	name  string
	from  provider.Protocol
	ban   string
	allow string
	field string
}{
	{"Messages", provider.Anthropic,
		`{"model":"m","max_tokens":100,"tool_choice":{"type":"auto","disable_parallel_tool_use":true},"tools":[{"name":"read","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","max_tokens":100,"tool_choice":{"type":"auto","disable_parallel_tool_use":false},"tools":[{"name":"read","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`,
		"tool_choice.disable_parallel_tool_use"},
	{"Chat", provider.Chat,
		`{"model":"m","parallel_tool_calls":false,"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","parallel_tool_calls":true,"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],"messages":[{"role":"user","content":"hi"}]}`,
		"parallel_tool_calls"},
	{"Responses", provider.Responses,
		`{"model":"m","parallel_tool_calls":false,"tools":[{"type":"function","name":"read","parameters":{"type":"object"}}],"input":[{"role":"user","content":"hi"}]}`,
		`{"model":"m","parallel_tool_calls":true,"tools":[{"type":"function","name":"read","parameters":{"type":"object"}}],"input":[{"role":"user","content":"hi"}]}`,
		"parallel_tool_calls"},
}

// testB03eParallelBanRejected: a ban the upstream cannot express is refused before sending, naming the field.
func testB03eParallelBanRejected(t *testing.T) {
	for _, tc := range parallelCases {
		t.Run(tc.name, func(t *testing.T) {
			status, msg, rec, tr := runAntigravity(t, tc.from, "gemini-3.8-flash-high", tc.ban)
			if calls := atomic.LoadInt64(&tr.callCount); calls != 0 {
				t.Errorf("upstream calls = %d, want 0", calls)
			}
			if status != http.StatusBadRequest || rec.Code != http.StatusBadRequest {
				t.Errorf("status = %d/%d, want 400", status, rec.Code)
			}
			var cerr struct {
				Error struct{ Message string } `json:"error"`
			}
			if err := json.Unmarshal(rec.Body.Bytes(), &cerr); err != nil {
				t.Fatalf("client error body: %v (%s)", err, rec.Body.String())
			}
			if !strings.Contains(cerr.Error.Message, tc.field) || !strings.Contains(msg, tc.field) {
				t.Errorf("error %q / %q does not name %s", cerr.Error.Message, msg, tc.field)
			}
		})
	}
}

// testB03eParallelAllowedPasses: allowing parallel calls is the upstream's own behavior and is sent through.
func testB03eParallelAllowedPasses(t *testing.T) {
	for _, tc := range parallelCases {
		t.Run(tc.name, func(t *testing.T) {
			status, msg, rec, tr := runAntigravity(t, tc.from, "gemini-3.8-flash-high", tc.allow)
			if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
				t.Fatalf("status = %d/%d, msg %q", status, rec.Code, msg)
			}
			if calls := atomic.LoadInt64(&tr.callCount); calls != 1 {
				t.Errorf("upstream calls = %d, want 1", calls)
			}
		})
	}
}
