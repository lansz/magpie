package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB04ToolCatalog covers B04: Antigravity functionDeclarations come only from
// the tools the client declared this turn. Magpie must not fill in undeclared
// tools, and must not treat a matching name as license to enable a native one.
func testB04ToolCatalog(t *testing.T) {
	t.Run("BuilderOnlyClientTools", testB04BuilderOnlyClientTools)
	t.Run("WireDeclaredToolsOnly", testB04WireDeclaredToolsOnly)
	t.Run("NoToolsOmitsCatalog", testB04NoToolsOmitsCatalog)
	t.Run("WebSearchDoesNotInjectTool", testB04WebSearchDoesNotInjectTool)
}

// functionDeclarationNames reads the upstream Code Assist envelope and returns
// the functionDeclarations names in order. Absent tools yields nil.
func functionDeclarationNames(t *testing.T, body []byte) []string {
	t.Helper()
	var env struct {
		Request struct {
			Tools []struct {
				FunctionDeclarations []struct {
					Name string `json:"name"`
				} `json:"functionDeclarations"`
			} `json:"tools"`
		} `json:"request"`
	}
	if err := json.Unmarshal(body, &env); err != nil {
		t.Fatalf("unmarshal upstream envelope: %v\nbody: %s", err, body)
	}
	if len(env.Request.Tools) == 0 {
		return nil
	}
	var names []string
	for _, block := range env.Request.Tools {
		for _, d := range block.FunctionDeclarations {
			names = append(names, d.Name)
		}
	}
	return names
}

// testB04BuilderOnlyClientTools: buildCodeAssist's catalog is exactly r.Tools,
// including names that collide with Antigravity natives and excluding any filler.
func testB04BuilderOnlyClientTools(t *testing.T) {
	cases := []struct {
		name string
		body string
		want []string
	}{
		{
			name: "CustomOnly",
			body: `{"model":"m","max_tokens":100,"tools":[{"name":"user_calculator","input_schema":{"type":"object"}}],"messages":[{"role":"user","content":"hi"}]}`,
			want: []string{"user_calculator"},
		},
		{
			name: "NativeLookingNamesStayClientOwned",
			body: `{"model":"m","max_tokens":100,"tools":[
				{"name":"view_file","input_schema":{"type":"object"}},
				{"name":"search_web","input_schema":{"type":"object"}},
				{"name":"WebSearch","input_schema":{"type":"object"}}
			],"messages":[{"role":"user","content":"hi"}]}`,
			want: []string{"view_file", "search_web", "WebSearch"},
		},
		{
			name: "EmptyTools",
			body: `{"model":"m","max_tokens":100,"tools":[],"messages":[{"role":"user","content":"hi"}]}`,
			want: nil,
		},
		{
			name: "TypedWebSearchAloneIsNotADeclaration",
			body: `{"model":"m","max_tokens":100,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}],"messages":[{"role":"user","content":"hi"}]}`,
			want: nil,
		},
	}
	for _, model := range antigravityModels {
		for _, tc := range cases {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				r, err := parse(provider.Anthropic, []byte(tc.body))
				if err != nil {
					t.Fatalf("parse: %v", err)
				}
				got := functionDeclarationNames(t, buildCodeAssist(r, model, "antigravity"))
				if strings.Join(got, ",") != strings.Join(tc.want, ",") {
					t.Fatalf("functionDeclarations = %v, want %v", got, tc.want)
				}
			})
		}
	}
}

// testB04WireDeclaredToolsOnly: translate keeps the client's custom tools and
// adds none, across Messages / Chat / Responses.
func testB04WireDeclaredToolsOnly(t *testing.T) {
	bodies := []struct {
		name string
		from provider.Protocol
		body string
	}{
		{"Messages", provider.Anthropic,
			`{"model":"m","max_tokens":100,"tools":[{"name":"read","description":"r","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}],"messages":[{"role":"user","content":"hi"}]}`},
		{"Chat", provider.Chat,
			`{"model":"m","tools":[{"type":"function","function":{"name":"read","description":"r","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}],"messages":[{"role":"user","content":"hi"}]}`},
		{"Responses", provider.Responses,
			`{"model":"m","tools":[{"type":"function","name":"read","description":"r","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}],"input":[{"role":"user","content":"hi"}]}`},
	}
	for _, model := range antigravityModels {
		for _, tc := range bodies {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				status, msg, rec, tr := runAntigravity(t, tc.from, model, tc.body)
				if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
					t.Fatalf("status=%d/%d msg=%q body=%s", status, rec.Code, msg, rec.Body.String())
				}
				if atomic.LoadInt64(&tr.callCount) != 1 {
					t.Fatalf("upstream calls = %d, want 1", tr.callCount)
				}
				got := functionDeclarationNames(t, tr.capturedBody)
				if strings.Join(got, ",") != "read" {
					t.Fatalf("functionDeclarations = %v, want [read]", got)
				}
			})
		}
	}
}

// testB04NoToolsOmitsCatalog: a request with no tools must not grow a catalog.
func testB04NoToolsOmitsCatalog(t *testing.T) {
	body := `{"model":"m","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`
	for _, model := range antigravityModels {
		t.Run(model, func(t *testing.T) {
			status, msg, rec, tr := runAntigravity(t, provider.Anthropic, model, body)
			if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
				t.Fatalf("status=%d/%d msg=%q", status, rec.Code, msg)
			}
			got := functionDeclarationNames(t, tr.capturedBody)
			if len(got) != 0 {
				t.Fatalf("functionDeclarations = %v, want none", got)
			}
		})
	}
}

// testB04WebSearchDoesNotInjectTool: even when magpie has a searcher, Antigravity
// must not receive an injected web_search the client never declared as a tool.
func testB04WebSearchDoesNotInjectTool(t *testing.T) {
	t.Setenv("XDG_CONFIG_HOME", t.TempDir())
	t.Setenv("XDG_CACHE_HOME", t.TempDir())
	t.Setenv("HOME", t.TempDir())

	search := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			io.WriteString(w, `{"data":[{"id":"claude-haiku-4-5"}]}`)
		default:
			http.Error(w, `{"error":{"message":"search must not run for this B04 case"}}`, 500)
		}
	}))
	t.Cleanup(search.Close)

	hosts := searchHosts[provider.Anthropic]
	searchHosts[provider.Anthropic] = append(hosts, provider.HostOf(search.URL))
	t.Cleanup(func() { searchHosts[provider.Anthropic] = hosts })

	if err := provider.Save(provider.Provider{ID: "b04-srch", Name: "B04Search", Key: "k", Anthropic: search.URL}); err != nil {
		t.Fatal(err)
	}
	p, err := provider.Find("b04-srch")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := p.Fetch(context.Background()); err != nil {
		t.Fatal(err)
	}
	if sp, _, ok := searcher(); !ok || sp.ID != "b04-srch" {
		t.Fatalf("searcher = %s %v, want b04-srch", sp.ID, ok)
	}

	bodies := []struct {
		name string
		body string
		want []string
	}{
		{
			name: "TypedWebSearchWithClientTool",
			body: `{"model":"m","max_tokens":100,"tools":[
				{"name":"read","input_schema":{"type":"object"}},
				{"type":"web_search_20250305","name":"web_search","max_uses":3}
			],"messages":[{"role":"user","content":"hi"}]}`,
			want: []string{"read"},
		},
		{
			name: "TypedWebSearchAlone",
			body: `{"model":"m","max_tokens":100,"tools":[{"type":"web_search_20250305","name":"web_search","max_uses":3}],"messages":[{"role":"user","content":"hi"}]}`,
			want: nil,
		},
	}

	for _, model := range antigravityModels {
		for _, tc := range bodies {
			t.Run(model+"/"+tc.name, func(t *testing.T) {
				s := New()
				tr := &mockCaptureTransport{sseReply: sse(`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}]}}`)}
				s.client = &http.Client{Transport: tr}
				target := provider.Provider{
					ID: "ag", Name: "AG", Key: "k", Models: []string{model},
					Account: &provider.Account{Agent: "antigravity"},
				}
				rec := httptest.NewRecorder()
				httpReq := httptest.NewRequest("POST", "/dummy", bytes.NewReader([]byte(tc.body)))
				var u Usage
				status, msg := s.translate(rec, httpReq, target, provider.Anthropic, provider.CodeAssist, model, []byte(tc.body), &u)
				if status != http.StatusOK || rec.Code != http.StatusOK || msg != "" {
					t.Fatalf("status=%d/%d msg=%q body=%s", status, rec.Code, msg, rec.Body.String())
				}
				if atomic.LoadInt64(&tr.callCount) != 1 {
					t.Fatalf("upstream calls = %d, want 1 (search must not inject a second round)", tr.callCount)
				}
				got := functionDeclarationNames(t, tr.capturedBody)
				if strings.Join(got, ",") != strings.Join(tc.want, ",") {
					t.Fatalf("functionDeclarations = %v, want %v; body=%s", got, tc.want, tr.capturedBody)
				}
				for _, n := range got {
					if n == "web_search" || strings.HasPrefix(n, "magpie_") {
						t.Fatalf("injected search tool %q appeared in catalog", n)
					}
				}
			})
		}
	}
}
