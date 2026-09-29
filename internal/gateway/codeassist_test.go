package gateway

import (
	"encoding/json"
	"strings"
	"testing"
)

func codeAssistRequest(t *testing.T) *Request {
	t.Helper()
	r, err := parseAnthropic([]byte(`{"model":"x","max_tokens":2000,"system":"be brief","stream":true,
		"thinking":{"type":"enabled","budget_tokens":4000},
		"tools":[{"name":"read","description":"read a file","input_schema":{"$schema":"http://json-schema.org/draft-07/schema#","type":"object",
			"properties":{"path":{"type":["string","null"],"format":"uri"},"mode":{"const":"r"},"opts":{"$ref":"#/$defs/Opts"},
			"n":{"anyOf":[{"type":"integer"},{"type":"null"}]}},"required":["path","gone"],"additionalProperties":false,
			"$defs":{"Opts":{"type":"object","properties":{"deep":{"type":"boolean","default":false}}}}}}],
		"messages":[
			{"role":"user","content":"read a"},
			{"role":"assistant","content":[{"type":"thinking","thinking":"hm","signature":"sig-from-claude"},
				{"type":"text","text":"ok"},{"type":"tool_use","id":"toolu_01:x","name":"read","input":{"path":"a"}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_01:x","content":"A!"}]},
			{"role":"user","content":"and?"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	return r
}

func TestBuildCodeAssistForGemini(t *testing.T) {
	var env struct {
		Model   string         `json:"model"`
		Request map[string]any `json:"request"`
	}
	json.Unmarshal(buildCodeAssist(codeAssistRequest(t), "gemini-2.5-pro", "gemini"), &env)
	req := env.Request
	if env.Model != "gemini-2.5-pro" {
		t.Errorf("model %q", env.Model)
	}
	b, _ := json.Marshal(req)
	s := string(b)
	for _, want := range []string{
		`"systemInstruction":{"parts":[{"text":"be brief"}],"role":"user"}`,
		`"functionCall":{"args":{"path":"a"},"id":"toolu_01:x","name":"read"},"thoughtSignature":"skip_thought_signature_validator"`,
		`"functionResponse":{"id":"toolu_01:x","name":"read","response":{"output":"A!"}}`,
		`"parametersJsonSchema":{`,
		`"thinkingConfig":{"includeThoughts":true,"thinkingBudget":4096}`,
		`"maxOutputTokens":2000`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %s in\n%s", want, s)
		}
	}
	if strings.Contains(s, "sig-from-claude") || strings.Contains(s, `"hm"`) {
		t.Error("thinking was sent back")
	}
	// the tool result and the next question are one user turn
	contents := req["contents"].([]any)
	if len(contents) != 3 || len(contents[2].(map[string]any)["parts"].([]any)) != 2 {
		t.Errorf("contents = %v", contents)
	}
}

func TestBuildCodeAssistForAntigravity(t *testing.T) {
	var env struct {
		Request map[string]any `json:"request"`
	}
	json.Unmarshal(buildCodeAssist(codeAssistRequest(t), "claude-sonnet-4-6", "antigravity"), &env)
	b, _ := json.Marshal(env.Request)
	s := string(b)
	for _, want := range []string{
		`"id":"toolu_01_x"`,
		`"response":{"output":"A!"}`,
		`"maxOutputTokens":2000`,
	} {
		if !strings.Contains(s, want) {
			t.Errorf("missing %s in\n%s", want, s)
		}
	}
	// Antigravity's Sonnet thinks as asked, its budget kept below max_tokens
	if !strings.Contains(s, `"thinkingConfig":{"includeThoughts":true,"thinkingBudget":1999}`) {
		t.Errorf("thinking not asked of Sonnet as the client wanted in\n%s", s)
	}
	// the official client sends auto tool choice without a toolConfig
	if strings.Contains(s, "toolConfig") {
		t.Error("toolConfig sent for auto tool choice")
	}
	decl := env.Request["tools"].([]any)[0].(map[string]any)["functionDeclarations"].([]any)[0].(map[string]any)
	params, _ := json.Marshal(decl["parameters"])
	want := `{"additionalProperties":false,"properties":{"mode":{"enum":["r"]},"n":{"nullable":true,"type":"integer"},"opts":{"properties":{"deep":{"type":"boolean"}},"type":"object"},"path":{"nullable":true,"type":"string"}},"required":["path"],"type":"object"}`
	if string(params) != want {
		t.Errorf("parameters =\n%s\nwant\n%s", params, want)
	}

	// a Gemini model on Antigravity preserves explicit maxOutputTokens, and a thinking
	// Claude has room past its budget
	json.Unmarshal(buildCodeAssist(codeAssistRequest(t), "gemini-3-flash", "antigravity"), &env)
	if gen, _ := env.Request["generationConfig"].(map[string]any); gen["maxOutputTokens"] != float64(2000) {
		t.Errorf("gen = %v, want maxOutputTokens 2000", gen)
	}
	r := codeAssistRequest(t)
	r.MaxTokens = 0
	env.Request = nil
	json.Unmarshal(buildCodeAssist(r, "claude-opus-4-6-thinking", "antigravity"), &env)
	gen := env.Request["generationConfig"].(map[string]any)
	tc := gen["thinkingConfig"].(map[string]any)
	if tc["thinkingBudget"].(float64) >= gen["maxOutputTokens"].(float64) {
		t.Errorf("gen = %v", gen)
	}
}

func TestCodeAssistDecoder(t *testing.T) {
	var got []Event
	d := &codeAssistDecoder{}
	for _, line := range []string{
		`{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"let me see","thought":true}]}}],"modelVersion":"gemini-2.5-pro","responseId":"r1"},"traceId":"t"}`,
		`{"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"Reading."}]}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":3}}}`,
		`{"response":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"read","args":{"path":"a"}},"thoughtSignature":"abc"}]},"finishReason":"STOP"}],
			"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":10,"thoughtsTokenCount":20,"cachedContentTokenCount":40}}}`,
	} {
		d.decode(line, func(ev Event) { got = append(got, ev) })
	}
	kinds := []EventKind{KStart, KThink, KText, KToolStart, KToolArgs, KSig, KStop, KUsage}
	if len(got) != len(kinds) {
		t.Fatalf("events = %+v", got)
	}
	for i, k := range kinds {
		if got[i].Kind != k {
			t.Fatalf("event %d = %+v, want kind %d", i, got[i], k)
		}
	}
	if got[0].Model != "gemini-2.5-pro" || got[3].Name != "read" || got[3].ID == "" || got[4].Text != `{"path":"a"}` {
		t.Errorf("events = %+v", got)
	}
	if got[6].Stop != "tool" {
		t.Errorf("stop = %q", got[6].Stop)
	}
	if u := got[7].Usage; u != (Usage{Input: 60, CacheRead: 40, Output: 30, Reasoning: 20}) {
		t.Errorf("usage = %+v", u)
	}

	var errs []Event
	(&codeAssistDecoder{}).decode(`{"error":{"code":429,"message":"quota"}}`, func(ev Event) { errs = append(errs, ev) })
	if len(errs) != 1 || errs[0].Kind != KError || errs[0].Text != "quota" {
		t.Errorf("error events = %+v", errs)
	}
}
