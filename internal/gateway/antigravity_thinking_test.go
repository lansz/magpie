package gateway

import (
	"encoding/json"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// testB03fThinkingDefaults covers Antigravity thinking: the official client's defaults when the
// caller doesn't say, and no thinkingConfig when the caller turns thinking off.
func testB03fThinkingDefaults(t *testing.T) {
	t.Run("DefaultsWhenUnspecified", testB03fDefaultsWhenUnspecified)
	t.Run("DisabledSendsNone", testB03fDisabledSendsNone)
	t.Run("ClaudeDefaultNotFittingMaxTokensOmitted", testB03fClaudeDefaultNotFitting)
	t.Run("ExplicitClaudeThinkingNotDroppedByName", testB03fExplicitClaudeThinking)
	t.Run("NonTargetUnchanged", testB03fNonTargetUnchanged)
}

// thinkingWire returns the generationConfig.thinkingConfig built for body, as JSON, or "" when absent.
func thinkingWire(t *testing.T, from provider.Protocol, body, model, agent string) string {
	t.Helper()
	r, err := parse(from, []byte(body))
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	var env struct {
		Request struct {
			GenerationConfig map[string]json.RawMessage `json:"generationConfig"`
		} `json:"request"`
	}
	if err := json.Unmarshal(buildCodeAssist(r, model, agent), &env); err != nil {
		t.Fatalf("unmarshal wire: %v", err)
	}
	return string(env.Request.GenerationConfig["thinkingConfig"])
}

// testB03fDefaultsWhenUnspecified: the budgets captured from the official client for each measured model.
func testB03fDefaultsWhenUnspecified(t *testing.T) {
	cases := []struct {
		model, want string
	}{
		{"gemini-3.8-flash-high", `{"includeThoughts":true,"thinkingBudget":-1}`},
		{"gemini-3.8-flash-low", `{"includeThoughts":true,"thinkingBudget":1000}`},
		{"claude-sonnet-4-6", `{"includeThoughts":true,"thinkingBudget":1024}`},
	}
	for _, tc := range cases {
		t.Run("Messages/"+tc.model, func(t *testing.T) {
			body := `{"model":"m","max_tokens":32000,"messages":[{"role":"user","content":"hi"}]}`
			if got := thinkingWire(t, provider.Anthropic, body, tc.model, "antigravity"); got != tc.want {
				t.Errorf("thinkingConfig = %s, want %s", got, tc.want)
			}
		})
		t.Run("Chat/"+tc.model, func(t *testing.T) {
			body := `{"model":"m","messages":[{"role":"user","content":"hi"}]}`
			if got := thinkingWire(t, provider.Chat, body, tc.model, "antigravity"); got != tc.want {
				t.Errorf("thinkingConfig = %s, want %s", got, tc.want)
			}
		})
	}
}

// testB03fDisabledSendsNone: an explicit thinking.type=disabled is kept apart from absent and sends no thinkingConfig.
func testB03fDisabledSendsNone(t *testing.T) {
	for _, model := range []string{"gemini-3.8-flash-high", "gemini-3.8-flash-low", "claude-sonnet-4-6"} {
		t.Run(model, func(t *testing.T) {
			body := `{"model":"m","max_tokens":32000,"thinking":{"type":"disabled"},"messages":[{"role":"user","content":"hi"}]}`
			if got := thinkingWire(t, provider.Anthropic, body, model, "antigravity"); got != "" {
				t.Errorf("thinkingConfig = %s, want none", got)
			}
		})
	}
}

// testB03fClaudeDefaultNotFitting: Claude's budget must stay below max_tokens, so a default that can't fit is left out.
func testB03fClaudeDefaultNotFitting(t *testing.T) {
	body := `{"model":"m","max_tokens":1000,"messages":[{"role":"user","content":"hi"}]}`
	if got := thinkingWire(t, provider.Anthropic, body, "claude-sonnet-4-6", "antigravity"); got != "" {
		t.Errorf("thinkingConfig = %s, want none", got)
	}
}

// testB03fExplicitClaudeThinking: asked-for thinking reaches claude-sonnet-4-6 though its name lacks "thinking".
func testB03fExplicitClaudeThinking(t *testing.T) {
	body := `{"model":"m","max_tokens":32000,"thinking":{"type":"enabled","budget_tokens":8000},"messages":[{"role":"user","content":"hi"}]}`
	got := thinkingWire(t, provider.Anthropic, body, "claude-sonnet-4-6", "antigravity")
	var tc struct {
		IncludeThoughts bool `json:"includeThoughts"`
		ThinkingBudget  *int `json:"thinkingBudget"`
	}
	if err := json.Unmarshal([]byte(got), &tc); err != nil {
		t.Fatalf("thinkingConfig %q: %v", got, err)
	}
	if !tc.IncludeThoughts || tc.ThinkingBudget == nil || *tc.ThinkingBudget == 1024 {
		t.Errorf("thinkingConfig = %s, want the asked-for thinking rather than none or the default", got)
	}
}

// testB03fNonTargetUnchanged: Gemini CLI requests that don't ask for thinking still send none.
func testB03fNonTargetUnchanged(t *testing.T) {
	body := `{"model":"m","max_tokens":32000,"messages":[{"role":"user","content":"hi"}]}`
	if got := thinkingWire(t, provider.Anthropic, body, "gemini-2.5-pro", "gemini"); got != "" {
		t.Errorf("thinkingConfig = %s, want none", got)
	}
}
