package provider

import (
	"strings"
	"testing"
)

// TestAntigravityProviderAudit checks real envelope and account constructors without credentials or network calls.
func TestAntigravityProviderAudit(t *testing.T) {
	t.Run("FinalEnvelopePreservesLargeInteger", func(t *testing.T) {
		body := []byte(`{"model":"gemini-3.8-flash-high","request":{"contents":[{"role":"model","parts":[{"functionCall":{"id":"synthetic","name":"read","args":{"business_id":9007199254740993}}}]}]}}`)
		out, _, err := codeAssistEnvelope("antigravity", body, "synthetic-project")
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(out), `9007199254740993`) {
			t.Errorf("final signer envelope rounded business integer: %s", out)
		}
	})
	t.Run("RealAccountExposesResolvedScopeProject", func(t *testing.T) {
		p := googleProvider(googleAccount{app: googleApp{agent: "antigravity"}, user: "synthetic-user", auth: googleAuth{Project: "synthetic-project"}}, "")
		if p.Account.Project != "synthetic-project" {
			t.Errorf("real constructor Account.Project=%q; gateway cannot scope by actual project", p.Account.Project)
		}
	})
}
