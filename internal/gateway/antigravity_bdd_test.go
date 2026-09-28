package gateway

import "testing"

// TestAntigravityBDD is the unified entrypoint for all Antigravity BDD scenarios.
// Subtests are strictly grouped as B01_..., B02_..., etc.
func TestAntigravityBDD(t *testing.T) {
	t.Run("B01_FixtureValidation", testB01FixtureValidation)
	t.Run("B01_JSONEqualExact", testB01JSONEqualExact)
}
