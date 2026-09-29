package gateway

import "testing"

// TestAntigravityBDD is the unified entrypoint for all Antigravity BDD scenarios.
// Subtests are strictly grouped as B01_..., B02_..., etc.
func TestAntigravityBDD(t *testing.T) {
	t.Run("B01_FixtureValidation", testB01FixtureValidation)
	t.Run("B01_JSONEqualExact", testB01JSONEqualExact)
	t.Run("B02_AgentBridge", testB02AgentBridge)
	t.Run("B03_OutputTokenLimit", testB03OutputTokenLimit)
	t.Run("B03b_ToolArgsValidation", testB03bToolArgsValidation)
	t.Run("B03c_ImageValidation", testB03cImageValidation)
	t.Run("B03d_BlockTypeValidation", testB03dBlockTypeValidation)
	t.Run("B03e_ToolContract", testB03eToolContract)
	t.Run("B03f_ThinkingDefaults", testB03fThinkingDefaults)
	t.Run("B04_ToolCatalog", testB04ToolCatalog)
}
