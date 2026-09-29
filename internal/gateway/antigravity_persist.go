package gateway

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

// persistedRound represents the serialized disk state of an Antigravity round.
type persistedRound struct {
	Scope     antigravitySessionScope  `json:"scope"`
	Session   string                   `json:"session"`
	Canonical []canonicalPart          `json:"canonical"`
	Bindings  []AntigravityToolBinding `json:"bindings"`
}

// antigravityPersister defines the atomic persistence contract for an Antigravity round.
type antigravityPersister interface {
	CommitRound(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding) error
	LoadRound(scope antigravitySessionScope, sessionRef string) (*persistedRound, error)
}

type fileAntigravityPersister struct {
	sync.Mutex
	baseDir string
}

func newFileAntigravityPersister(baseDir string) *fileAntigravityPersister {
	if baseDir == "" {
		baseDir = filepath.Join(os.Getenv("XDG_CACHE_HOME"), "antigravity_rounds")
	}
	_ = os.MkdirAll(baseDir, 0755)
	return &fileAntigravityPersister{baseDir: baseDir}
}

func (p *fileAntigravityPersister) CommitRound(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding) error {
	p.Lock()
	defer p.Unlock()

	key := buildSessionLedgerKey(scope, sessionRef)
	if key == "" {
		return nil
	}

	payload := persistedRound{
		Scope:     scope,
		Session:   sessionRef,
		Canonical: canonical,
		Bindings:  bindings,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	// Atomic write using temp file and rename
	tmpFile, err := os.CreateTemp(p.baseDir, "round_*.tmp")
	if err != nil {
		return fmt.Errorf("failed to create round temp file: %w", err)
	}
	tmpName := tmpFile.Name()
	defer os.Remove(tmpName)

	if _, err := tmpFile.Write(data); err != nil {
		tmpFile.Close()
		return fmt.Errorf("failed to write round temp file: %w", err)
	}
	if err := tmpFile.Sync(); err != nil {
		tmpFile.Close()
		return fmt.Errorf("failed to fsync round temp file: %w", err)
	}
	tmpFile.Close()

	destName := filepath.Join(p.baseDir, sanitizeFilename(key)+".json")
	if err := os.Rename(tmpName, destName); err != nil {
		return fmt.Errorf("failed to atomically commit round file: %w", err)
	}
	return nil
}

func (p *fileAntigravityPersister) LoadRound(scope antigravitySessionScope, sessionRef string) (*persistedRound, error) {
	p.Lock()
	defer p.Unlock()

	key := buildSessionLedgerKey(scope, sessionRef)
	if key == "" {
		return nil, os.ErrNotExist
	}

	destName := filepath.Join(p.baseDir, sanitizeFilename(key)+".json")
	data, err := os.ReadFile(destName)
	if err != nil {
		return nil, err
	}

	var round persistedRound
	if err := json.Unmarshal(data, &round); err != nil {
		return nil, fmt.Errorf("corrupted session record: invalid json in %s: %w", destName, err)
	}
	return &round, nil
}

func sanitizeFilename(s string) string {
	var b []byte
	for i := 0; i < len(s); i++ {
		c := s[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' {
			b = append(b, c)
		} else {
			b = append(b, '_')
		}
	}
	return string(b)
}

var currentAntigravityPersister antigravityPersister = newFileAntigravityPersister("")

// commitAntigravityRoundWithPersister atomically stages bindings, commits to persister, and rolls back on failure.
func commitAntigravityRoundWithPersister(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding, persister antigravityPersister) error {
	if persister == nil {
		persister = currentAntigravityPersister
	}

	// 1. Stage tool bindings
	if len(bindings) > 0 {
		if err := defaultToolBindingStore.Bind(bindings...); err != nil {
			return err
		}
	}

	// 2. Commit round to persistent storage
	if err := persister.CommitRound(scope, sessionRef, canonical, bindings); err != nil {
		// Roll back staged bindings on storage failure (atomic rollback)
		if len(bindings) > 0 {
			defaultToolBindingStore.Lock()
			for _, b := range bindings {
				delete(defaultToolBindingStore.byClient, b.ClientID)
				delete(defaultToolBindingStore.byNative, b.NativeID)
				delete(defaultToolBindingStore.byClientName, b.ClientName)
			}
			defaultToolBindingStore.Unlock()
		}
		return err
	}

	// 3. Update in-memory session ledger
	if sessionRef != "" {
		defaultAntigravityLedger.Store(scope, sessionRef, &antigravityLedgerRecord{
			SessionID: sessionRef,
			Scope:     scope,
			Data: map[string]any{
				"history": canonical,
			},
		})
	}
	return nil
}

func commitAntigravityRound(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding) error {
	return commitAntigravityRoundWithPersister(scope, sessionRef, canonical, bindings, currentAntigravityPersister)
}
