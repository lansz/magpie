package gateway

import (
	"crypto/sha256"
	"encoding/hex"
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
	Revision  int                      `json:"revision,omitempty"`
	Canonical []canonicalPart          `json:"canonical"`
	Bindings  []AntigravityToolBinding `json:"bindings"`
}

// antigravityPersister defines the atomic persistence contract for an Antigravity round.
type antigravityPersister interface {
	CommitRound(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding) error
	CommitRoundWithRevision(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding, revision int) error
	LoadRound(scope antigravitySessionScope, sessionRef string) (*persistedRound, error)
}

type fileAntigravityPersister struct {
	sync.Mutex
	baseDir string
}

func (p *fileAntigravityPersister) dir() string {
	if p.baseDir != "" {
		_ = os.MkdirAll(p.baseDir, 0700)
		return p.baseDir
	}
	d := filepath.Join(os.Getenv("XDG_CACHE_HOME"), "antigravity_rounds")
	if os.Getenv("XDG_CACHE_HOME") == "" {
		home, _ := os.UserHomeDir()
		d = filepath.Join(home, ".cache", "antigravity_rounds")
	}
	_ = os.MkdirAll(d, 0700)
	return d
}

func newFileAntigravityPersister(baseDir string) *fileAntigravityPersister {
	return &fileAntigravityPersister{baseDir: baseDir}
}

func (p *fileAntigravityPersister) CommitRound(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding) error {
	return p.CommitRoundWithRevision(scope, sessionRef, canonical, bindings, 0)
}

func (p *fileAntigravityPersister) CommitRoundWithRevision(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding, revision int) error {
	p.Lock()
	defer p.Unlock()

	key := buildSessionLedgerKey(scope, sessionRef)
	if key == "" {
		return nil
	}

	destName := filepath.Join(p.dir(), sanitizeFilename(key)+".json")

	// Optimistic concurrency check: stale revision cannot overwrite newer state
	if data, err := os.ReadFile(destName); err == nil {
		var existing persistedRound
		if json.Unmarshal(data, &existing) == nil && existing.Revision > 0 && existing.Revision >= revision {
			return fmt.Errorf("revision conflict on session %q: current revision %d >= proposed revision %d", sessionRef, existing.Revision, revision)
		}
	}

	payload := persistedRound{
		Scope:     scope,
		Session:   sessionRef,
		Revision:  revision,
		Canonical: canonical,
		Bindings:  bindings,
	}

	data, err := json.Marshal(payload)
	if err != nil {
		return err
	}

	// Atomic write using temp file and rename
	tmpFile, err := os.CreateTemp(p.dir(), "round_*.tmp")
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

	destName := filepath.Join(p.dir(), sanitizeFilename(key)+".json")
	data, err := os.ReadFile(destName)
	if err != nil {
		return nil, err
	}

	var round persistedRound
	if err := json.Unmarshal(data, &round); err != nil {
		return nil, fmt.Errorf("corrupted session record: invalid json in %s: %w", destName, err)
	}
	if round.Scope != scope || round.Session != sessionRef {
		return nil, os.ErrNotExist
	}
	return &round, nil
}

func sanitizeFilename(s string) string {
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

var currentAntigravityPersister antigravityPersister = newFileAntigravityPersister("")

// commitAntigravityRoundWithPersister atomically stages bindings, commits to persister, and rolls back on failure.
func commitAntigravityRoundWithPersister(scope antigravitySessionScope, sessionRef string, canonical []canonicalPart, bindings []AntigravityToolBinding, persister antigravityPersister) error {
	if persister == nil {
		persister = currentAntigravityPersister
	}

	// 1. Stage tool bindings with snapshot for atomic rollback
	var snapClient map[string]*AntigravityToolBinding
	var snapNative map[string]*AntigravityToolBinding
	var snapName map[string]string
	if len(bindings) > 0 {
		defaultToolBindingStore.RLock()
		snapClient = make(map[string]*AntigravityToolBinding, len(defaultToolBindingStore.byClient))
		for k, v := range defaultToolBindingStore.byClient {
			snapClient[k] = v
		}
		snapNative = make(map[string]*AntigravityToolBinding, len(defaultToolBindingStore.byNative))
		for k, v := range defaultToolBindingStore.byNative {
			snapNative[k] = v
		}
		snapName = make(map[string]string, len(defaultToolBindingStore.byClientName))
		for k, v := range defaultToolBindingStore.byClientName {
			snapName[k] = v
		}
		defaultToolBindingStore.RUnlock()

		if err := defaultToolBindingStore.Bind(bindings...); err != nil {
			return err
		}
	}

	rev := 1
	if sessionRef != "" {
		if rec, ok := defaultAntigravityLedger.Lookup(scope, sessionRef); ok && rec.TurnCount > 0 {
			rev = rec.TurnCount + 1
		}
		if existing, err := persister.LoadRound(scope, sessionRef); err == nil && existing != nil {
			if existing.Revision >= rev {
				rev = existing.Revision + 1
			}
		}
	}

	// 2. Commit round to persistent storage
	if err := persister.CommitRoundWithRevision(scope, sessionRef, canonical, bindings, rev); err != nil {
		// Roll back staged bindings on storage failure (restore previous snapshot)
		if len(bindings) > 0 {
			defaultToolBindingStore.Lock()
			defaultToolBindingStore.byClient = snapClient
			defaultToolBindingStore.byNative = snapNative
			defaultToolBindingStore.byClientName = snapName
			defaultToolBindingStore.Unlock()
		}
		return err
	}

	// 3. Update in-memory session ledger
	if sessionRef != "" {
		defaultAntigravityLedger.Store(scope, sessionRef, &antigravityLedgerRecord{
			SessionID: sessionRef,
			Scope:     scope,
			TurnCount: rev,
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
