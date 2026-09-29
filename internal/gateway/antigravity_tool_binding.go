package gateway

import (
	"encoding/json"
	"fmt"
	"sync"
)

// AntigravityToolBinding preserves the exact bidirectional contract between
// upstream native calls (e.g. view_file) and client-exposed tools (e.g. Read).
type AntigravityToolBinding struct {
	NativeID        string          `json:"native_id"`
	NativeName      string          `json:"native_name"`
	NativeArgs      json.RawMessage `json:"native_args"`
	NativeSignature string          `json:"native_signature,omitempty"`

	ClientID   string          `json:"client_id"`
	ClientName string          `json:"client_name"`
	ClientArgs json.RawMessage `json:"client_args,omitempty"`
}

type antigravityToolBindingStore struct {
	sync.RWMutex
	byClient     map[string]*AntigravityToolBinding // key: clientID
	byNative     map[string]*AntigravityToolBinding // key: nativeID
	byClientName map[string]string                  // key: clientName -> nativeName
}

var defaultToolBindingStore = &antigravityToolBindingStore{
	byClient:     make(map[string]*AntigravityToolBinding),
	byNative:     make(map[string]*AntigravityToolBinding),
	byClientName: make(map[string]string),
}

func (s *antigravityToolBindingStore) clear() {
	s.Lock()
	s.byClient = make(map[string]*AntigravityToolBinding)
	s.byNative = make(map[string]*AntigravityToolBinding)
	s.byClientName = make(map[string]string)
	s.Unlock()
}

// Bind atomically stores all tool bindings in the slice.
// If any binding is invalid or collides, all changes are rolled back (no partial state).
func (s *antigravityToolBindingStore) Bind(bindings ...AntigravityToolBinding) error {
	s.Lock()
	defer s.Unlock()

	// 1. Validation pass
	for i, b := range bindings {
		if b.NativeID == "" || b.NativeName == "" || len(b.NativeArgs) == 0 {
			return fmt.Errorf("binding[%d] invalid native spec: ID=%q Name=%q ArgsLen=%d", i, b.NativeID, b.NativeName, len(b.NativeArgs))
		}
		if b.ClientID == "" || b.ClientName == "" {
			return fmt.Errorf("binding[%d] invalid client spec: ID=%q Name=%q", i, b.ClientID, b.ClientName)
		}
		if !json.Valid(b.NativeArgs) {
			return fmt.Errorf("binding[%d] native args is not valid JSON", i)
		}
		if existing, ok := s.byClient[b.ClientID]; ok {
			if existing.NativeID != b.NativeID || existing.NativeName != b.NativeName {
				return fmt.Errorf("binding collision for client ID %q", b.ClientID)
			}
		}
	}

	// 2. Atomic assignment
	newClient := make(map[string]*AntigravityToolBinding, len(s.byClient)+len(bindings))
	newNative := make(map[string]*AntigravityToolBinding, len(s.byNative)+len(bindings))
	newClientName := make(map[string]string, len(s.byClientName)+len(bindings))
	for k, v := range s.byClient {
		newClient[k] = v
	}
	for k, v := range s.byNative {
		newNative[k] = v
	}
	for k, v := range s.byClientName {
		newClientName[k] = v
	}

	for _, b := range bindings {
		entry := b
		newClient[b.ClientID] = &entry
		newNative[b.NativeID] = &entry
		newClientName[b.ClientName] = b.NativeName
	}

	s.byClient = newClient
	s.byNative = newNative
	s.byClientName = newClientName
	return nil
}

func (s *antigravityToolBindingStore) LookupByClientID(clientID string) (*AntigravityToolBinding, bool) {
	if clientID == "" {
		return nil, false
	}
	s.RLock()
	defer s.RUnlock()
	b, ok := s.byClient[clientID]
	return b, ok
}

func (s *antigravityToolBindingStore) LookupByNativeID(nativeID string) (*AntigravityToolBinding, bool) {
	if nativeID == "" {
		return nil, false
	}
	s.RLock()
	defer s.RUnlock()
	b, ok := s.byNative[nativeID]
	return b, ok
}

func (s *antigravityToolBindingStore) LookupNativeName(clientName string) (string, bool) {
	if clientName == "" {
		return "", false
	}
	s.RLock()
	defer s.RUnlock()
	native, ok := s.byClientName[clientName]
	return native, ok
}
