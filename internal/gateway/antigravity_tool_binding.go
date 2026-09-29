package gateway

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
)

// AntigravityToolBinding preserves the exact bidirectional contract between
// upstream native calls (e.g. view_file) and client-exposed tools (e.g. Read).
type AntigravityToolBinding struct {
	Scope           antigravitySessionScope `json:"scope,omitempty"`
	NativeID        string                  `json:"native_id"`
	NativeName      string                  `json:"native_name"`
	NativeArgs      json.RawMessage         `json:"native_args"`
	NativeSignature string                  `json:"native_signature,omitempty"`

	ClientID   string          `json:"client_id"`
	ClientName string          `json:"client_name"`
	ClientArgs json.RawMessage `json:"client_args,omitempty"`
}

type antigravityToolBindingStore struct {
	sync.RWMutex
	byClient     map[string]*AntigravityToolBinding // key: scopeAccount\x00clientID or clientID
	byNative     map[string]*AntigravityToolBinding // key: scopeAccount\x00nativeID or nativeID
	byClientName map[string]string                  // key: scopeAccount\x00clientName -> nativeName
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

func bindingKey(scope antigravitySessionScope, id string) string {
	if scope.Account != "" {
		return scope.Account + "\x00" + id
	}
	return id
}

// Bind atomically stores all tool bindings in the slice.
// If any binding is invalid or collides, all changes are rolled back (no partial state).
func (s *antigravityToolBindingStore) Bind(bindings ...AntigravityToolBinding) error {
	s.Lock()
	defer s.Unlock()

	batchClientSeen := make(map[string]AntigravityToolBinding)

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
		kClient := bindingKey(b.Scope, b.ClientID)
		if prev, ok := batchClientSeen[kClient]; ok {
			if prev.NativeID != b.NativeID || prev.NativeName != b.NativeName {
				return fmt.Errorf("binding collision within batch for client ID %q", b.ClientID)
			}
		}
		batchClientSeen[kClient] = b

		if existing, ok := s.byClient[kClient]; ok {
			if existing.NativeID != b.NativeID || existing.NativeName != b.NativeName {
				return fmt.Errorf("binding collision for client ID %q", b.ClientID)
			}
		}
	}

	// 2. Assignment
	for _, b := range bindings {
		entry := b
		kClient := bindingKey(b.Scope, b.ClientID)
		kNative := bindingKey(b.Scope, b.NativeID)
		kName := bindingKey(b.Scope, b.ClientName)

		s.byClient[kClient] = &entry
		s.byNative[kNative] = &entry
		s.byClientName[kName] = b.NativeName
	}

	return nil
}

func (s *antigravityToolBindingStore) Rollback(bindings ...AntigravityToolBinding) {
	s.Lock()
	defer s.Unlock()
	for _, b := range bindings {
		kClient := bindingKey(b.Scope, b.ClientID)
		kNative := bindingKey(b.Scope, b.NativeID)
		kName := bindingKey(b.Scope, b.ClientName)
		if cur, ok := s.byClient[kClient]; ok && cur.NativeID == b.NativeID {
			delete(s.byClient, kClient)
		}
		if cur, ok := s.byNative[kNative]; ok && cur.ClientID == b.ClientID {
			delete(s.byNative, kNative)
		}
		if cur, ok := s.byClientName[kName]; ok && cur == b.NativeName {
			delete(s.byClientName, kName)
		}
	}
}

func (s *antigravityToolBindingStore) LookupByClientID(clientID string) (*AntigravityToolBinding, bool) {
	return s.LookupByClientIDScoped(antigravitySessionScope{}, clientID)
}

func (s *antigravityToolBindingStore) LookupByClientIDScoped(scope antigravitySessionScope, clientID string) (*AntigravityToolBinding, bool) {
	if clientID == "" {
		return nil, false
	}
	s.RLock()
	defer s.RUnlock()
	if scope.Account != "" {
		if b, ok := s.byClient[bindingKey(scope, clientID)]; ok {
			return b, true
		}
		if b, ok := s.byClient[clientID]; ok && b.Scope.Account == "" {
			return b, true
		}
		return nil, false
	}
	if b, ok := s.byClient[clientID]; ok {
		return b, true
	}
	for _, b := range s.byClient {
		if b.ClientID == clientID {
			return b, true
		}
	}
	return nil, false
}

func (s *antigravityToolBindingStore) LookupByNativeID(nativeID string) (*AntigravityToolBinding, bool) {
	return s.LookupByNativeIDScoped(antigravitySessionScope{}, nativeID)
}

func (s *antigravityToolBindingStore) LookupByNativeIDScoped(scope antigravitySessionScope, nativeID string) (*AntigravityToolBinding, bool) {
	if nativeID == "" {
		return nil, false
	}
	s.RLock()
	defer s.RUnlock()
	if scope.Account != "" {
		if b, ok := s.byNative[bindingKey(scope, nativeID)]; ok {
			return b, true
		}
		if b, ok := s.byNative[nativeID]; ok && b.Scope.Account == "" {
			return b, true
		}
		return nil, false
	}
	if b, ok := s.byNative[nativeID]; ok {
		return b, true
	}
	for _, b := range s.byNative {
		if b.NativeID == nativeID {
			return b, true
		}
	}
	return nil, false
}

func (s *antigravityToolBindingStore) LookupNativeName(clientName string) (string, bool) {
	return s.LookupNativeNameScoped(antigravitySessionScope{}, clientName)
}

func (s *antigravityToolBindingStore) LookupNativeNameScoped(scope antigravitySessionScope, clientName string) (string, bool) {
	if clientName == "" {
		return "", false
	}
	s.RLock()
	defer s.RUnlock()
	if scope.Account != "" {
		if native, ok := s.byClientName[bindingKey(scope, clientName)]; ok {
			return native, true
		}
		if native, ok := s.byClientName[clientName]; ok {
			return native, true
		}
		return "", false
	}
	if native, ok := s.byClientName[clientName]; ok {
		return native, true
	}
	for k, native := range s.byClientName {
		if idx := strings.IndexByte(k, '\x00'); idx >= 0 && k[idx+1:] == clientName {
			return native, true
		}
	}
	return "", false
}
