package gateway

import (
	"github.com/yetone/magpie/internal/provider"
)

// bridgeRequest wraps Request with protocol translation and client profile metadata.
type bridgeRequest struct {
	*Request
	sourceProto   provider.Protocol
	clientProfile string
}

// newBridgeRequest returns a bridgeRequest wrapping the given Request with source protocol and client profile.
func newBridgeRequest(req *Request, proto provider.Protocol, profile string) *bridgeRequest {
	if profile == "" {
		profile = "unknown"
	}
	return &bridgeRequest{
		Request:       req,
		sourceProto:   proto,
		clientProfile: profile,
	}
}
