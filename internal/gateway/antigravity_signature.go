package gateway

import (
	"strings"
)

// canonicalizeAntigravityParts normalizes native raw parts into canonical representations
// according to model-specific signature and thought rules.
func canonicalizeAntigravityParts(model string, raw []rawPart) []canonicalPart {
	if len(raw) == 0 {
		return nil
	}

	claude := strings.Contains(strings.ToLower(model), "claude")
	var out []canonicalPart

	if !claude {
		// Gemini model rules:
		// Trailing empty text with thoughtSignature belongs to the preceding merged text part.
		var pendingText strings.Builder
		for _, p := range raw {
			switch {
			case p.FunctionCall != nil:
				if pendingText.Len() > 0 {
					out = append(out, canonicalPart{Kind: Text, Text: pendingText.String()})
					pendingText.Reset()
				}
				out = append(out, canonicalPart{
					Kind:             ToolCall,
					CallID:           p.FunctionCall.ID,
					Name:             p.FunctionCall.Name,
					Args:             string(p.FunctionCall.Args),
					ThoughtSignature: p.ThoughtSignature,
				})
			case p.FunctionResponse != nil:
				if pendingText.Len() > 0 {
					out = append(out, canonicalPart{Kind: Text, Text: pendingText.String()})
					pendingText.Reset()
				}
				out = append(out, canonicalPart{
					Kind:   ToolResult,
					CallID: p.FunctionResponse.ID,
					Name:   p.FunctionResponse.Name,
					Args:   string(p.FunctionResponse.Response),
				})
			case p.Thought:
				if pendingText.Len() > 0 {
					out = append(out, canonicalPart{Kind: Text, Text: pendingText.String()})
					pendingText.Reset()
				}
				out = append(out, canonicalPart{
					Kind:             Thinking,
					Text:             p.Text,
					ThoughtSignature: p.ThoughtSignature,
				})
			default: // Text part
				if p.Text == "" && p.ThoughtSignature != "" {
					// Trailing empty text signature attaches to the preceding merged text
					if pendingText.Len() > 0 {
						out = append(out, canonicalPart{
							Kind:             Text,
							Text:             pendingText.String(),
							ThoughtSignature: p.ThoughtSignature,
						})
						pendingText.Reset()
					} else if len(out) > 0 && out[len(out)-1].Kind == Text {
						out[len(out)-1].ThoughtSignature = p.ThoughtSignature
					}
				} else if p.Text != "" {
					pendingText.WriteString(p.Text)
					if p.ThoughtSignature != "" {
						out = append(out, canonicalPart{
							Kind:             Text,
							Text:             pendingText.String(),
							ThoughtSignature: p.ThoughtSignature,
						})
						pendingText.Reset()
					}
				}
			}
		}
		if pendingText.Len() > 0 {
			out = append(out, canonicalPart{Kind: Text, Text: pendingText.String()})
		}
	} else {
		// Claude model rules:
		// Detached signature attaches to the next semantic part (first text chunk),
		// and subsequent text chunks are merged without replicating the signature.
		var pendingSig string
		var pendingText strings.Builder
		textHasSig := false

		flushText := func() {
			if pendingText.Len() > 0 {
				sig := ""
				if textHasSig {
					sig = pendingSig
					pendingSig = ""
					textHasSig = false
				}
				out = append(out, canonicalPart{
					Kind:             Text,
					Text:             pendingText.String(),
					ThoughtSignature: sig,
				})
				pendingText.Reset()
			}
		}

		for _, p := range raw {
			switch {
			case p.Thought:
				flushText()
				if p.Text != "" {
					out = append(out, canonicalPart{
						Kind: Thinking,
						Text: p.Text,
					})
				}
				if p.ThoughtSignature != "" {
					pendingSig = p.ThoughtSignature
				}
			case p.FunctionCall != nil:
				flushText()
				sig := p.ThoughtSignature
				if sig == "" && pendingSig != "" {
					sig = pendingSig
					pendingSig = ""
				}
				out = append(out, canonicalPart{
					Kind:             ToolCall,
					CallID:           p.FunctionCall.ID,
					Name:             p.FunctionCall.Name,
					Args:             string(p.FunctionCall.Args),
					ThoughtSignature: sig,
				})
			case p.FunctionResponse != nil:
				flushText()
				out = append(out, canonicalPart{
					Kind:   ToolResult,
					CallID: p.FunctionResponse.ID,
					Name:   p.FunctionResponse.Name,
					Args:   string(p.FunctionResponse.Response),
				})
			default: // Text part
				if p.Text != "" {
					if pendingSig != "" && !textHasSig {
						textHasSig = true
					}
					pendingText.WriteString(p.Text)
				}
			}
		}
		flushText()
	}

	return out
}
