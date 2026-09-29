package gateway

import (
	"encoding/json"
	"regexp"
	"strings"
)

// ---- Google Code Assist (provider side) --------------------------------------
//
// Gemini CLI and Antigravity sign in with Google and talk to Code Assist
// (cloudcode-pa.googleapis.com), which takes a Gemini request wrapped in an
// envelope — {model, project, request} — and streams replies wrapped the
// same way, {response}. The gateway builds {model, request}; the account
// adds the project and the ids its app sends when it signs the request.

// skipSignature stands in for a thought signature on a function call the
// model didn't make here: Google checks the ones Gemini 3 hands out, and
// this one tells it not to.
const skipSignature = "skip_thought_signature_validator"

var unsafeToolID = regexp.MustCompile(`[^a-zA-Z0-9_-]`)

// buildCodeAssist builds the Code Assist envelope for a request, for the
// app the account belongs to: "gemini" or "antigravity".
func buildCodeAssist(r *Request, model, agent string) []byte {
	ag := agent == "antigravity"
	claude := strings.Contains(strings.ToLower(model), "claude")
	toolID := func(id string) string {
		if !ag || id == "" {
			return id
		}
		return unsafeToolID.ReplaceAllString(id, "_")
	}
	names := map[string]string{}
	var contents []map[string]any
	for _, m := range r.Messages {
		role := "user"
		if m.Role == "assistant" {
			role = "model"
		}
		var parts []map[string]any
		var pendingSignature string
		for _, p := range m.Parts {
			switch p.Kind {
			case Thinking:
				if p.Signature != "" && ag {
					pendingSignature = p.Signature
				}
			case Text:
				sig := p.Signature
				if sig == "" && pendingSignature != "" && ag {
					sig = pendingSignature
					pendingSignature = ""
				}
				textPart := map[string]any{"text": p.Text}
				if sig != "" && ag {
					textPart["thoughtSignature"] = sig
				}
				if p.Text != "" {
					parts = append(parts, textPart)
				}
			case Image, File:
				if p.Data != "" {
					parts = append(parts, map[string]any{"inlineData": map[string]any{"mimeType": p.MediaType, "data": p.Data}})
				} else if p.URL != "" {
					parts = append(parts, map[string]any{"fileData": map[string]any{"mimeType": p.MediaType, "fileUri": p.URL}})
				}
			case ToolCall:
				name := p.Name
				args := json.RawMessage(argsOf(p))
				id := p.ID
				sig := p.Signature
				if sig == "" && pendingSignature != "" && ag {
					sig = pendingSignature
					pendingSignature = ""
				}
				if ag {
					if binding, ok := defaultToolBindingStore.LookupByClientID(p.ID); ok {
						name = binding.NativeName
						args = json.RawMessage(binding.NativeArgs)
						id = binding.NativeID
						if binding.NativeSignature != "" {
							sig = binding.NativeSignature
						}
					}
				}
				names[p.ID] = name
				call := map[string]any{"name": name, "args": args}
				if tid := toolID(id); tid != "" {
					call["id"] = tid
				}
				callPart := map[string]any{"functionCall": call}
				if ag {
					// Reason: B11 requires real signatures only on Antigravity; eliminate skipSignature pseudo-signature
					if sig != "" {
						callPart["thoughtSignature"] = sig
					}
				} else {
					callPart["thoughtSignature"] = skipSignature
				}
				parts = append(parts, callPart)
			case ToolResult:
				name := names[p.CallID]
				resID := p.CallID
				if ag {
					if binding, ok := defaultToolBindingStore.LookupByClientID(p.CallID); ok {
						name = binding.NativeName
						resID = binding.NativeID
					}
				}
				if name == "" {
					name = "tool"
				}
				// Reason: Antigravity's official client puts success and error text alike in
				// response.output; other Code Assist callers keep their error key
				key := "output"
				if p.IsError && !ag {
					key = "error"
				}
				res := map[string]any{"name": name, "response": map[string]any{key: p.Text}}
				if tid := toolID(resID); tid != "" {
					res["id"] = tid
				}
				if len(p.Images) > 0 {
					var imgParts []map[string]any
					for _, img := range p.Images {
						if img.Data != "" {
							imgParts = append(imgParts, map[string]any{"inlineData": map[string]any{"mimeType": img.MediaType, "data": img.Data}})
						} else if img.URL != "" {
							imgParts = append(imgParts, map[string]any{"fileData": map[string]any{"mimeType": img.MediaType, "fileUri": img.URL}})
						}
					}
					if len(imgParts) > 0 {
						res["parts"] = imgParts
					}
				}
				parts = append(parts, map[string]any{"functionResponse": res})
			}
			// thinking isn't sent back: its signatures belong to whoever
			// made them, and Google turns away ones it didn't
		}
		if len(parts) == 0 {
			continue
		}
		isResultTurn := false
		for _, p := range parts {
			if _, ok := p["functionResponse"]; ok {
				isResultTurn = true
				break
			}
		}
		// Reason: On Antigravity Gemini, tool results form an independent role=model content,
		// whereas Claude uses an independent role=user content.
		if isResultTurn && ag && !claude {
			role = "model"
		}
		prevHasCall := false
		if n := len(contents); n > 0 {
			if prevParts, ok := contents[n-1]["parts"].([]map[string]any); ok {
				for _, pp := range prevParts {
					if _, ok := pp["functionCall"]; ok {
						prevHasCall = true
						break
					}
				}
			}
		}
		// Never merge functionResponse into previous functionCall content
		if n := len(contents); n > 0 && contents[n-1]["role"] == role && (!isResultTurn || !prevHasCall) && (!ag || !isResultTurn || role != "model") {
			contents[n-1]["parts"] = append(contents[n-1]["parts"].([]map[string]any), parts...)
			continue
		}
		contents = append(contents, map[string]any{"role": role, "parts": parts})
	}
	req := map[string]any{"contents": contents}
	if r.System != "" {
		req["systemInstruction"] = map[string]any{"role": "user", "parts": []map[string]any{{"text": r.System}}}
	}

	if len(r.Tools) > 0 && r.ToolChoice != "none" {
		var decls []map[string]any
		for _, t := range r.Tools {
			d := map[string]any{"name": t.Name}
			if t.Description != "" {
				d["description"] = t.Description
			}
			schema := t.Schema
			if len(schema) == 0 {
				schema = json.RawMessage(`{"type":"object","properties":{}}`)
			}
			if ag {
				d["parameters"] = plainSchema(schema)
			} else {
				d["parametersJsonSchema"] = schema
			}
			decls = append(decls, d)
		}
		req["tools"] = []map[string]any{{"functionDeclarations": decls}}
		mode := "AUTO"
		fc := map[string]any{}
		switch {
		case r.ToolChoice == "required":
			mode = "ANY"
		case strings.HasPrefix(r.ToolChoice, "name:"):
			mode = "ANY"
			targetName := strings.TrimPrefix(r.ToolChoice, "name:")
			if ag {
				if nativeName, ok := defaultToolBindingStore.LookupNativeName(targetName); ok {
					targetName = nativeName
				} else if binding, ok := defaultToolBindingStore.LookupByClientID(targetName); ok {
					targetName = binding.NativeName
				}
			}
			fc["allowedFunctionNames"] = []string{targetName}
		}
		// Reason: Antigravity's official client sends automatic tool choice with
		// no toolConfig at all, for Gemini and Claude alike
		if mode != "AUTO" || !ag {
			fc["mode"] = mode
			req["toolConfig"] = map[string]any{"functionCallingConfig": fc}
		}
	}

	gen := map[string]any{}
	if r.MaxTokens > 0 {
		gen["maxOutputTokens"] = r.MaxTokens
	}
	if r.Temp != nil {
		gen["temperature"] = *r.Temp
	}
	if r.TopP != nil {
		gen["topP"] = *r.TopP
	}
	if len(r.Stop) > 0 {
		gen["stopSequences"] = r.Stop
	}
	if tc := thinkingConfig(r, model, claude, ag); tc != nil {
		gen["thinkingConfig"] = tc
		// Claude's answer has to have room past the thinking it was asked for;
		// a default budget leaves the output limit to the upstream, as it was
		asked := r.Thinking || r.Effort != ""
		if b, ok := tc["thinkingBudget"].(int); ok && claude && asked && gen["maxOutputTokens"] == nil {
			gen["maxOutputTokens"] = b + 32000
		}
	}
	if len(gen) > 0 {
		req["generationConfig"] = gen
	}
	b, _ := json.Marshal(map[string]any{"model": model, "request": req})
	return b
}

// antigravityThinkingBudgets are the thinking budgets Antigravity's official
// client sends each model when nobody says how hard to think.
var antigravityThinkingBudgets = map[string]int{
	"gemini-3.8-flash-high": -1, // dynamic: the model decides
	"gemini-3.8-flash-low":  1000,
	"claude-sonnet-4-6":     1024,
}

// thinkingConfig says how hard the model should think: a level for
// Gemini 3, a budget for the rest. A model that doesn't think gets none.
func thinkingConfig(r *Request, model string, claude, ag bool) map[string]any {
	m := strings.ToLower(model)
	if ag {
		if r.NoThinking {
			return nil
		}
		// Reason: the caller didn't say, so do as Antigravity's own client does
		// for the models it was measured with; others keep the rules below
		if budget, ok := antigravityThinkingBudgets[m]; ok && r.Effort == "" && !r.Thinking {
			if claude && r.MaxTokens > 0 && budget >= r.MaxTokens {
				// Claude's thinking has to fit below its output limit
				return nil
			}
			return map[string]any{"includeThoughts": true, "thinkingBudget": budget}
		}
	}
	// Antigravity's Claude thinks whatever its name, as its own client shows
	if strings.HasPrefix(m, "gpt-oss") || claude && !ag && !strings.Contains(m, "thinking") {
		return nil
	}
	if r.Effort == "" && !r.Thinking {
		// Claude asked to think still has to be told how much
		if !claude {
			return nil
		}
	}
	tc := map[string]any{"includeThoughts": true}
	if strings.HasPrefix(m, "gemini-3") || strings.HasPrefix(m, "gemini-pro-agent") {
		if r.Effort != "" {
			level := "high"
			if r.Effort == "low" {
				level = "low"
			}
			tc["thinkingLevel"] = level
		}
		return tc
	}
	budget := budgetOf(effortOf(r.Effort))
	if claude && r.MaxTokens > 0 && budget >= r.MaxTokens {
		budget = r.MaxTokens - 1
		if budget < 1024 {
			return nil
		}
	}
	tc["thinkingBudget"] = budget
	return tc
}

// plainSchema is a JSON schema cut down to what Antigravity's function
// declarations take: an OpenAPI-style subset with no references, unions
// or keywords outside it.
func plainSchema(raw json.RawMessage) json.RawMessage {
	var root map[string]any
	if json.Unmarshal(raw, &root) != nil {
		return raw
	}
	defs := map[string]any{}
	for _, k := range []string{"$defs", "definitions"} {
		if d, ok := root[k].(map[string]any); ok {
			for n, v := range d {
				defs[n] = v
			}
		}
	}
	var walk func(v any, depth int) any
	walk = func(v any, depth int) any {
		switch x := v.(type) {
		case []any:
			out := make([]any, len(x))
			for i, e := range x {
				out[i] = walk(e, depth)
			}
			return out
		case map[string]any:
			if depth > 32 {
				return map[string]any{"type": "object"}
			}
			if ref, ok := x["$ref"].(string); ok {
				name := ref[strings.LastIndex(ref, "/")+1:]
				if d, ok := defs[name]; ok {
					return walk(d, depth+1)
				}
				return map[string]any{"type": "object"}
			}
			// allOf: one schema with all the members' fields
			if all, ok := x["allOf"].([]any); ok {
				merged := map[string]any{}
				for k, v := range x {
					if k != "allOf" {
						merged[k] = v
					}
				}
				for _, e := range all {
					if m, ok := walk(e, depth+1).(map[string]any); ok {
						for k, v := range m {
							if k == "properties" {
								props, _ := merged["properties"].(map[string]any)
								if props == nil {
									props = map[string]any{}
								}
								for pk, pv := range v.(map[string]any) {
									props[pk] = pv
								}
								merged["properties"] = props
							} else if _, has := merged[k]; !has {
								merged[k] = v
							}
						}
					}
				}
				return walk(merged, depth+1)
			}
			// anyOf / oneOf: the first member that isn't null, nullable
			for _, k := range []string{"anyOf", "oneOf"} {
				if alts, ok := x[k].([]any); ok {
					var pick map[string]any
					nullable := false
					for _, a := range alts {
						m, _ := a.(map[string]any)
						if m["type"] == "null" {
							nullable = true
						} else if pick == nil && m != nil {
							pick = m
						}
					}
					out := map[string]any{}
					if pick != nil {
						if m, ok := walk(pick, depth+1).(map[string]any); ok {
							out = m
						}
					}
					if d, ok := x["description"].(string); ok && out["description"] == nil {
						out["description"] = d
					}
					if nullable {
						out["nullable"] = true
					}
					if out["type"] == nil {
						out["type"] = "string"
					}
					return out
				}
			}
			out := map[string]any{}
			for k, v := range x {
				switch k {
				case "$schema", "$defs", "definitions", "$id", "$comment", "additionalProperties", "format", "default",
					"examples", "example", "title", "patternProperties", "enumDescriptions", "prefill", "deprecated",
					"propertyNames", "unevaluatedProperties", "readOnly", "writeOnly", "const":
				case "type":
					if ts, ok := v.([]any); ok {
						for _, t := range ts {
							if t == "null" {
								out["nullable"] = true
							} else if out["type"] == nil {
								out["type"] = t
							}
						}
					} else {
						out["type"] = v
					}
				case "properties":
					props := map[string]any{}
					if m, ok := v.(map[string]any); ok {
						for pk, pv := range m {
							props[pk] = walk(pv, depth+1)
						}
					}
					out[k] = props
				case "items":
					out[k] = walk(v, depth+1)
				case "enum":
					// Google takes string enums only
					if es, ok := v.([]any); ok {
						strs := make([]any, 0, len(es))
						for _, e := range es {
							if s, ok := e.(string); ok {
								strs = append(strs, s)
							}
						}
						if len(strs) == len(es) {
							out[k] = strs
						}
					}
				default:
					out[k] = walk(v, depth+1)
				}
			}
			if c, ok := x["const"]; ok {
				if s, ok := c.(string); ok {
					out["enum"] = []any{s}
				}
			}
			if out["type"] == nil {
				switch {
				case out["properties"] != nil:
					out["type"] = "object"
				case out["items"] != nil:
					out["type"] = "array"
				}
			}
			if req, ok := out["required"].([]any); ok {
				// required names only the properties there are
				props, _ := out["properties"].(map[string]any)
				keep := []any{}
				for _, n := range req {
					if s, ok := n.(string); ok && props[s] != nil {
						keep = append(keep, s)
					}
				}
				if len(keep) == 0 {
					delete(out, "required")
				} else {
					out["required"] = keep
				}
			}
			return out
		}
		return v
	}
	b, err := json.Marshal(walk(root, 0))
	if err != nil {
		return raw
	}
	return b
}

// codeAssistDecoder reads Code Assist's stream: Gemini chunks, each wrapped
// in {response}.
type codeAssistDecoder struct {
	started bool
	tools   bool
	stopped bool
	usage   *Usage
}

func (d *codeAssistDecoder) decode(data string, emit func(Event)) error {
	var ch struct {
		Response *geminiChunk `json:"response"`
		Error    *struct {
			Message string `json:"message"`
		} `json:"error"`
		geminiChunk
	}
	if err := json.Unmarshal([]byte(data), &ch); err != nil {
		return nil
	}
	if ch.Error != nil {
		emit(Event{Kind: KError, Text: ch.Error.Message})
		return nil
	}
	c := ch.geminiChunk
	if ch.Response != nil {
		c = *ch.Response
	}
	if !d.started {
		d.started = true
		emit(Event{Kind: KStart, MsgID: c.ResponseID, Model: c.ModelVersion})
	}
	if c.UsageMetadata != nil {
		u := c.UsageMetadata
		d.usage = &Usage{Input: max(u.Prompt-u.Cached, 0), CacheRead: u.Cached,
			Output: u.Candidates + u.Thoughts, Reasoning: u.Thoughts}
	}
	for _, cand := range c.Candidates {
		for _, p := range cand.Content.Parts {
			switch {
			case p.FunctionCall != nil:
				id := p.FunctionCall.ID
				if id == "" {
					id = "call_" + newID()
				}
				args := p.FunctionCall.Args
				if len(args) == 0 || string(args) == "null" {
					args = json.RawMessage("{}")
				}
				d.tools = true
				emit(Event{Kind: KToolStart, ID: id, Name: p.FunctionCall.Name})
				emit(Event{Kind: KToolArgs, Text: string(args)})
			case p.Thought:
				if p.Text != "" {
					emit(Event{Kind: KThink, Text: p.Text})
				}
				if p.Signature != "" {
					emit(Event{Kind: KSig, Text: p.Signature})
				}
			case p.Text != "":
				emit(Event{Kind: KText, Text: p.Text})
			}
		}
		if cand.FinishReason != "" && !d.stopped {
			d.stopped = true
			stop := stopFromGemini(cand.FinishReason)
			if d.tools && stop == "stop" {
				stop = "tool"
			}
			emit(Event{Kind: KStop, Stop: stop})
		}
	}
	if d.stopped && d.usage != nil {
		emit(Event{Kind: KUsage, Usage: *d.usage})
		d.usage = nil
	}
	return nil
}

type geminiChunk struct {
	Candidates []struct {
		Content struct {
			Parts []gPart `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata *struct {
		Prompt     int `json:"promptTokenCount"`
		Candidates int `json:"candidatesTokenCount"`
		Thoughts   int `json:"thoughtsTokenCount"`
		Cached     int `json:"cachedContentTokenCount"`
	} `json:"usageMetadata"`
	ModelVersion string `json:"modelVersion"`
	ResponseID   string `json:"responseId"`
}

func stopFromGemini(s string) string {
	switch s {
	case "MAX_TOKENS":
		return "length"
	case "STOP", "FINISH_REASON_UNSPECIFIED", "OTHER":
		return "stop"
	case "MALFORMED_FUNCTION_CALL", "UNEXPECTED_TOOL_CALL":
		return "stop"
	}
	// SAFETY, RECITATION, BLOCKLIST, PROHIBITED_CONTENT, SPII, ...
	return "filter"
}
