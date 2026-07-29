import Foundation

// One structured event of a chat session's internal life -- the unit the
// app's debug view graphs and the transcript log prints. Emitted by
// ChatSession through its trace sink. `text` carries the payload VERBATIM
// (the rendered delta laid into the KV, the raw decoded stream, a tool's
// exact result), so a saved transcript reconstructs the session without
// screenshots.
public struct TraceEvent: Sendable, Identifiable {
    public enum Kind: String, Sendable {
        case user
        case render      // template render (text = the exact delta laid)
        case prefill     // tokens extended into the KV
        case vision      // tower encode of the turn's image tiles
        case decode      // a decode stream ended (summary leads with WHY)
        case toolCall = "tool"
        case toolResult = "result"
        case inject      // </think> injected (caps / Quick Answer)
        case rewind      // KV rolled back to the turn mark
        case reset       // conversation state cleared
        case answer      // the committed answer
        case diag        // tool-endpoint diagnostics (never in the KV)
    }

    public let id = UUID()
    public let kind: Kind
    public let t0: Date
    public let t1: Date
    public let ctx: Int          // KV tokens after the event; -1 = n/a
    public let tokens: Int       // tokens this event moved; 0 = n/a
    public let summary: String   // one line for a list row
    public let text: String      // payload verbatim; "" = none
    // Tiny JPEG of a vision turn's image (<=128px, a few KB), so the debug
    // view shows WHAT was attached, not just its tile count.
    public let image: Data?

    public init(kind: Kind, t0: Date, t1: Date, ctx: Int, tokens: Int,
                summary: String, text: String, image: Data? = nil) {
        self.kind = kind
        self.t0 = t0
        self.t1 = t1
        self.ctx = ctx
        self.tokens = tokens
        self.summary = summary
        self.text = text
        self.image = image
    }
}
