import Foundation

// Find / Find Next across the transcript's PER-MESSAGE single-surface text
// views. The App registers each answer bubble's text view (keyed by message
// id) and the transcript order; the controller highlights every match in every
// registered view, tracks one global cursor across them, and -- through the
// App's scroll closure -- scrolls the transcript to the active match's message
// before selecting it in that bubble. One registered view (the whole-document
// reading surface) degenerates to plain single-view find.

@MainActor
public final class MarkdownFindController {

    public struct Options: Sendable {
        public var caseSensitive: Bool
        public init(caseSensitive: Bool = false) {
            self.caseSensitive = caseSensitive
        }
    }

    // Registered views by message id; the transcript order and the App's
    // scroll-to-message closure are set by the content view.
    private var targets: [UUID: any FindableTextView] = [:]
    public var order: [UUID] = []
    // Scroll the transcript to a match: the message id plus the match's vertical
    // position as a fraction (0...1) of that message, so the exact line lands in
    // view rather than merely centering a long bubble. nil fraction => center.
    public var scrollTo: ((UUID, CGFloat?) -> Void)?

    public private(set) var matchCount = 0
    public private(set) var currentMatch = 0   // 1-based; 0 when no match

    // Per-view match counts in transcript order, and the 0-based global cursor.
    private var slots: [(id: UUID, view: any FindableTextView, count: Int)] = []
    private var cursor = -1
    private var query = ""
    private var caseSensitive = false
    // The view currently holding the active-match selection, so stepping clears
    // ONLY its selection -- a manual selection the user made in another bubble
    // survives navigation.
    private weak var activeView: (any FindableTextView)?

    public init() {}

    // A bubble's text view registers on creation, unregisters on teardown. A
    // registration during an active find re-highlights that view so a bubble
    // scrolled into existence mid-search still shows its matches.
    func register(_ id: UUID, _ view: any FindableTextView) {
        targets[id] = view
        if !query.isEmpty {
            _ = view.findAll(query, caseSensitive: caseSensitive)
        }
    }

    func unregister(_ id: UUID, _ view: any FindableTextView) {
        if targets[id] === view { targets[id] = nil }
    }

    // Search from the top: highlight every match in every view, then activate
    // the first. Returns the total match count.
    @discardableResult
    public func find(_ q: String,
                     options: Options = Options()) -> Int {
        query = q
        caseSensitive = options.caseSensitive
        rebuildSlots()
        cursor = -1
        if matchCount > 0 { _ = step(forward: true) }
        return matchCount
    }

    @discardableResult
    public func findNext() -> Int { step(forward: true) }

    @discardableResult
    public func findPrevious() -> Int { step(forward: false) }

    public func clear() {
        for (_, v) in targets { v.clearFind() }
        query = ""
        slots = []
        cursor = -1
        matchCount = 0
        currentMatch = 0
        activeView = nil
    }

    // Re-highlight every view from scratch and total the counts. Only find()
    // uses this (it re-tints); navigation uses recountSlots (no re-tint).
    private func rebuildSlots() {
        slots = order.compactMap { id in
            targets[id].map { v in
                (id, v, v.findAll(query, caseSensitive: caseSensitive))
            }
        }
        matchCount = slots.reduce(0) { sum, s in sum + s.count }
    }

    // Refresh per-view counts from each view's CURRENT matches WITHOUT
    // re-highlighting, so navigation picks up views that changed while find was
    // open -- a bubble that streamed new tokens (reapplyFind grew its matches)
    // or a bubble registered after find() ran. The cursor is clamped to the
    // new total.
    private func recountSlots() {
        slots = order.compactMap { id in
            targets[id].map { v in (id, v, v.liveFindCount) }
        }
        matchCount = slots.reduce(0) { sum, s in sum + s.count }
        if matchCount == 0 {
            cursor = -1
        } else if cursor >= matchCount {
            cursor = matchCount - 1
        }
    }

    private func step(forward: Bool) -> Int {
        recountSlots()
        var result = 0
        activeView?.setActive(nil)
        activeView = nil
        if matchCount > 0 {
            cursor = ((cursor + (forward ? 1 : -1)) % matchCount
                      + matchCount) % matchCount
            if let hit = locate(cursor) {
                hit.view.setActive(hit.local)
                activeView = hit.view
                scrollTo?(hit.id, hit.view.activeMatchFraction())
                currentMatch = cursor + 1
                result = currentMatch
            }
        } else {
            currentMatch = 0
        }
        return result
    }

    // Map a global match index to its view and the view-local match index.
    private func locate(_ global: Int)
        -> (id: UUID, view: any FindableTextView, local: Int)? {
        var g = global
        var result: (id: UUID, view: any FindableTextView, local: Int)? = nil
        var i = 0
        while result == nil && i < slots.count {
            if g < slots[i].count {
                result = (slots[i].id, slots[i].view, g)
            } else {
                g -= slots[i].count
            }
            i += 1
        }
        return result
    }
}

// Implemented by the platform text views (Bridges-iOS / Bridges-macOS). A view
// highlights ALL its matches on findAll (no selection); setActive selects and
// scrolls one of them into view (or clears the active selection with nil).
@MainActor
protocol FindableTextView: AnyObject {
    func findAll(_ query: String, caseSensitive: Bool) -> Int
    func setActive(_ localIndex: Int?)
    func clearFind()
    // Current match count without recomputing, so the controller can re-total
    // after a view's matches change under streaming (reapplyFind).
    var liveFindCount: Int { get }
    // Vertical center of the active match as a fraction (0...1) of the view's
    // height, so the transcript scrolls the exact line into view -- not just
    // the middle of a long message. nil when there is no active match.
    func activeMatchFraction() -> CGFloat?
}

// All (overlapping-free) ranges of `query` in `text`. Non-advancing matches are
// guarded so an empty or degenerate query terminates.
func markdownFindRanges(in text: String, query: String,
                        caseSensitive: Bool) -> [NSRange] {
    var result: [NSRange] = []
    if !query.isEmpty {
        let ns = text as NSString
        let opts: NSString.CompareOptions =
            caseSensitive ? [] : .caseInsensitive
        var start = 0
        var searching = true
        while searching {
            let scope = NSRange(location: start, length: ns.length - start)
            let r = ns.range(of: query, options: opts, range: scope)
            if r.location == NSNotFound {
                searching = false
            } else {
                result.append(r)
                start = r.location + max(r.length, 1)
                if start >= ns.length { searching = false }
            }
        }
    }
    return result
}
