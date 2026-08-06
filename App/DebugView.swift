import ImageIO
import LLM
import SwiftUI

// Session drill-in (the Option-hold ladybug), routed in-view like Settings:
// the context/tool time graph over the structured trace on top, the event
// transcript below. Clicking near a point on the graph scrolls its event
// into view; clicking a row expands its payload. The footer shows where the
// same trace is mirrored on disk (transcript.log.txt).

// Timestamps are elapsed since the app started, not time of day: a session
// trace is read as "how long in" and "how far apart", and a wall clock makes
// the reader do that subtraction. Hours appear only once there are any.

private func elapsed(_ t: Date, since zero: Date) -> String {
    let ms = Int((max(t.timeIntervalSince(zero), 0) * 1000).rounded())
    let out: String
    if ms >= 3_600_000 {
        out = String(format: "%02d:%02d:%02d.%03d", ms / 3_600_000,
                     ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
    } else {
        out = String(format: "%02d:%02d.%03d",
                     ms / 60_000, ms / 1000 % 60, ms % 1000)
    }
    return out
}

struct DebugView: View {

    let model: ChatModel
    let onClose: () -> Void
    @State private var selected: UUID?

    // Zero for every timestamp here: the app's launch, unless the trace
    // predates it -- a REOPENED conversation carries the events of the run
    // that recorded it, and against this launch every one of them would
    // clamp to 00:00.000. Then its own first event is the zero.
    private var zero: Date {
        min(Instrument.launched, model.traceEvents.first?.t1
            ?? Instrument.launched)
    }

    // Dismissal comes from the SAME place Settings gets it: the navigation
    // stack's own confirmation slot on iOS, a header of ours on macOS. Two
    // hand-rolled headers rendered two different Dones -- Settings' system
    // one and a bare tinted button here -- for one word doing one job.
    var body: some View {
        Group {
            if isOS {
                NavigationStack {
                    content.navigationTitle("Session Debug")
                        .toolbar { DoneToolbar(action: onClose) }
                }
            } else {
                VStack(spacing: 0) {
                    header
                    content
                }
            }
        }
    }

    private var content: some View {
        VStack(spacing: 0) {
            if isOS { eventCount.padding(.horizontal, 16) }
            TraceGraph(events: model.traceEvents, zero: zero,
                       selected: $selected)
                .frame(height: 190)
                .padding(.horizontal, 16)
                .padding(.bottom, 8)
            Divider()
            transcript
            Divider()
            footer
        }
    }

    // The navigation title carries the name on iOS, so the count travels on
    // its own rather than being dropped with the header that held it.
    private var eventCount: some View {
        Text("\(model.traceEvents.count) events")
            .appFont(.caption)
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var header: some View {
        HStack {
            Label("Session Debug", systemImage: "ladybug")
                .appFont(.headline)
            eventCount.fixedSize()
            Spacer()
            DoneButton(action: onClose)
            EscapeToClose(action: onClose)
        }
        .padding(12)
    }

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                // A log is scanned, not read: the gap between rows sits well
                // under the height of a row's own line, so a screenful is
                // events rather than air.
                LazyVStack(alignment: .leading, spacing: 2) {
                    ForEach(model.traceEvents) { e in
                        TraceRow(e: e, zero: zero, selected: e.id == selected)
                            .id(e.id)
                            .onTapGesture {
                                selected = selected == e.id ? nil : e.id
                            }
                    }
                }
                .padding(12)
            }
            .onChange(of: selected) { _, sel in
                if let sel {
                    withAnimation { proxy.scrollTo(sel, anchor: .center) }
                }
            }
        }
    }

    // Both logs by NAME and size, each a button that puts that file's contents
    // on the clipboard. The full path was accurate and useless: on a phone it
    // is a sandbox UUID nobody can act on, it needed head-truncating to fit,
    // and it named only ONE of the two files -- diag.log had no route out of
    // the app at all.
    private var footer: some View {
        HStack(spacing: 8) {
            logButton("Transcript", model.tracePath)
            Text("\u{00B7}").foregroundStyle(.tertiary)
            logButton("Diagnostics", model.diagPath)
            Spacer()
        }
        .appFont(.caption2)
        .padding(8)
    }

    // The checkmark is the same acknowledgement the transcript's code blocks
    // give: a clipboard write is invisible, and a button that says nothing is
    // indistinguishable from a dead one.
    @State private var copied: String?

    private func logButton(_ label: String, _ path: String) -> some View {
        Button { copyLog(path) } label: {
            HStack(spacing: 4) {
                Image(systemName: copied == path
                      ? "checkmark" : "doc.on.doc")
                Text("\(label): \(DebugView.folder(path)) "
                     + "(\(DebugView.size(path)))")
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.secondary)
        .help("Copy the \(label.lowercased()) log to the clipboard")
    }

    private func copyLog(_ path: String) {
        let text = (try? String(contentsOfFile: path, encoding: .utf8)) ?? ""
        setClipboard(text)
        copied = path
        Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(1200))
            if copied == path { copied = nil }
        }
    }

    // The FOLDER, not the file: every run writes a `current.txt`, so the name
    // that tells the two apart is the directory holding it.
    private static func folder(_ path: String) -> String {
        URL(fileURLWithPath: path).deletingLastPathComponent()
            .lastPathComponent
    }

    private static func size(_ path: String) -> String {
        let attrs = try? FileManager.default.attributesOfItem(atPath: path)
        let bytes = (attrs?[.size] as? Int64) ?? 0
        return ByteCountFormatter.string(fromByteCount: bytes,
                                         countStyle: .file)
    }
}

// Context size over time as a step line (a rewind reads as a visible drop),
// a pp/tg rate overlay (green, its own scale), and a marker lane on top:
// user turns as vertical rules, tool calls / injections / rewinds as
// colored dots. A tap selects the nearest event in time so any region of
// the graph drills into the transcript; hovering shows the event's stats
// in a bubble (tool wall time rides the result event's duration).

private struct TraceGraph: View {
    let events: [TraceEvent]
    let zero: Date
    @Binding var selected: UUID?
    @State private var hovered: TraceEvent?
    @State private var hoverAt: CGPoint = .zero
    // A Canvas draws Text values, not views, so its labels cannot take the
    // .appFont modifier the rest of the app uses and read the scale directly.
    @Environment(\.appTextScale) private var textScale

    private static let pad: CGFloat = 8

    var body: some View {
        GeometryReader { geo in
            Canvas { ctx, size in
                draw(&ctx, size)
            }
            .background(Color.gray.opacity(0.06),
                        in: RoundedRectangle(cornerRadius: 8))
            .onTapGesture(coordinateSpace: .local) { p in
                selected = nearest(p.x, geo.size)?.id
            }
            .onContinuousHover(coordinateSpace: .local) { phase in
                switch phase {
                case .active(let p):
                    hovered = nearest(p.x, geo.size)
                    hoverAt = p
                case .ended:
                    hovered = nil
                }
            }
            .overlay(alignment: .topLeading) {
                if let hovered {
                    HoverBubble(e: hovered, zero: zero)
                        .offset(
                            x: min(max(hoverAt.x + 12, 0),
                                   max(geo.size.width - 240, 0)),
                            y: min(max(hoverAt.y - 56, 0),
                                   max(geo.size.height - 64, 0)))
                        .allowsHitTesting(false)
                }
            }
        }
    }

    // The axis starts at the first event's MARK, not at its start. Everything
    // is plotted at t1, so anchoring to t0 spends the opening event's whole
    // duration on blank canvas -- and the opening event is the precooked
    // system prefill, ~15 s of it, which left a session's real activity
    // squeezed against the right edge.
    private var span: (start: Date, seconds: Double) {
        let start = events.first?.t1 ?? Date()
        let end = events.last?.t1 ?? start
        return (start, max(end.timeIntervalSince(start), 1))
    }

    private var maxCtx: Int {
        max(events.map { e in e.ctx }.max() ?? 1, 1)
    }

    private func x(_ t: Date, _ w: CGFloat) -> CGFloat {
        let s = span
        let f = t.timeIntervalSince(s.start) / s.seconds
        return TraceGraph.pad + CGFloat(f) * (w - 2 * TraceGraph.pad)
    }

    private func y(_ ctx: Int, _ h: CGFloat) -> CGFloat {
        let top = 24.0                    // marker lane above the line
        let f = CGFloat(ctx) / CGFloat(maxCtx)
        return h - TraceGraph.pad - f * (h - top - 2 * TraceGraph.pad)
    }

    private static func markerColor(_ kind: TraceEvent.Kind) -> Color? {
        switch kind {
        case .toolCall: return .orange
        case .toolResult: return .teal
        case .inject: return .yellow
        case .rewind, .reset: return .red
        default: return nil
        }
    }

    // Tokens per second of a prefill / decode event; 0 = not a rate event.
    private static func rate(_ e: TraceEvent) -> Double {
        let dur = e.t1.timeIntervalSince(e.t0)
        let rated = e.kind == .prefill || e.kind == .decode
        return rated && e.tokens > 0 && dur > 0
            ? Double(e.tokens) / dur : 0
    }

    private var maxRate: Double {
        max(events.map { e in TraceGraph.rate(e) }.max() ?? 1, 1)
    }

    private func draw(_ ctx: inout GraphicsContext, _ size: CGSize) {
        var line = Path()
        var lastY: CGFloat? = nil
        let rateTop = maxRate
        for e in events {
            let ex = x(e.t1, size.width)
            if e.ctx >= 0 {
                let ey = y(e.ctx, size.height)
                if let ly = lastY {
                    line.addLine(to: CGPoint(x: ex, y: ly))
                    line.addLine(to: CGPoint(x: ex, y: ey))
                } else {
                    line.move(to: CGPoint(x: ex, y: ey))
                }
                lastY = ey
            }
            if e.kind == .user {
                var rule = Path()
                rule.move(to: CGPoint(x: ex, y: 20))
                rule.addLine(to: CGPoint(x: ex, y: size.height - 4))
                ctx.stroke(rule, with: .color(.accentColor.opacity(0.5)),
                           style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
            }
            // pp/tg overlay on its own scale: prefill hollow, decode solid.
            let r = TraceGraph.rate(e)
            if r > 0 {
                let ry = y(Int(r / rateTop * Double(maxCtx)), size.height)
                let box = CGRect(x: ex - 2.5, y: ry - 2.5,
                                 width: 5, height: 5)
                if e.kind == .decode {
                    ctx.fill(Path(ellipseIn: box), with: .color(.green))
                } else {
                    ctx.stroke(Path(ellipseIn: box),
                               with: .color(.green), lineWidth: 1)
                }
            }
            if let color = TraceGraph.markerColor(e.kind) {
                let radius: CGFloat = e.id == selected ? 5 : 3.5
                let dot = Path(ellipseIn: CGRect(
                    x: ex - radius, y: 10 - radius,
                    width: 2 * radius, height: 2 * radius))
                ctx.fill(dot, with: .color(color))
            }
        }
        ctx.stroke(line, with: .color(.blue), lineWidth: 1.5)
        let label = appTextFont(.caption2, textScale)
        ctx.draw(Text("ctx \(maxCtx)")
                     .font(label).foregroundStyle(.secondary),
                 at: CGPoint(x: size.width - 34, y: 8))
        ctx.draw(Text(String(format: "%.0f t/s", maxRate))
                     .font(label).foregroundStyle(.green.opacity(0.7)),
                 at: CGPoint(x: size.width - 34, y: 22))
    }

    // The event nearest in time to the tapped x, markers preferred: within
    // the tap slop a marker (tool / rewind / inject) beats the prefill or
    // decode that shares its instant, since markers are what the lane shows.
    private func nearest(_ tapX: CGFloat, _ size: CGSize) -> TraceEvent? {
        var best: (e: TraceEvent, d: CGFloat)? = nil
        for e in events {
            var d = abs(x(e.t1, size.width) - tapX)
            if TraceGraph.markerColor(e.kind) != nil || e.kind == .user {
                d -= 6
            }
            if best == nil || d < best!.d { best = (e, d) }
        }
        return best?.e
    }
}

// The graph's hover bubble: the event's time, kind, summary, and stats
// (tokens, ctx, wall time, derived t/s) at a glance without leaving the
// graph.

private struct HoverBubble: View {
    let e: TraceEvent
    let zero: Date

    private var stats: String {
        var parts: [String] = []
        if e.tokens > 0 { parts.append("\(e.tokens) tok") }
        if e.ctx >= 0 { parts.append("ctx \(e.ctx)") }
        let dur = e.t1.timeIntervalSince(e.t0)
        if dur >= 0.001 { parts.append(String(format: "%.2fs", dur)) }
        if e.tokens > 0 && dur > 0 {
            parts.append(String(format: "%.0f t/s",
                                Double(e.tokens) / dur))
        }
        return parts.joined(separator: "  ")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("\(elapsed(e.t1, since: zero))  \(e.kind.rawValue)")
                .appFont(.caption2).monospaced()
                .foregroundStyle(.secondary)
            Text(e.summary)
                .appFont(.caption)
                .lineLimit(2)
            if !stats.isEmpty {
                Text(stats)
                    .appFont(.caption2).monospaced()
                    .foregroundStyle(.secondary)
            }
        }
        .padding(8)
        .frame(maxWidth: 240, alignment: .leading)
        .background(.thinMaterial,
                    in: RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8)
            .stroke(.separator, lineWidth: 0.5))
    }
}

// One transcript row: time, a kind chip, the summary; the payload (rendered
// delta, raw decode, tool result) collapsed to two lines until selected.

private struct TraceRow: View {
    let e: TraceEvent
    let zero: Date
    let selected: Bool
    @State private var payloadHeight: CGFloat = 20

    private var color: Color {
        switch e.kind {
        case .user: return .accentColor
        case .render: return .secondary
        case .prefill: return .blue
        case .vision: return .purple
        case .decode: return .primary
        case .toolCall: return .orange
        case .toolResult: return .teal
        case .inject: return .yellow
        case .rewind, .reset: return .red
        case .answer: return .green
        case .diag: return .indigo
        }
    }

    private var meta: String {
        var parts: [String] = []
        if e.tokens > 0 { parts.append("\(e.tokens) tok") }
        if e.ctx >= 0 { parts.append("ctx \(e.ctx)") }
        let dur = e.t1.timeIntervalSince(e.t0)
        if dur >= 0.001 { parts.append(String(format: "%.2fs", dur)) }
        return parts.joined(separator: "  ")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(elapsed(e.t1, since: zero))
                    .appFont(.caption).monospaced()
                    .foregroundStyle(.secondary)
                Text(e.kind.rawValue)
                    .appFont(.caption).bold()
                    .padding(.horizontal, 5)
                    .padding(.vertical, 1)
                    .background(color.opacity(0.18), in: Capsule())
                    .foregroundStyle(color)
                if !meta.isEmpty {
                    Text(meta)
                        .appFont(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(e.summary)
                    .appFont(.caption)
                    .lineLimit(1)
                    .truncationMode(.tail)
            }
            if let cg = TraceRow.thumb(e.image) {
                Image(decorative: cg, scale: 1)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(maxWidth: 128, maxHeight: 128)
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                    .padding(.leading, 4)
            }
            if !e.text.isEmpty {
                payload
            }
        }
        // Tighter vertically than horizontally, but not to nothing: the
        // selection highlight needs some body around the text or it reads as
        // a stray tint rather than as a selected row.
        .padding(.horizontal, 6)
        .padding(.vertical, 3)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(selected ? Color.accentColor.opacity(0.12) : .clear,
                    in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
    }

    // An expanded payload scrolls inside a content-hugging capped frame (like
    // the tool popover): a selected render is many KB, and a row that
    // balloons to thousands of points collapses the transcript's scroll
    // geometry into a blank void when it shrinks back.
    @ViewBuilder private var payload: some View {
        let text = Text(e.text)
            .appFont(.caption2).monospaced()
            .foregroundStyle(.secondary)
            .textSelection(.enabled)
            .padding(.leading, 4)
        if selected {
            ScrollView {
                text
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onGeometryChange(for: CGFloat.self, of: { geo in
                        geo.size.height
                    }, action: { h in payloadHeight = h })
            }
            .frame(height: min(max(payloadHeight, 20), 300))
        } else {
            text.lineLimit(2)
        }
    }

    private static func thumb(_ data: Data?) -> CGImage? {
        data.flatMap { d in
            CGImageSourceCreateWithData(d as CFData, nil).flatMap { src in
                CGImageSourceCreateImageAtIndex(src, 0, nil)
            }
        }
    }
}
