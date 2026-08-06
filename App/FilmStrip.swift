import CoreGraphics
import SwiftUI

// The frames of an attached video, shown behind the composer while the model
// encodes them -- so a forty second wait says "it is looking at YOUR video"
// rather than "something is happening".
//
// It sits on the composer's bottom edge at a FIXED height, so the card growing
// as someone types never resizes it. The frame is FIT to that height and is
// therefore narrower than the card; FilmFrame owns that look.
//
// WHY THIS SHAPE AND NOT AN EFFECT. A new frame flips one `@State` here and
// the cross-fade is an OPACITY, which the render server interpolates from a
// single commit. That is ~32 state changes for the whole show. A
// `TimelineView` or a `.symbolEffect` would instead redraw every display
// refresh for as long as it ran, which is the shape that froze this app --
// see the STANDING RULE in [[voice-turn-and-render-stalls]].

struct FilmStrip: View {

    var model: ChatModel
    // Typing wins. The composer is the one surface that has to stay calm
    // while it is being used, so composing retires the picture rather than
    // dimming it further.
    let quiet: Bool

    // `shown` is opaque and `arriving` fades in over it. Two layers, one
    // animated property; the handover is in `accept`.
    @State private var shown: CGImage?
    @State private var arriving: CGImage?
    @State private var fade: Double = 0
    @State private var previous: Date?
    // The whole strip's presence. Frames stop when the encode ends and the
    // prefill's half-minute begins, and the last one fades out from there
    // rather than sitting on screen for the rest of it.
    @State private var alive: Double = 1

    // The pair's alphas SUM TO ONE at every instant: the frame underneath
    // stays fully opaque and the new one fades in over it, so the composite
    // is `f * new + (1 - f) * old` and total brightness never moves.
    //
    // The dim is applied ONCE, to the pair, and that is the whole reason this
    // stopped flickering. Dimming each layer instead let 30% of black through
    // the one underneath, so brightness climbed through every dissolve and
    // then dropped at the handover -- a visible pulse per frame, 32 times.
    //
    // `compositingGroup` IS LOAD-BEARING, not tidiness: without it SwiftUI
    // pushes the opacity down onto each child and the result is byte for byte
    // the broken version. MEASURED, mean luminance across a dissolve --
    // per-layer 25.9 -> 34.6 (a 34% climb), grouped 25.9 -> 27.5, where the
    // remainder is the two frames genuinely differing in brightness.
    var body: some View {
        ZStack {
            if let shown { FilmFrame(image: shown, fade: 1) }
            if let arriving { FilmFrame(image: arriving, fade: fade) }
        }
        .compositingGroup()
        // Both multiply the ALREADY COMPOSITED pair, so neither can
        // reintroduce the per-layer blending the group exists to stop.
        .opacity(FilmFrame.dim * alive)
        // FILLS the card rather than sitting on its bottom edge at a fixed
        // height, which left dead space above the picture. The frame is still
        // fit, so the card being far wider than it is tall makes the HEIGHT
        // the limiting side -- it always reaches top and bottom, and the
        // vertical feather is what keeps that from reading as a cut.
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(false)
        .opacity(quiet ? 1 : 0)
        .animation(.easeInOut(duration: 0.2), value: quiet)
        .onChange(of: model.lookingAt) { _, next in
            accept(next?.image)
        }
        // Linear, not eased: over five seconds an ease-out sits near full
        // brightness for the first second and then drops, which reads as the
        // picture hanging and then being cut rather than as it receding.
        .onChange(of: model.watching) { _, live in
            if live {
                alive = 1
            } else {
                withAnimation(.linear(duration: 5)) { alive = 0 }
            }
        }
    }

    // NO completion handler, deliberately. Promoting the finished frame from
    // one raced the next arrival: a frame landing mid-fade reset `fade` while
    // the previous animation's completion was still pending, and that stale
    // callback then clobbered the fade in flight. The handover happens HERE
    // instead -- whatever was fading in becomes the base -- so the only thing
    // that ever schedules work is an arrival, and there is nothing to race.
    //
    // The dissolve takes a FRACTION of the gap, so every frame then rests
    // alone for the remainder of it. One spanning the whole interval never
    // lets it: the pair is superimposed at every instant, and two moments of
    // one clip half a second apart read as two copies of the video playing
    // over each other. Clamped both ways -- the first gap is unknown, and a
    // slow frame must not stretch its dissolve back into a double exposure.
    private func accept(_ next: CGImage?) {
        let now = Date()
        let gap = previous.map { at in now.timeIntervalSince(at) } ?? 0.35
        previous = now
        if let next {
            shown = arriving ?? shown
            arriving = next
            // Two writes to `fade` in one tick would coalesce and animate
            // from wherever it already was -- which is 1, so the new frame
            // would cut in with no dissolve at all. The reset lands this
            // tick and the animation starts on the next.
            fade = 0
            let seconds = min(max(gap * 0.3, 0.08), 0.25)
            Task { @MainActor in
                withAnimation(.easeInOut(duration: seconds)) { fade = 1 }
            }
        } else {
            shown = nil
            arriving = nil
            previous = nil
        }
    }
}
