import AVKit
import AppKit
import SwiftUI

// The clip's playback surface on macOS: AVPlayerView at MINIMAL controls, a
// play/pause overlay and nothing else.
//
// Not SwiftUI's VideoPlayer, which is fixed at the inline control bar -- and
// that bar's volume flyout is broken inside AppKit. NSGlassView pins the
// popover to 180pt; within it AVGlassVolumeSlider is left 107pt of room
// (180 - 12 - 9 outer, then - 12 - 9 - 22 - 9 inner) while simultaneously
// being required to be 128. Unsatisfiable by 21 points, in Apple's own view,
// at its own fixed size, and logged in full every time it opens.
//
// There is no per-control switch to reach for: AVPlayerView offers
// showsFrameSteppingButtons, showsSharingServiceButton and
// showsFullScreenToggleButton, and no volume equivalent. So losing the
// control means losing the whole bar, which costs the scrubber and the time
// readout. Taken deliberately for a ten-second clip in a chat bubble, where
// play/pause is enough and a recurring framework warning is not.

struct ClipSurface: NSViewRepresentable {

    let player: AVPlayer?

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .minimal
        view.player = player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }
}
