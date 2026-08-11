import AVKit
import AppKit
import SwiftUI

struct ClipSurface: NSViewRepresentable {

    let player: AVPlayer?

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        // .minimal: the inline bar's volume popover is unsatisfiable by
        // construction in AppKit (NSGlassView pins it to 180pt, the slider
        // inside needs 128) and logs a framework warning every time it opens.
        view.controlsStyle = .minimal
        view.player = player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }
}
