import AVKit
import AppKit
import SwiftUI

struct ClipSurface: NSViewRepresentable {

    let player: AVPlayer?

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        // AVPlayerView's inline volume popover is unsatisfiable in AppKit
        // (NSGlassView pins it to 180pt, its slider needs 128) and warns on
        // every open; there is no per-control switch, only the whole bar.
        view.controlsStyle = .minimal
        view.player = player
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        if view.player !== player { view.player = player }
    }

}
