import AVKit
import SwiftUI

// The clip's playback surface on iOS: SwiftUI's own VideoPlayer, unchanged.
// Its transport is the UIKit one, not the AppKit bar the macOS twin has to
// avoid, so there is nothing to hand-roll here and the scrubber stays.

struct ClipSurface: View {

    let player: AVPlayer?

    var body: some View {
        VideoPlayer(player: player)
    }
}
