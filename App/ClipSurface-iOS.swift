import AVKit
import SwiftUI

struct ClipSurface: View {

    let player: AVPlayer?

    var body: some View {
        VideoPlayer(player: player)
    }
}
