import Foundation
import LLM

// Speech probe: text in, WAV out, no model load. The acceptance test for the
// TTS engine is a BYTE COMPARISON against the reference binary it was ported
// from, so this writes the same 24 kHz mono PCM WAV that one does:
//
//   gadeon-cli --tts "Hello there." --tts-out ours.wav
//   tts-cli -p "Hello there." -s Kiki -o ref.wav
//   cmp ours.wav ref.wav
//
//   gadeon-cli --tts-voices              # list the voices
//   gadeon-cli --tts @speech.txt --tts-voice Luna --tts-speed 1.2

// A NaN sample fails both comparisons and reaches the conversion. The
// reference C's cast is undefined there and lowers to fcvtzs, which answers 0
// on arm64; Swift's Int16(Float) traps instead. Long utterances do produce
// NaN, so the reference binary's answer is the one to give.

func wavClamp(_ v: Float) -> Int16 {
    var s = v
    if s > 1.0 { s = 1.0 }
    if s < -1.0 { s = -1.0 }
    return s.isNaN ? 0 : Int16(s * 32767.0)
}

func wavPutU32(_ f: inout [UInt8], _ v: UInt32) {
    f.append(UInt8(truncatingIfNeeded: v))
    f.append(UInt8(truncatingIfNeeded: v >> 8))
    f.append(UInt8(truncatingIfNeeded: v >> 16))
    f.append(UInt8(truncatingIfNeeded: v >> 24))
}

func wavPutU16(_ f: inout [UInt8], _ v: UInt16) {
    f.append(UInt8(truncatingIfNeeded: v))
    f.append(UInt8(truncatingIfNeeded: v >> 8))
}

func wavBytes(_ pcm: [Float], rate: Int) -> [UInt8] {
    var f = [UInt8]()
    let dataBytes = UInt32(pcm.count * 2)
    f.append(contentsOf: Array("RIFF".utf8))
    wavPutU32(&f, 36 + dataBytes)
    f.append(contentsOf: Array("WAVE".utf8))
    f.append(contentsOf: Array("fmt ".utf8))
    wavPutU32(&f, 16)
    wavPutU16(&f, 1)                        // PCM
    wavPutU16(&f, 1)                        // mono
    wavPutU32(&f, UInt32(rate))
    wavPutU32(&f, UInt32(rate * 2))         // byte rate
    wavPutU16(&f, 2)                        // block align
    wavPutU16(&f, 16)                       // bits
    f.append(contentsOf: Array("data".utf8))
    wavPutU32(&f, dataBytes)
    for sample in pcm {
        wavPutU16(&f, UInt16(bitPattern: wavClamp(sample)))
    }
    return f
}

@MainActor func probeTTS() throws {
    let listing = rawArgs.contains("--tts-voices")
    let outPath = stripValue(&rawArgs, "--tts-out") ?? "tts.wav"
    let voiceName = stripValue(&rawArgs, "--tts-voice")
    let speed = stripValue(&rawArgs, "--tts-speed", Float.init) ?? 1.0
    // @path reads the text from a file, matching the turn-argument convention.
    let text = stripValue(&rawArgs, "--tts").map { arg in
        arg.hasPrefix("@")
            ? ((try? String(contentsOfFile: String(arg.dropFirst()),
                            encoding: .utf8)) ?? arg)
            : arg
    }
    if listing {
        for v in Speech.voices { err("  \(v.name)\n") }
        exit(0)
    }
    // --tts-md runs the text through the app's chunker first, so what is
    // synthesized is exactly what a spoken reply would be -- code blocks and
    // tables described rather than read. --tts stays raw, because the byte
    // gate compares against a reference that has no chunker.
    let markdown = stripValue(&rawArgs, "--tts-md").map { arg in
        arg.hasPrefix("@")
            ? ((try? String(contentsOfFile: String(arg.dropFirst()),
                            encoding: .utf8)) ?? arg)
            : arg
    }
    let chunked = markdown.map { source -> String in
        var chunker = SpeakableText()
        var segments = chunker.push(source)
        segments.append(contentsOf: chunker.finish())
        for s in segments { err("  say: \(s)\n") }
        // One segment per line: the engine's own splitter then gives each the
        // inter-sentence pause a spoken reply would have between them.
        return segments.joined(separator: "\n")
    }
    if let text = chunked ?? text {
        let voice = voiceName.flatMap { name in Speech.voice(named: name) }
        if voiceName != nil && voice == nil {
            err("unknown voice '\(voiceName!)' (try --tts-voices)\n")
            exit(2)
        }
        let t0 = Date()
        let speech = Speech()
        if speech == nil {
            err("speech engine unavailable (bundled resources missing)\n")
            exit(1)
        }
        let load = Date().timeIntervalSince(t0)
        let t1 = Date()
        let pcm = speech!.synthesize(text, voice: voice, speed: speed)
        let synth = Date().timeIntervalSince(t1)
        if pcm.isEmpty {
            err("no audio produced\n")
            exit(1)
        }
        let url = URL(fileURLWithPath: outPath)
        try Data(wavBytes(pcm, rate: Speech.sampleRate)).write(to: url)
        let seconds = Double(pcm.count) / Double(Speech.sampleRate)
        err(String(format:
            "wrote %@  (%.2fs audio, voice %@, speed %.2f)\n"
            + "[tts] load %.2fs, synth %.2fs, %.1fx realtime\n",
            outPath, seconds, (voice ?? Speech.defaultVoice).name, speed,
            load, synth, synth > 0 ? seconds / synth : 0))
        exit(0)
    }
}
