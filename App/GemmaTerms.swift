import LLM
import SwiftUI

// A consent gate for a model whose licence binds the END USER, shown before
// that model is downloaded.
//
// WHY it can exist and name nothing: a licence that asks something of the
// user is a property of the model, not of the app, and it can arrive with any
// future release. Keeping the gate armed and empty costs one lookup and means
// the next such model needs a name here rather than a screen built under time
// pressure.
//
// Deliberately NOT a reproduction of any licence: linking the canonical text
// is the only version that cannot go stale here.
enum GemmaTerms {
    // EMPTY, and that is a finding rather than an oversight. Every gemma-4
    // checkpoint Google publishes is Apache-2.0 and UNGATED on the Hub --
    // license_link resolves to a page titled "Apache License 2.0", and the
    // gemma-3 repos beside them are still `gated: manual` under `license:
    // gemma`, which is what makes the change deliberate rather than sloppy
    // metadata. Apache-2.0 asks nothing of an end user, so there is nothing
    // left to consent to.
    //
    // What it DOES ask of a redistributor is met elsewhere: the licence text,
    // the attribution and the statement that these weights are modified (they
    // are repacked from the QAT checkpoints) belong with the credits, not
    // behind a gate.
    static let modelNames: Set<String> = []

    static func applies(to name: String) -> Bool { modelNames.contains(name) }

    static let termsURL = URL(string: "https://ai.google.dev/gemma/terms")!
    static let policyURL =
        URL(string: "https://ai.google.dev/gemma/prohibited_use_policy")!

    // Persisted per model, so the gate is shown once and a later re-download
    // (or a delete and fetch again) does not re-ask.
    private static let key = "gemmaTermsAccepted"

    static var accepted: Bool {
        UserDefaults.standard.bool(forKey: key)
    }

    static func accept() {
        UserDefaults.standard.set(true, forKey: key)
    }
}

// Shown in place of the download-consent panel when the chosen model carries
// the Gemma terms. Agreeing leads INTO the normal size/consent panel rather
// than starting a download itself: the licence and the "this costs you 2.7 GB"
// question are different consents and should not be collapsed into one tap.
struct GemmaTermsView: View {
    let model: String
    let onAgree: () -> Void
    let onCancel: () -> Void

    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "doc.text")
                .appFont(.largeTitle)
                .foregroundStyle(.secondary)
            Text("Gemma Terms of Use").appFont(.title2).bold()
            Text("\(Models.display(model)) is provided by "
               + "Google under the Gemma Terms of Use, not the open-source "
               + "licence the other models use. By continuing you agree to "
               + "those terms and to the Prohibited Use Policy, which "
               + "restrict what the model may be used for.")
                .multilineTextAlignment(.center)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            VStack(spacing: 6) {
                Link("Gemma Terms of Use", destination: GemmaTerms.termsURL)
                Link("Prohibited Use Policy", destination: GemmaTerms.policyURL)
            }
            .appFont(.callout)
            Text("Gemma is a Google model. This app is not affiliated with "
               + "or endorsed by Google.")
                .appFont(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            VStack(spacing: 14) {
                Button("Agree and Continue", action: onAgree)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .keyboardShortcut(.defaultAction)
                Button("Cancel", action: onCancel)
                    .controlSize(.large)
                    .keyboardShortcut(.cancelAction)
            }
            .padding(.top, 8)
        }
        .padding(32)
        .frame(maxWidth: 460)
    }
}
